"""
author: Florian Krach
based on code copied from:
Improved Wasserstein-GAN: https://github.com/igul222/improved_wgan_training
with their code taken from:
From https://github.com/openai/improved-gan/blob/master/inception_score/model.py
Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import math
import sys
import settings

MODEL_DIR = 'tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads = 1


# this function loads the graph every time again -> not good if called more often
def get_inception_score(images, splits=10, model_dir=MODEL_DIR, softmax=None):
    assert (type(images) == list)
    assert (type(images[0]) == np.ndarray)
    assert (len(images[0].shape) == 3)
    assert (np.max(images[0]) > 10)
    assert (np.min(images[0]) >= 0.0)
    inps = []
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
    bs = 1  # FK: changed, see: https://github.com/openai/improved-gan/blob/master/inception_score/model.py

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = DATA_URL.split('/')[-1]
    filename1 = 'classify_image_graph_def.pb'
    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(model_dir+'/'+filename1):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded {} {} bytes.'.format(filename, statinfo.st_size))
        tarfile.open(filepath, 'r:gz').extractall(model_dir)

    if softmax is None:
        with tf.gfile.GFile(os.path.join(
                model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            print('successfully loaded Inception model')

            _ = tf.import_graph_def(graph_def, name='')
        with tf.Session(config=config) as sess:
            pool3 = sess.graph.get_tensor_by_name('pool_3:0')
            w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
            logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)  # FK: changed from: ... tf.squeeze(pool3) ...
            softmax = tf.nn.softmax(logits)
            print('loaded softmax')

            # if settings.send_email:
            #     send_email.send_email(subject='start computing IS', body='', file_names=None)

            preds = []
            n_batches = int(math.ceil(float(len(inps)) / float(bs)))
            for i in range(n_batches):
                if not settings.euler:
                    # progress bar
                    sys.stdout.write('\r>> Compute IS: %.1f%%' % (float(i) / float(n_batches) * 100.0))
                    sys.stdout.flush()
                inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
                inp = np.concatenate(inp, 0)
                pred = sess.run(softmax, {'ExpandDims:0': inp})
                preds.append(pred)
    else:
        with tf.Session(config=config) as sess:
            print('softmax already exists')

            preds = []
            n_batches = int(math.ceil(float(len(inps)) / float(bs)))
            for i in range(n_batches):
                if not settings.euler:
                    # progress bar
                    sys.stdout.write('\r>> Compute IS: %.1f%%' % (float(i) / float(n_batches) * 100.0))
                    sys.stdout.flush()
                inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
                inp = np.concatenate(inp, 0)
                pred = sess.run(softmax, {'ExpandDims:0': inp})
                preds.append(pred)

    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores), softmax




# # Call this function with list of images. Each of elements should be a
# # numpy array with values ranging from 0 to 255.
# def get_inception_score(images, splits=10):
#     assert (type(images) == list)
#     assert (type(images[0]) == np.ndarray)
#     assert (len(images[0].shape) == 3)
#     assert (np.max(images[0]) > 10)
#     assert (np.min(images[0]) >= 0.0)
#     inps = []
#     for img in images:
#         img = img.astype(np.float32)
#         inps.append(np.expand_dims(img, 0))
#     bs = 1  # FK: changed, see: https://github.com/openai/improved-gan/blob/master/inception_score/model.py
#     with tf.Session(config=config) as sess:
#         preds = []
#         n_batches = int(math.ceil(float(len(inps)) / float(bs)))
#         for i in range(n_batches):
#             if not settings.euler:
#                 # progress bar
#                 sys.stdout.write('\r>> Compute IS: %.1f%%' % (float(i) / float(n_batches) * 100.0))
#                 sys.stdout.flush()
#             # sys.stdout.write(".")
#             # sys.stdout.flush()
#             inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
#             inp = np.concatenate(inp, 0)
#             pred = sess.run(softmax, {'ExpandDims:0': inp})
#             preds.append(pred)
#         preds = np.concatenate(preds, 0)
#         scores = []
#         for i in range(splits):
#             part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
#             kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
#             kl = np.mean(np.sum(kl, 1))
#             scores.append(np.exp(kl))
#         return np.mean(scores), np.std(scores)
#
#
# # This function is called automatically.
# def _init_inception():
#     global softmax
#     if not os.path.exists(MODEL_DIR):
#         os.makedirs(MODEL_DIR)
#     filename = DATA_URL.split('/')[-1]
#     filename1 = 'classify_image_graph_def.pb'
#     filepath = os.path.join(MODEL_DIR, filename)
#     if not os.path.exists(MODEL_DIR+'/'+filename1):
#         def _progress(count, block_size, total_size):
#             sys.stdout.write('\r>> Downloading %s %.1f%%' % (
#                 filename, float(count * block_size) / float(total_size) * 100.0))
#             sys.stdout.flush()
#
#         filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
#         print()
#         statinfo = os.stat(filepath)
#         print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
#         tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
#
#     with tf.gfile.GFile(os.path.join(
#             MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         print('successfully loaded Inception model')
#
#         _ = tf.import_graph_def(graph_def, name='')
#     # Works with an arbitrary minibatch size.
#     with tf.Session(config=config) as sess:
#         pool3 = sess.graph.get_tensor_by_name('pool_3:0')
#         # # the following doesn't do anything (see if statement in the end, which is never true) => deleted
#         # ops = pool3.graph.get_operations()
#         # for op_idx, op in enumerate(ops):
#         #     for o in op.outputs:
#         #         shape = o.get_shape()
#         #         shape = [s.value for s in shape]
#         #         new_shape = []
#         #         for j, s in enumerate(shape):
#         #             if s == 1 and j == 0:
#         #                 new_shape.append(None)
#         #             else:
#         #                 new_shape.append(s)
#         #         o.set_shape(tf.TensorShape(new_shape))  # FK: changed from: o._shape = tf.TensorShape(new_shape)
#         #         shape1 = o.get_shape()
#         #         shape1 = [s.value for s in shape1]
#         #         print(shape, shape1)
#         #         if shape != shape1:
#         #             print('\nhere something changed\n')
#         w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
#         logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)  # FK: changed from: ... tf.squeeze(pool3) ...
#         softmax = tf.nn.softmax(logits)
#         sess.close()
#
#
# if softmax is None:
#     _init_inception()
