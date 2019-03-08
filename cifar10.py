"""
code copied from:
Improved Wasserstein-GAN, https://github.com/igul222/improved_wgan_training
"""

import numpy as np

import os
import urllib
import gzip
import cPickle as pickle


# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html
# and fill in the path to the extracted files in settings.py as filepath_cifar10


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']

def cifar_generator(filenames, batch_size, data_dir):
    all_data = []
    all_labels = []
    for filename in filenames:
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # change the format from BCHW to BHWC
    images = images.reshape([-1, 3, 32, 32])
    images = images.transpose((0, 2, 3, 1))
    images = images.reshape((-1, 3*32*32))

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(batch_size, data_dir):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir),
        cifar_generator(['test_batch'], batch_size, data_dir)
    )