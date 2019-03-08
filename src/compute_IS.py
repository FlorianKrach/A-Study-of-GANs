"""
author: Florian Krach
the function to compute the inception score is a modified version of the code from:
Improved Wasserstein-GAN: https://github.com/igul222/improved_wgan_training
with their code taken from:
From https://github.com/openai/improved-gan/blob/master/inception_score/model.py
Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
"""
import numpy as np
import math
import tensorflow as tf
from tensorflow.contrib import layers
import save_images
import settings
import send_email
import os, sys
import pandas as pd
import time
from joblib import Parallel, delayed
import os.path
import tarfile
from six.moves import urllib


if settings.euler:
    N_CPUS_TF = 1  # set to None to use value of settings.py
    N_CPUS_PARALLEL = 8  # set to None to use value of settings.py
else:
    N_CPUS_TF = 3  # set to None to use value of settings.py
    N_CPUS_PARALLEL = 1  # set to None to use value of settings.py

MODEL_DIR = 'tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

config = tf.ConfigProto()
if N_CPUS_TF is None:
    number_cpus_tf = settings.number_cpus
else:
    number_cpus_tf = N_CPUS_TF
config.intra_op_parallelism_threads = number_cpus_tf
config.inter_op_parallelism_threads = number_cpus_tf


# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10, model_dir=MODEL_DIR, path=''):
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
        print
        statinfo = os.stat(filepath)
        print 'Succesfully downloaded {} {} bytes.'.format(filename, statinfo.st_size)
        tarfile.open(filepath, 'r:gz').extractall(model_dir)
    with tf.gfile.GFile(os.path.join(
            model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        print 'successfully loaded Inception model'

        _ = tf.import_graph_def(graph_def, name='')
    with tf.Session(config=config) as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)  # FK: changed from: ... tf.squeeze(pool3) ...
        softmax = tf.nn.softmax(logits)

        # if settings.send_email:
        #     send_email.send_email(subject='start computing IS', body='', file_names=None)

        preds = []
        n_batches = int(math.ceil(float(len(inps)) / float(bs)))
        for i in range(n_batches):
            if not settings.euler:
                # progress bar
                sys.stdout.write('\r>> Compute IS: %.1f%%' % (float(i) / float(n_batches) * 100.0))
                sys.stdout.flush()
            else:
                prog = (float(i) / float(n_batches))
                np.save(file=path+'progress_IS.npy', arr=prog)
                np.savetxt(path+'progress_IS.csv', [prog])
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
        return np.mean(scores), np.std(scores)


def generator(z, n_features_first=256, n_features_reduction_factor=2, min_features=64,
              BN=True, power=5, init_method='He', n_features_image=1, extra_layer=False):

    if BN:
        normalizer = layers.batch_norm
    else:
        normalizer = None

    if init_method in ['He']:
        init = layers.variance_scaling_initializer()
    else:
        init = layers.xavier_initializer()

    # the layers use relu activations (default)
    with tf.variable_scope('generator'):
        # first layer (fully connected) -> [B, 4, 4, n_features]
        z = layers.fully_connected(z, num_outputs=4*4*n_features_first, trainable=True,
                                   normalizer_fn=normalizer, weights_initializer=init, activation_fn=tf.nn.relu,)
        z = tf.reshape(z, [-1, 4, 4, n_features_first])  # we use the dimensions as NHWC resp. BHWC

        # middle layers (convolutions) -> [B, 4*(2**(power-3)), 4*(2**(power-3)), n_features2]
        for i in range(power-3):
            n_out = max(int(n_features_first/(n_features_reduction_factor**(i+1))), min_features)
            z = layers.conv2d_transpose(z, num_outputs=n_out, kernel_size=4, padding='SAME',
                                        stride=2, trainable=True, normalizer_fn=normalizer,
                                        weights_initializer=init, activation_fn=tf.nn.relu)

        if extra_layer:
            n_out = max(int(n_features_first / (n_features_reduction_factor ** (power-3))), min_features)
            z = layers.conv2d_transpose(z, num_outputs=n_out, kernel_size=4, padding='SAME',
                                        stride=1, trainable=True, normalizer_fn=normalizer,
                                        weights_initializer=init, activation_fn=tf.nn.relu)

        # last layer (convolution) -> [B, (2**power), (2**power), n_features_last] -> [B, n_features_last*(2**power)**2]
        z = layers.conv2d_transpose(z, num_outputs=n_features_image, kernel_size=4, stride=2, padding='SAME',
                                    activation_fn=tf.nn.tanh, trainable=True, weights_initializer=init)
        return layers.flatten(z)


def discriminator(x, reuse, n_features_last=256, n_features_increase_factor=2, min_features=64,
                  d_BN=None, power=5, n_features_image=1, init_method='He'):

    if d_BN == 'LN':
        normalizer = layers.layer_norm
    elif d_BN == 'BN':
        normalizer = layers.batch_norm
    else:
        normalizer = None

    if init_method in ['He']:
        init = layers.variance_scaling_initializer()
    else:
        init = layers.xavier_initializer()

    with tf.variable_scope('discriminator', reuse=reuse):
        # first reshape: [B,  n_features_image*(2**power)**2] -> [B, (2**power), (2**power), n_features_image]
        size = 2**power
        x = tf.reshape(x, shape=[-1, size, size, n_features_image])
        # first layers (convolutions): [B, (2**(power)), (2**(power)), n_features_picture] -> [B, 4, 4, n_features_last]
        for i in range(power - 2):
            n_out = max(int(n_features_last / (n_features_increase_factor ** (power - i - 3))), min_features)
            x = layers.conv2d(x, num_outputs=n_out, kernel_size=4, stride=2, trainable=True, normalizer_fn=normalizer,
                              activation_fn=tf.nn.leaky_relu, weights_initializer=init, padding='SAME')

        # last layer: [B, 4, 4, n_features_last] -> [B, size] -> [B, n_classes] with probability values
        last = layers.flatten(x)
        # softmax is applied in the loss function directly
        prediction_without_softmax = layers.fully_connected(last, num_outputs=1, activation_fn=None,
                                                            trainable=True, weights_initializer=init)

        return prediction_without_softmax, last


def compute_IS(data, input_dim, batch_size, learning_rate,
               n_features_first_g, n_features_reduction_factor,
               min_features, n_features_last_d, extra_layer_g, d_freq, d_steps,
               architecture, init_method, BN, d_BN, JL_dim, n_projections,
               path='JLSWGAN', n_IS=50000):

    # -------------------------------------------------------
    # setting for sending emails and getting statistics
    send = settings.send_email

    # -------------------------------------------------------
    # architecture default
    use_JL = True
    if architecture not in ['SWGAN']:
        architecture = 'JLSWGAN'
    if architecture == 'SWGAN':
        use_JL = False
        JL_error = None
        JL_dim = None

    if d_BN not in ['LN', 'BN']:
        d_BN = None

    # -------------------------------------------------------
    # data set default
    assert(data == 'cifar10')

    picture_size = 32 * 32 * 3
    picture_dim = [-1, 32, 32, 3]
    power = 5
    n_features_image = 3
    d_last_layer_size = 4 * 4 * n_features_last_d

    # -------------------------------------------------------
    # init_method default
    if init_method not in ['uniform']:
        init_method = 'He'

    # -------------------------------------------------------
    # JL_dim:
    if JL_dim is None and use_JL:
        use_JL = False
        architecture = 'SWGAN'
    if use_JL and JL_dim >= picture_size:
        use_JL = False
        architecture = 'SWGAN'
        JL_dim = None

    # -------------------------------------------------------
    # get folder name
    dir1 = path
    directory = dir1 + str(data) + '_' + \
                str(input_dim) + '_' + str(batch_size) + '_' + str(learning_rate) + '_' + \
                str(n_features_first_g) + '_' + str(n_features_reduction_factor) + '_' + str(min_features) + '_' + \
                str(n_features_last_d) + '_' + str(extra_layer_g) + '_' + str(d_freq) + '_' + str(d_steps) + '_' + \
                str(architecture) + '_' + str(init_method) + '_' + str(BN) + '_' + str(d_BN) + '_' + \
                str(JL_dim) + '_' + str(n_projections) + '/'
    model_dir = directory + 'model/'

    # -------------------------------------------------------
    # initialize a TF session
    session = tf.Session(config=config)

    def build_model():
        """
        - function to build the model
        """
        with tf.name_scope('placeholders'):
            real_data_int = tf.placeholder(tf.int32, shape=[None, picture_size])
            x_true = 2 * ((tf.cast(real_data_int, tf.float32) / 255.) - .5)
            z = tf.placeholder(tf.float32, [None, input_dim])
            if use_JL:
                JL = tf.placeholder(tf.float32, [d_last_layer_size, JL_dim])
                P_non_normalized = tf.placeholder(tf.float32, [JL_dim, n_projections])
                P_non_normalized_SWD = tf.placeholder(tf.float32, [picture_size, n_projections])
            else:
                JL = None
                P_non_normalized = tf.placeholder(tf.float32, [d_last_layer_size, n_projections])
                P_non_normalized_SWD = tf.placeholder(tf.float32, [picture_size, n_projections])

        x_generated = generator(z, n_features_first=n_features_first_g,
                                n_features_reduction_factor=n_features_reduction_factor, min_features=64,
                                BN=BN, power=power, extra_layer=extra_layer_g,
                                init_method=init_method, n_features_image=n_features_image)

        d_pred_true, d_last_true = discriminator(x_true, reuse=False, n_features_last=n_features_last_d,
                                                 n_features_increase_factor=n_features_reduction_factor,
                                                 min_features=min_features, d_BN=d_BN, power=power,
                                                 n_features_image=n_features_image, init_method=init_method)
        d_pred_gen, d_last_gen = discriminator(x_generated, reuse=True, n_features_last=n_features_last_d,
                                               n_features_increase_factor=n_features_reduction_factor,
                                               min_features=min_features, d_BN=d_BN, power=power,
                                               n_features_image=n_features_image, init_method=init_method)

        # define generator loss (big part taken from SWG)
        with tf.name_scope('g_loss'):
            # apply the Johnson-Lindenstrauss map, if wanted, to the flattened array
            if use_JL:
                JL_true = tf.matmul(d_last_true, JL)
                JL_gen = tf.matmul(d_last_gen, JL)
            else:
                JL_true = d_last_true
                JL_gen = d_last_gen

            # next project the samples (images). After being transposed, we have tensors
            # of the format: [[projected_image1_proj1, projected_image2_proj1, ...],
            #                 [projected_image1_proj2, projected_image2_proj2, ...],...]
            # Each row has the projections along one direction. This makes it easier for the sorting that follows.
            # first normalize the random normal vectors to lie in the sphere
            P = tf.nn.l2_normalize(P_non_normalized, axis=0)

            projected_true = tf.transpose(tf.matmul(JL_true, P))
            projected_fake = tf.transpose(tf.matmul(JL_gen, P))

            sorted_true, true_indices = tf.nn.top_k(input=projected_true, k=batch_size)
            sorted_fake, fake_indices = tf.nn.top_k(input=projected_fake, k=batch_size)

            # For faster gradient computation, we do not use sorted_fake to compute
            # loss. Instead we re-order the sorted_true so that the samples from the
            # true distribution go to the correct sample from the fake distribution.

            # It is less expensive (memory-wise) to rearrange arrays in TF.
            # Flatten the sorted_true from dim [n_projections, batch_size].
            flat_true = tf.reshape(sorted_true, [-1])

            # Modify the indices to reflect this transition to an array.
            # new index = row + index
            rows = np.asarray([batch_size * np.floor(i * 1.0 / batch_size) for i in range(n_projections * batch_size)])
            rows = rows.astype(np.int32)
            flat_idx = tf.reshape(fake_indices, [-1, 1]) + np.reshape(rows, [-1, 1])

            # The scatter operation takes care of reshaping to the rearranged matrix
            shape = tf.constant([batch_size * n_projections])
            rearranged_true = tf.reshape(tf.scatter_nd(flat_idx, flat_true, shape), [n_projections, batch_size])

            generator_loss = tf.reduce_mean(tf.square(projected_fake - rearranged_true))

        # get the sliced Wasserstein distance (SWD) (since SWD and JLSWD are not comparable)
        with tf.name_scope('SWD'):
            P_SWD = tf.nn.l2_normalize(P_non_normalized_SWD, axis=0)

            projected_true_SWD = tf.transpose(tf.matmul(x_true, P_SWD))
            projected_fake_SWD = tf.transpose(tf.matmul(x_generated, P_SWD))

            sorted_true_SWD, true_indices_SWD = tf.nn.top_k(input=projected_true_SWD, k=batch_size)
            sorted_fake_SWD, fake_indices_SWD = tf.nn.top_k(input=projected_fake_SWD, k=batch_size)

            flat_true_SWD = tf.reshape(sorted_true_SWD, [-1])
            flat_idx_SWD = tf.reshape(fake_indices_SWD, [-1, 1]) + np.reshape(rows, [-1, 1])

            rearranged_true_SWD = tf.reshape(tf.scatter_nd(flat_idx_SWD, flat_true_SWD, shape),
                                             [n_projections, batch_size])

            SWD = tf.reduce_mean(tf.square(projected_fake_SWD - rearranged_true_SWD))

        # define the discriminator loss
        with tf.name_scope('d_loss'):
            d_true_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_pred_true), logits=d_pred_true)
            d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_pred_gen), logits=d_pred_gen)
            discriminator_loss = tf.reduce_mean(d_true_loss + d_fake_loss)

        with tf.name_scope('g_optimizer'):
            generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
            g_train = g_optimizer.minimize(generator_loss, var_list=generator_vars)

        with tf.name_scope('d_optimizer'):
            discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
            d_train = d_optimizer.minimize(discriminator_loss, var_list=discriminator_vars)

        return real_data_int, z, x_generated, JL, P_non_normalized, P_non_normalized_SWD, SWD, g_train, d_train

    # -------------------------------------------------------
    # build the model
    real_data_int, z, x_generated, JL, P_non_normalized, P_non_normalized_SWD, SWD, g_train, d_train = build_model()

    # -------------------------------------------------------
    # load the model
    saver = tf.train.Saver(max_to_keep=1)
    iterations_trained = 0

    # restore the model:
    saver.restore(sess=session, save_path=model_dir + 'saved_model')
    iterations_trained = int(np.loadtxt(fname=model_dir + 'epochs.csv'))
    print 'loaded model, which was trained for {} iterations'.format(iterations_trained)
    model_config = pd.read_csv(filepath_or_buffer=directory + 'model_config.csv', index_col=0, header=0)
    print
    print model_config
    print

    # -------------------------------------------------------
    # compute IS
    compute = True
    if os.path.isfile(path=directory + 'inception_score.csv'):
        IS_table_ = pd.read_csv(filepath_or_buffer=directory + 'inception_score.csv', index_col=0, header=0)
        iterations = IS_table_['iteration'].values
        n_ISs = IS_table_['n_IS'].values
        for i in range(len(iterations)):
            if iterations_trained == iterations[i] and n_IS == n_ISs[i]:
                compute = False
                send = False
                break
    else:
        IS_table_ = pd.DataFrame(data=None, index=None, columns=['inception_score', 'std', 'iteration'])

    if compute:
        print 'generate samples'
        all_samples = []
        for i in xrange(n_IS / 100):
            if not settings.euler:
                # progress bar
                sys.stdout.write(
                    '\r>> generate samples %.1f%%' % (float(i) / float(n_IS / 100) * 100))
                sys.stdout.flush()
            z_input = np.random.randn(100, input_dim)
            all_samples.append(session.run(x_generated, feed_dict={z: z_input}))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples + 1.) * (255. / 2)).astype('int32')
        all_samples = all_samples.reshape((-1, 32, 32, 3))
        # save_images.save_images(all_samples[:100], 'test_IS.png')
        print '\ncompute inception score'
        IS_mean, IS_std = get_inception_score(list(all_samples), path=directory)
        print
        print 'IS = {} +/- {}'.format(IS_mean, IS_std)

        IS_table = pd.DataFrame(data=[[IS_mean, IS_std, iterations_trained, n_IS]],
                                index=None, columns=['inception_score', 'std', 'iteration', 'n_IS'])
        IS_table = pd.concat([IS_table_, IS_table])
        IS_table.to_csv(path_or_buf=directory + 'inception_score.csv')
    else:
        print 'inception score already computed'
    print '\n'+'='*120+'\n'

    # -------------------------------------------------------
    # after training close the session
    session.close()
    tf.reset_default_graph()

    # -------------------------------------------------------
    # when training is done send email
    if send:
        subject = 'JL-SWGAN (cifar10) inception score computation finished'
        body = '{}\n\n'.format(model_config)
        files = [directory + 'model_config.csv', directory + 'inception_score.csv']
        send_email.send_email(subject=subject, body=body, file_names=files)

    return 0


def parallel_training(parameters=None, path='', n_IS=50000, nb_jobs=-1):
    """
    :param parameters: an array of arrays with all the parameters
    :param nb_jobs: number of jobs that run parallel, -1 means all available cpus are used
    :return:
    """

    results = Parallel(n_jobs=nb_jobs)(delayed(compute_IS)(data, input_dim, batch_size, learning_rate,
                                                           n_features_first_g, n_features_reduction_factor,
                                                           min_features, n_features_last_d, extra_layer_g, d_freq, d_steps,
                                                           architecture, init_method, BN, d_BN, JL_dim, n_projections,
                                                           path, n_IS)
                                       for data, input_dim, batch_size, learning_rate, iterations,
                                           fixed_noise_size, n_features_first_g, n_features_reduction_factor,
                                           min_features, n_features_last_d, extra_layer_g, d_freq, d_steps,
                                           architecture, init_method, BN, d_BN, JL_dim, n_projections in parameters)

    if settings.send_email:
        subject = 'parallel: JL-SWGAN (cifar10) inception score computation finished'
        send_email.send_email(subject=subject, body='', file_names=None)
    return 0


if __name__ == '__main__':
    param_array = settings.JLSWGAN_param_array1_0

    if N_CPUS_PARALLEL is None:
        nb_jobs = settings.number_parallel_jobs
    else:
        nb_jobs = N_CPUS_PARALLEL

    if settings.euler:
        path = '/cluster/scratch/fkrach/JLSWGAN/'
        n_IS = 50000
    else:
        path = '/Users/Flo/Desktop/Training results/JLSWGAN/cifar10/'
        n_IS = 1000

    # parallel_training(param_array, path=path, n_IS=n_IS, nb_jobs=nb_jobs)

    data, input_dim, batch_size, learning_rate, iterations, \
    fixed_noise_size, n_features_first_g, n_features_reduction_factor,\
    min_features, n_features_last_d, extra_layer_g, d_freq, d_steps,\
    architecture, init_method, BN, d_BN, JL_dim, n_projections = param_array[0]

    compute_IS(data, input_dim, batch_size, learning_rate,
               n_features_first_g, n_features_reduction_factor,
               min_features, n_features_last_d, True, d_freq, d_steps,
               'SWGAN', init_method, BN, 'BN', None, 10000,
               path=path, n_IS=10000)

    compute_IS(data, input_dim, batch_size, learning_rate,
               n_features_first_g, n_features_reduction_factor,
               min_features, n_features_last_d, True, d_freq, d_steps,
               'JLSWGAN', init_method, BN, 'BN', 512, 10000,
               path=path, n_IS=10000)


