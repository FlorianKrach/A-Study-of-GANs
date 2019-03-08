"""
author: Florian Krach
used parts of the code of the following implementations:
- Improved Wasserstein-GAN, https://github.com/igul222/improved_wgan_training
- Minimal implementation of Wasserstein GAN for MNIST, https://github.com/adler-j/minimal_wgan
- SWG, https://github.com/ishansd/swg
"""

import numpy as np
import math
import tensorflow as tf
from tensorflow.contrib import layers
import save_images
import settings
import send_email
import os, sys
from subprocess import call
import tensorflow.contrib.slim as slim
import pandas as pd
import time
from joblib import Parallel, delayed
import cifar10, celeba,preprocessing_mnist
import inception_score


BATCH_SIZE = 64  # Batch size
ITERS = 200000  # How many generator iterations to train for
INPUT_DIM = 128
N_FEATURES_FIRST = 256  # FK: has to be divisible by 4
FIXED_NOISE_SIZE = 128

D_FREQ = 1
D_STEPS = 1

COMPUTE_IS = True
N_IS = 10000  # number of samples to compute inception score
IS_FREQ = 1000  # the frequency to compute the inception score (in iterations), should be multiple of 100
START_COMPUTING_LOSS = 0  # after how many iterations to start to compute loss
STEP_SIZE_LOSS_COMPUTATION = 1000

if settings.euler:
    N_CPUS_TF = 1  # set to None to use value of settings.py
    N_CPUS_PARALLEL = 12  # set to None to use value of settings.py
else:
    N_CPUS_TF = 1  # set to None to use value of settings.py
    N_CPUS_PARALLEL = 2  # set to None to use value of settings.py


# print locals().copy()


# FK: model summary of
def model_summary(scope):
    model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    return slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def memory():
    """
    this function can be used to get the used memory of the python process
    it was copied from https://stackoverflow.com/questions/276052/how-to-get-current-cpu-and-ram-usage-in-python
    """
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB
    return memoryUse


def generator(z, n_features_first=N_FEATURES_FIRST, n_features_reduction_factor=2, min_features=64,
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


def discriminator(x, reuse, n_features_last=N_FEATURES_FIRST, n_features_increase_factor=2, min_features=64,
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


def train(data='mnist', input_dim=INPUT_DIM, batch_size=BATCH_SIZE,
          learning_rate=1e-4, iterations=ITERS, fixed_noise_size=FIXED_NOISE_SIZE,
          n_features_first_g=N_FEATURES_FIRST, n_features_reduction_factor=2, min_features=64,
          n_features_last_d=N_FEATURES_FIRST, extra_layer_g=True, d_freq=1, d_steps=1,
          architecture='JLSWGAN', init_method='He', BN=True, d_BN='LN',
          JL_dim=None, n_projections=10000,
          load_saved=True):
    """
    - this is the function to use to train an Johnson-Lindenstrauss Generative Adversarial Network model which uses the
      sliced Wasserstein-2 distance as objective funtion (JL-SWGAN), with the configuration given by the
      parameters
    - the function computes losses and auto-saves the model every STEP_SIZE_LOSS_COMPUTATION steps and automatically
      resumes training where it stopped (when load_saved=True)
    - automatically saves the model with the best inception score in model_dir/best/

    :param input_dim: the dimension of the latent space -> Z
    :param batch_size: the batch size, should be a divisor of 50k
    :param learning_rate:
    :param n_features_first_g: the number of feature maps in the first step of the generator
    :param n_features_last_d: the number of feature maps in last step of the discriminator
    :param extra_layer_g: whether an extra conv layer with stride one is used in end of generator
    :param d_freq: the frequency that the discriminator is updated (every d_freq updates of generator)
    :param d_steps: the number of discriminator updates, when the discriminator is updated
    :param iterations: the number of iterations to train for (number epochs: iterations*batch_size/train_data_samples)
    :param fixed_noise_size: the number of pictures that is generated during training for visual progress
    :param n_features_reduction_factor: integer, e.g.: 1: use same number of feature-maps everywhere, 2: half the number
           of feature-maps in every step
    :param min_features: the minimal number of features
    :param architecture: right now only supports 'JLSWGAN', 'SWGAN', defaults to 'JLSWGAN'
    :param init_method: the method with which the variables are initialized, support: 'uniform', 'He', defaults to 'He'
    :param BN: shall batch normalization be used
    :param JL_dim: the target dimension of the JL mapping
    :param n_projections: number of random projections in sliced Wasserstein-2 distance
    :param data: the data set which shall be used for training: celebA32, celebA32_bw, cifar10, mnist
    :param load_saved: whether an already existing training progress shall be loaded to continue there (if one exists)
    :return:
    """

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
    if data not in ['cifar10', 'celebA32', 'celebA32_bw', 'mnist', 'celebA64']:
        data = 'cifar10'
    if data in ['celebA32_bw', 'mnist']:
        picture_size = 32*32
        picture_dim = [-1, 32, 32]
        power = 5
        n_features_image = 1
    elif data in ['cifar10', 'celebA32']:
        picture_size = 32*32*3
        picture_dim = [-1, 32, 32, 3]
        power = 5
        n_features_image = 3
    elif data in ['celebA64']:
        picture_size = 64*64 * 3
        picture_dim = [-1, 64, 64, 3]
        power = 6
        n_features_image = 3
    print 'data set: {}'.format(data)
    print

    d_last_layer_size = 4*4*n_features_last_d

    # -------------------------------------------------------
    # init_method default
    if init_method not in ['uniform']:
        init_method = 'He'

    # -------------------------------------------------------
    # JL_dim:
    if JL_dim is None and use_JL:
        use_JL = False
        architecture = 'SWGAN'
        print
        print 'architecture changed to SWGAN, since JL_dim was None'
        print

    if use_JL and JL_dim >= picture_size:
        use_JL = False
        architecture='SWGAN'
        JL_error = None
        JL_dim = None
        print
        print 'JL mapping is not used, since the target dimension was chosen bigger than the input dimension'
        print

    if use_JL:
        JL_error = np.round(np.sqrt(8*np.log(2*batch_size)/JL_dim), decimals=4)
    else:
        JL_error = None

    print 'JL_dim = {}'.format(JL_dim)
    print 'JL_error approximation = {}'.format(JL_error)
    print

    # -------------------------------------------------------
    # create unique folder name
    if settings.euler:
        dir1 = '/cluster/scratch/fkrach/JLSWGAN/'
    else:
        dir1 = 'JLSWGAN/'
    directory = dir1 + str(data) + '_' + \
                str(input_dim) + '_' + str(batch_size) + '_' + str(learning_rate) + '_' + \
                str(n_features_first_g) + '_' + str(n_features_reduction_factor) + '_' + str(min_features) + '_' + \
                str(n_features_last_d) + '_' + str(extra_layer_g) + '_' + str(d_freq) + '_' + str(d_steps) + '_' +\
                str(architecture) + '_' + str(init_method) + '_' + str(BN) + '_' + str(d_BN) + '_' + \
                str(JL_dim) + '_' + str(n_projections) + '/'
    samples_dir = directory+'samples/'
    model_dir = directory+'model/'

    # create directories if they don't exist
    if not os.path.isdir(dir1):
        call(['mkdir', dir1])

    if not os.path.isdir(directory):
        load_saved = False
        print 'make new directory:', directory
        print
        call(['mkdir', directory])
        call(['mkdir', samples_dir])
        call(['mkdir', model_dir])

    # if directories already exist, but model wasn't saved so far, set load_saved to False
    if 'training_progress.csv' not in os.listdir(directory):
        load_saved = False

    # -------------------------------------------------------
    # initialize a TF session
    config = tf.ConfigProto()
    if N_CPUS_TF is None:
        number_cpus_tf = settings.number_cpus
    else:
        number_cpus_tf = N_CPUS_TF
    config.intra_op_parallelism_threads = number_cpus_tf
    config.inter_op_parallelism_threads = number_cpus_tf
    session = tf.Session(config=config)

    # -------------------------------------------------------
    # convenience function to build the model
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

    # initialize variables using init_method
    session.run(tf.global_variables_initializer())

    # -------------------------------------------------------
    # For creating and saving samples (taken from IWGAN)
    fixed_noise = np.random.normal(size=(fixed_noise_size, input_dim)).astype('float32')

    def generate_image(frame):
        samples = session.run(x_generated, feed_dict={z: fixed_noise})
        samples = ((samples + 1.) * (255. / 2)).astype('uint8')  # transform linearly from [-1,1] to [0,255]
        samples = samples.reshape(picture_dim)
        save_images.save_images(
            samples,
            samples_dir + 'iteration_{}.png'.format(frame)
        )

    # -------------------------------------------------------
    # For calculating inception score
    softmax = None

    def get_inception_score(n=N_IS, softmax=softmax):
        all_samples = []
        for i in xrange(n / 100):
            z_input = np.random.randn(100, input_dim)
            all_samples.append(session.run(x_generated, feed_dict={z: z_input}))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples + 1.) * (255. / 2)).astype('int32')
        all_samples = all_samples.reshape((-1, 32, 32, 3))
        return inception_score.get_inception_score(list(all_samples), softmax=softmax)

    # -------------------------------------------------------
    # get the dataset as infinite generator
    if data == 'cifar10':
        data_dir = settings.filepath_cifar10
        train_gen, dev_gen = cifar10.load(batch_size, data_dir=data_dir)
        n_dev_samples = 10000
    elif data == 'celebA32':
        data_dir = settings.filepath_celebA32
        train_gen, dev_gen = celeba.load(batch_size, data_dir=data_dir, black_white=False)
        n_dev_samples = 10000
    elif data == 'mnist':
        filename = '../data/MNIST/mnist32_zoom_1'
        train_gen, n_samples_train, dev_gen, n_samples_test = preprocessing_mnist.load(filename, batch_size, npy=True)
        n_dev_samples = 10000
    elif data == 'celebA32_bw':
        data_dir = settings.filepath_celebA32
        train_gen, dev_gen = celeba.load(batch_size, data_dir=data_dir, black_white=True)
        n_dev_samples = 10000
    elif data == 'celebA64':
        if settings.euler:
            if JL_dim is None:
                time.sleep(300)
            data_dir = settings.filepath_celebA64_euler
        else:
            data_dir = settings.filepath_celebA64
        train_gen, dev_gen = celeba.load(batch_size, data_dir=data_dir, black_white=False)
        n_dev_samples = 10000

    def inf_train_gen():
        while True:
            for images, _ in train_gen():
                yield images

    gen = inf_train_gen()

    # -------------------------------------------------------
    # for saving the model create a saver
    saver = tf.train.Saver(max_to_keep=1)
    iterations_trained = 0
    if data == 'cifar10' and COMPUTE_IS:
        tp_columns = ['iteration', 'time_for_iterations', 'SWD_approximation', 'time_for_SWD', 'IS', 'time_for_IS']
    else:
        tp_columns = ['iteration', 'time_for_iterations', 'SWD_approximation', 'time_for_SWD']
    training_progress = pd.DataFrame(data=None, index=None, columns=tp_columns)

    if data == 'cifar10' and COMPUTE_IS:
        best_saver = tf.train.Saver(max_to_keep=1)
        best_IS = 0.
        if os.path.exists(directory + 'best_IS.csv'):
            best_IS = np.loadtxt(fname=directory + 'best_IS.csv')[0]
            print 'currently best IS: {}'.format(best_IS)

    # restore the model:
    if load_saved:
        saver.restore(sess=session, save_path=model_dir + 'saved_model')
        iterations_trained = int(np.loadtxt(fname=model_dir + 'epochs.csv'))
        tp_app = pd.read_csv(filepath_or_buffer=directory + 'training_progress.csv', index_col=0, header=0)
        training_progress = pd.concat([training_progress, tp_app], axis=0, ignore_index=True)
        print 'loaded training progress & model, which was already trained for {} iterations'.format(iterations_trained)
        print training_progress
        print

    # if the network is already trained completely, set send to false
    if iterations_trained == iterations:
        send = False

    # -------------------------------------------------------
    # print and get model summary
    n_params_gen = model_summary(scope='generator')[0]
    print
    n_params_disc = model_summary(scope='discriminator')[0]
    print

    # -------------------------------------------------------
    # FK: print model config to file
    model_config = [['data', 'input_dim', 'batch_size', 'learning_rate',
                     'n_features_first_g', 'n_features_last_d', 'n_features_reduction_factor',
                     'min_features', 'extra_layer_g', 'd_freq', 'd_steps',
                     'architecture', 'init_method', 'BN', 'd_BN',
                     'JL_dim', 'JL_error', 'n_projections',
                     'n_trainable_params_gen', 'n_trainable_params_disc'],
                    [data, input_dim, batch_size, learning_rate,
                     n_features_first_g, n_features_last_d, n_features_reduction_factor,
                     min_features, extra_layer_g, d_freq, d_steps,
                     architecture, init_method, BN, d_BN,
                     JL_dim, JL_error, n_projections,
                     n_params_gen, n_params_disc]]
    model_config = np.transpose(model_config)
    model_config = pd.DataFrame(data=model_config)
    model_config.to_csv(path_or_buf=directory + 'model_config.csv')
    print 'saved model configuration'
    print

    # -------------------------------------------------------
    # training loop
    print 'train model with config:'
    print model_config
    print

    t = time.time()  # get start time

    for i in xrange(iterations - iterations_trained):
        # print the current iteration
        print('iteration={}/{}'.format(i + iterations_trained + 1, iterations))

        # update generator
        images = gen.next()
        z_train = np.random.randn(batch_size, input_dim)
        if use_JL:
            JL_train = np.random.randn(d_last_layer_size, JL_dim)
            P_train = np.random.randn(JL_dim, n_projections)
            session.run(g_train, feed_dict={real_data_int: images, z: z_train, JL: JL_train, P_non_normalized: P_train})
        else:
            P_train = np.random.randn(d_last_layer_size, n_projections)
            session.run(g_train, feed_dict={real_data_int: images, z: z_train, P_non_normalized: P_train})

        # update discriminator
        if (i + iterations_trained) % d_freq == (d_freq - 1):
            for k in xrange(d_steps):
                images = gen.next()
                z_train = np.random.randn(batch_size, input_dim)
                session.run(d_train, feed_dict={real_data_int: images, z: z_train})

        if not settings.euler:
            mem = memory()
            print'memory use (GB): {}'.format(mem)

        # all STEP_SIZE_LOSS_COMPUTATION steps compute the losses and elapsed times, and generate images, and save model
        if (i + iterations_trained) % STEP_SIZE_LOSS_COMPUTATION == (STEP_SIZE_LOSS_COMPUTATION - 1):
            # get time for last 100 iterations
            elapsed_time = time.time() - t

            # generate sample images from fixed noise
            generate_image(i + iterations_trained + 1)
            print 'generated images'

            # compute and save losses on dev set, starting after ??? iterations
            if i + iterations_trained + 1 >= START_COMPUTING_LOSS:
                t = time.time()
                dev_d_loss = []
                print 'compute loss'
                j = 0
                for images_dev, _ in dev_gen():
                    if not settings.euler:
                        # progress bar
                        sys.stdout.write(
                            '\r>> Compute SWD %.1f%%' % (float(j) / float(n_dev_samples / batch_size) * 100.0))
                        sys.stdout.flush()
                        j += 1
                    z_train_dev = np.random.randn(batch_size, input_dim)
                    P_train_dev = np.random.randn(picture_size, n_projections)

                    _dev_d_loss = session.run(SWD, feed_dict={real_data_int: images_dev, z: z_train_dev,
                                                              P_non_normalized_SWD: P_train_dev})
                    dev_d_loss.append(_dev_d_loss)
                dev_loss = np.mean(dev_d_loss)
                t_loss = time.time() - t
                print

                # compute inception score (IS)
                if data == 'cifar10' and COMPUTE_IS:
                    if (i + iterations_trained) % IS_FREQ == (IS_FREQ - 1):
                        print 'compute inception score'
                        t = time.time()
                        IS_mean, IS_std, softmax = get_inception_score(N_IS, softmax=softmax)
                        IS = (IS_mean, IS_std)
                        t_IS = time.time() - t
                        print

                        # check whether this IS is the best
                        if IS_mean > best_IS:
                            best_saver.save(sess=session, save_path=model_dir + 'best/best_model')
                            best_IS = IS_mean
                            np.savetxt(fname=directory + 'best_IS.csv', X=[IS_mean, IS_std])
                            print 'saved new best model, with IS: {}'.format(IS_mean)

                    else:
                        IS = None
                        t_IS = None
            else:
                dev_loss = None
                t_loss = None
                IS = None
                t_IS = None

            if data == 'cifar10' and COMPUTE_IS:
                tp_app = pd.DataFrame(data=[[i + iterations_trained + 1, elapsed_time, dev_loss, t_loss, IS, t_IS]],
                                      index=None, columns=tp_columns)
            else:
                tp_app = pd.DataFrame(data=[[i + iterations_trained + 1, elapsed_time, dev_loss, t_loss]],
                                      index=None, columns=tp_columns)
            training_progress = pd.concat([training_progress, tp_app], axis=0, ignore_index=True)

            # save model
            saver.save(sess=session, save_path=model_dir + 'saved_model')
            # save number of iterations trained
            np.savetxt(fname=model_dir + 'epochs.csv', X=[i + iterations_trained + 1])
            print 'saved model after training iterations {}'.format(i + iterations_trained + 1)
            # save training progress
            training_progress.to_csv(path_or_buf=directory + 'training_progress.csv')
            print 'saved training progress'
            print

            # fix new start time
            t = time.time()

    # -------------------------------------------------------
    # after training close the session
    session.close()
    tf.reset_default_graph()

    # -------------------------------------------------------
    # when training is done send email
    if send:
        subject = 'JL-SWGAN ({}) training finished'.format(data)
        body = 'to download the results of this model use (in the terminal):\n\n'
        body += 'scp -r fkrach@euler.ethz.ch:' + directory + ' .'
        files = [directory + 'model_config.csv', directory + 'training_progress.csv',
                 samples_dir + 'iteration_{}.png'.format(iterations)]
        send_email.send_email(subject=subject, body=body, file_names=files)

    return directory


def parallel_training(parameters=None, nb_jobs=-1):
    """
    :param parameters: an array of arrays with all the parameters
    :param nb_jobs: number of jobs that run parallel, -1 means all available cpus are used
    :return:
    """
    if parameters is None:
        parameters = [('mnist', INPUT_DIM, BATCH_SIZE, 1e-4, ITERS, FIXED_NOISE_SIZE, N_FEATURES_FIRST, 2, 64,
                       N_FEATURES_FIRST, False, 1, 1, 'JLSWGN', 'He', False, 'LN', 32*32/2, 10000),
                      ('mnist', INPUT_DIM, BATCH_SIZE, 1e-4, ITERS, FIXED_NOISE_SIZE, N_FEATURES_FIRST, 2, 64,
                       N_FEATURES_FIRST, False, 1, 1, 'SWGN', 'He', False, 'LN', None, 10000)]
    results = Parallel(n_jobs=nb_jobs)(delayed(train)(data, input_dim, batch_size, learning_rate, iterations,
                                                      fixed_noise_size, n_features_first_g, n_features_reduction_factor,
                                                      min_features, n_features_last_d, extra_layer_g, d_freq, d_steps,
                                                      architecture, init_method, BN, d_BN, JL_dim, n_projections)
                                       for data, input_dim, batch_size, learning_rate, iterations,
                                           fixed_noise_size, n_features_first_g, n_features_reduction_factor,
                                           min_features, n_features_last_d, extra_layer_g, d_freq, d_steps,
                                           architecture, init_method, BN, d_BN, JL_dim, n_projections in parameters)

    if settings.send_email:
        subject = 'JL-SWGAN parallel training finished'
        body = 'to download all the results of this parallel run use (in the terminal):\n\n'
        for directory in results:
            body += 'scp -r fkrach@euler.ethz.ch:'+directory+' .; '
        send_email.send_email(subject=subject, body=body, file_names=None)
    return 0


if __name__ == '__main__':
    # -------------------------------------------------------
    # parallel training
    param_array = settings.JLSWGAN_param_array1_3
    if N_CPUS_PARALLEL is None:
        nb_jobs = settings.number_parallel_jobs
    else:
        nb_jobs = N_CPUS_PARALLEL

    parallel_training(parameters=param_array, nb_jobs=nb_jobs)
    # parallel_training(nb_jobs=nb_jobs)

    # train(data='cifar10', JL_dim=128, n_projections=1000, d_BN='BN', extra_layer_g=False, learning_rate=1e-3,
    #       n_features_first_g=256, n_features_last_d=256)





