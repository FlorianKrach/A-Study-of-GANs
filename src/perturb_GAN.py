"""
author: Florian Krach
"""


import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import mnist
import save_images
import settings
import send_email
import os
from subprocess import call
import tensorflow.contrib.slim as slim
import pandas as pd
import time
from joblib import Parallel, delayed


# FK: variable definitions
BATCH_SIZE = 50
INPUT_DIM = 128
N_FEATURES_FIRST = 256  # FK: has to be divisible by 4
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 20000 # How many generator iterations to train for
FIXED_NOISE_SIZE = 128

if settings.euler:
    N_CPUS_TF = 1  # set to None to use value of settings.py
    N_CPUS_PARALLEL = 4  # set to None to use value of settings.py
else:
    N_CPUS_TF = 1  # set to None to use value of settings.py
    N_CPUS_PARALLEL = 1  # set to None to use value of settings.py


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


# FK: model summary of
def model_summary(scope):
    model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    return slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def leaky_relu(x):
    return tf.maximum(x, 0.2 * x)


def generator(z, n_features_first=N_FEATURES_FIRST, n_features_reduction_factor=2, fix_first_layers=False,
              fix_last_layer=False, fix_2last_layer=False, architecture='WGANGP'):

    first_layers_trainable = not fix_first_layers
    last_layer_trainable = not fix_last_layer
    last2_layer_trainable = not fix_2last_layer

    if architecture == 'DCGAN':
        normalizer = layers.batch_norm
    else:  # WGAN-GP
        normalizer = None

    # the layers use relu activations (default)
    with tf.variable_scope('generator'):
        z = layers.fully_connected(z, num_outputs=4*4*n_features_first, trainable=first_layers_trainable,
                                   normalizer_fn=normalizer)
        z = tf.reshape(z, [-1, 4, 4, n_features_first])  # we use the dimensions as NHWC resp. BHWC

        z = layers.conv2d_transpose(z, num_outputs=int(n_features_first/n_features_reduction_factor), kernel_size=5,
                                    stride=2, trainable=first_layers_trainable, normalizer_fn=normalizer)
        z = layers.conv2d_transpose(z, num_outputs=int(n_features_first/(n_features_reduction_factor**2)),
                                    kernel_size=5, stride=2, trainable=last2_layer_trainable, normalizer_fn=normalizer)
        z = layers.conv2d_transpose(z, num_outputs=1, kernel_size=5, stride=2, activation_fn=tf.nn.sigmoid,
                                    trainable=last_layer_trainable)
        return z[:, 2:-2, 2:-2, :]  # FK: of the 32x32 image leave away the outer border of 4 pixels to get a 28x28
        # image as in the training set of MNIST


def discriminator(x, reuse, n_features_first=N_FEATURES_FIRST, n_features_reduction_factor=2, fix_first_layers=False,
                  fix_last_layer=False, fix_2last_layer=False, architecture='WGANGP'):

    first_layers_trainable = not fix_first_layers
    last_layer_trainable = not fix_last_layer
    last2_layer_trainable = not fix_2last_layer

    if architecture == 'DCGAN':
        normalizer = layers.batch_norm
    else:  # WGAN-GP
        normalizer = None

    with tf.variable_scope('discriminator', reuse=reuse):
        x = layers.conv2d(x, num_outputs=int(n_features_first/(n_features_reduction_factor**2)), kernel_size=5,
                          stride=2, activation_fn=leaky_relu, trainable=first_layers_trainable,
                          normalizer_fn=normalizer)
        x = layers.conv2d(x, num_outputs=int(n_features_first/n_features_reduction_factor), kernel_size=5, stride=2,
                          activation_fn=leaky_relu, trainable=first_layers_trainable, normalizer_fn=normalizer)
        x = layers.conv2d(x, num_outputs=n_features_first, kernel_size=5, stride=2,
                          activation_fn=leaky_relu, trainable=last2_layer_trainable, normalizer_fn=normalizer)

        x = layers.flatten(x)
        return layers.fully_connected(x, num_outputs=1, activation_fn=None, trainable=last_layer_trainable)


# split up discriminator into two parts, to make gradient computation less expensive, if first layers are fixed
def discriminator1(x, reuse, n_features_first=N_FEATURES_FIRST, n_features_reduction_factor=2, fix_first_layers=False,
                   architecture='WGANGP', **kwargs):

    first_layers_trainable = not fix_first_layers

    if architecture == 'DCGAN':
        normalizer = layers.batch_norm
    else:  # WGAN-GP
        normalizer = None

    with tf.variable_scope('discriminator1', reuse=reuse):
        x = layers.conv2d(x, num_outputs=int(n_features_first/(n_features_reduction_factor**2)), kernel_size=5,
                          stride=2, activation_fn=leaky_relu, trainable=first_layers_trainable,
                          normalizer_fn=normalizer)
        return layers.conv2d(x, num_outputs=int(n_features_first/n_features_reduction_factor), kernel_size=5, stride=2,
                             activation_fn=leaky_relu, trainable=first_layers_trainable, normalizer_fn=normalizer)


def discriminator2(x, reuse, n_features_first=N_FEATURES_FIRST, fix_last_layer=False, fix_2last_layer=False,
                   architecture='WGANGP', **kwargs):

    last_layer_trainable = not fix_last_layer
    last2_layer_trainable = not fix_2last_layer

    if architecture == 'DCGAN':
        normalizer = layers.batch_norm
    else:  # WGAN-GP
        normalizer = None

    with tf.variable_scope('discriminator2', reuse=reuse):
        x = layers.conv2d(x, num_outputs=n_features_first, kernel_size=5, stride=2,
                          activation_fn=leaky_relu, trainable=last2_layer_trainable, normalizer_fn=normalizer)

        x = layers.flatten(x)
        return layers.fully_connected(x, num_outputs=1, activation_fn=None, trainable=last_layer_trainable)


# TODO: try out: instead of perturbing, shuffle the weights a.) all, b.) in the 5x5 features, c.) only the features
def perturbed(standard_deviation_factor=1., perturb_distribution='unif', perturbe_last_layer=True, runs=1,
              pretrained_model='DCGAN'):

    # -------------------------------------------------------
    # setting for sending emails
    send = settings.send_email

    # -------------------------------------------------------
    # perturb_distribution default
    if perturb_distribution in ['unif', 'uniform']:
        perturb_distribution = 'uniform'
    else:
        perturb_distribution = 'normal'

    # -------------------------------------------------------
    # pretrained model & default
    if pretrained_model == 'WGANGP':
        load_path = 'pretrained_models/' + \
                    '128_50_256_5_10_0.0001_128_2_False_False_False_False_False_False_WGANGP_False_False/'

    else:  # DCGAN
        load_path = 'pretrained_models/' + \
                    '128_50_256_1_10_0.0001_128_2_False_False_False_False_False_False_DCGAN_False_False/'
        pretrained_model = 'DCGAN'
    load_path_extended = load_path + 'model/saved_model'

    # -------------------------------------------------------
    # create unique folder name
    dir1 = 'perturbed_GAN/'
    directory = dir1 + str(perturb_distribution) + '_' + str(standard_deviation_factor) + '_' + \
                str(perturbe_last_layer) + '_' +str(pretrained_model) + '/'
    samples_dir = directory + 'samples/'
    model_dir = directory + 'model/'

    # create directories if they don't exist
    if not os.path.isdir(dir1):
        call(['mkdir', dir1])

    if not os.path.isdir(directory):
        load_saved = False
        print 'make new directory:', directory
        print
        call(['mkdir', directory])
        call(['mkdir', samples_dir])

    # -------------------------------------------------------
    # load hyperparameters from the config file
    model_config_old = pd.read_csv(filepath_or_buffer=load_path+'model_config.csv', index_col=0, header=0)
    input_dim = int(model_config_old.values[0, 1])
    batch_size = int(model_config_old.values[1, 1])
    n_features_first = int(model_config_old.values[2, 1])
    critic_iters = int(model_config_old.values[3, 1])
    lambda_reg = float(model_config_old.values[4, 1])
    learning_rate = float(model_config_old.values[5, 1])
    fixed_noise_size = int(model_config_old.values[6, 1])
    n_features_reduction_factor = int(model_config_old.values[7, 1])
    architecture = str(model_config_old.values[14, 1])

    print 'configuration of the pretrained model:'
    print model_config_old
    print

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
    def build_model(fix_first_layers_gen_b=False, fix_last_layer_gen_b=False,
                    fix_2last_layer_gen_b=False,
                    fix_first_layers_disc_b=False, fix_last_layer_disc_b=False,
                    fix_2last_layer_disc_b=False
                    ):
        """
        - function to build the model, where the layers can be fixed/trainable as wanted
        :param fix_first_layers_gen_b:
        :param fix_last_layer_gen_b:
        :param fix_2last_layer_gen_b:
        :param fix_first_layers_disc_b:
        :param fix_last_layer_disc_b:
        :param fix_2last_layer_disc_b:
        :return:
        """

        with tf.name_scope('placeholders'):
            x_true = tf.placeholder(tf.float32, [None, 28, 28, 1])
            z = tf.placeholder(tf.float32, [None, input_dim])

        x_generated = generator(z, n_features_first=n_features_first,
                                n_features_reduction_factor=n_features_reduction_factor,
                                fix_first_layers=fix_first_layers_gen_b, fix_last_layer=fix_last_layer_gen_b,
                                fix_2last_layer=fix_2last_layer_gen_b, architecture=architecture)

        if architecture == 'DCGAN':
            d_true1 = discriminator1(x_true, reuse=False, n_features_first=n_features_first,
                                     n_features_reduction_factor=n_features_reduction_factor,
                                     fix_first_layers=fix_first_layers_disc_b, fix_last_layer=fix_last_layer_disc_b,
                                     fix_2last_layer=fix_2last_layer_disc_b, architecture=architecture)
            d_true = discriminator2(d_true1, reuse=False, n_features_first=n_features_first,
                                    n_features_reduction_factor=n_features_reduction_factor,
                                    fix_first_layers=fix_first_layers_disc_b, fix_last_layer=fix_last_layer_disc_b,
                                    fix_2last_layer=fix_2last_layer_disc_b, architecture=architecture)

            d_generated1 = discriminator1(x_generated, reuse=True, n_features_first=n_features_first,
                                          n_features_reduction_factor=n_features_reduction_factor,
                                          fix_first_layers=fix_first_layers_disc_b, fix_last_layer=fix_last_layer_disc_b,
                                          fix_2last_layer=fix_2last_layer_disc_b, architecture=architecture)
            d_generated = discriminator2(d_generated1, reuse=True, n_features_first=n_features_first,
                                         n_features_reduction_factor=n_features_reduction_factor,
                                         fix_first_layers=fix_first_layers_disc_b, fix_last_layer=fix_last_layer_disc_b,
                                         fix_2last_layer=fix_2last_layer_disc_b, architecture=architecture)

        else:  # WGAN-GP
            d_true = discriminator(x_true, reuse=False, n_features_first=n_features_first,
                                   n_features_reduction_factor=n_features_reduction_factor,
                                   fix_first_layers=fix_first_layers_disc_b, fix_last_layer=fix_last_layer_disc_b,
                                   fix_2last_layer=fix_2last_layer_disc_b, architecture=architecture)

            d_generated = discriminator(x_generated, reuse=True, n_features_first=n_features_first,
                                        n_features_reduction_factor=n_features_reduction_factor,
                                        fix_first_layers=fix_first_layers_disc_b, fix_last_layer=fix_last_layer_disc_b,
                                        fix_2last_layer=fix_2last_layer_disc_b, architecture=architecture)

        if architecture == 'DCGAN':
            with tf.name_scope('loss'):
                g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_generated,
                                                                                labels=tf.ones_like(d_generated)))
                d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_generated,
                                                                                labels=tf.zeros_like(d_generated))) +\
                         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_true,
                                                                                labels=tf.ones_like(d_true)))
                d_loss = d_loss/2.

            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=2*learning_rate, beta1=0.5)

                g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                g_train = optimizer.minimize(g_loss, var_list=g_vars)
                d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                d_train = optimizer.minimize(d_loss, var_list=d_vars)

        else:  # WGAN-GP
            with tf.name_scope('regularizer'):
                epsilon = tf.random_uniform([batch_size, 1, 1, 1], 0.0, 1.0)
                x_hat = epsilon * x_true + (1 - epsilon) * x_generated

                # without splitting the discriminator
                d_hat = discriminator(x_hat, reuse=True, n_features_first=n_features_first,
                                      n_features_reduction_factor=n_features_reduction_factor,
                                      fix_first_layers=fix_first_layers_disc_b, fix_last_layer=fix_last_layer_disc_b,
                                      fix_2last_layer=fix_2last_layer_disc_b, architecture=architecture)

                gradients = tf.gradients(d_hat, x_hat)[0]
                ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
                d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)

            with tf.name_scope('loss'):
                g_loss = -tf.reduce_mean(d_generated)
                wasserstein_dist = tf.reduce_mean(d_true) - tf.reduce_mean(d_generated)
                d_loss = -wasserstein_dist + lambda_reg * d_regularizer

            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0, beta2=0.9)
                # FK: TODO: beta1 = 0.5 in IWGAN, here 0 -> change? In experiments (only 1000 epochs) it seemed better with 0

                g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                g_train = optimizer.minimize(g_loss, var_list=g_vars)
                d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                d_train = optimizer.minimize(d_loss, var_list=d_vars)

        # initialize variables
        session.run(tf.global_variables_initializer())

        if architecture == 'DCGAN':
            return x_true, z, x_generated, d_true1, d_true, d_generated1, d_generated, g_loss, d_loss, optimizer, \
                   g_vars, g_train, d_vars, d_train
        else:  # WGANGP
            return x_true, z, x_generated, d_true, d_generated, epsilon, x_hat, d_hat, gradients, ddx, d_regularizer, \
                   g_loss, wasserstein_dist, d_loss, optimizer, g_vars, g_train, d_vars, d_train

    # -------------------------------------------------------
    # FK: For saving samples, taken from IWGAN
    np.random.seed(345)
    fixed_noise = np.random.normal(size=(fixed_noise_size, input_dim)).astype('float32')
    np.random.seed()

    def generate_image(frame):
        samples = session.run(x_generated, feed_dict={z: fixed_noise}).squeeze()
        if type(frame) == int:
            save_images.save_images(
                samples.reshape((fixed_noise_size, 28, 28)),
                samples_dir + 'iteration_{}.png'.format(frame)
            )
        elif type(frame) == str:
            save_images.save_images(
                samples.reshape((fixed_noise_size, 28, 28)),
                samples_dir + frame + '.png'
            )

    # -------------------------------------------------------
    # FK: load the model either from pretrained_model or, if perturbed_retraining already started from there
    # first load the model s.t. all variables are trainable


    pixel_wise_error = np.zeros(runs)
    highest_error = 0
    # saver = tf.train.Saver(max_to_keep=1)
    # trainable_vars = tf.trainable_variables(scope='generator')
    for run in range(runs):
        # restore the model that was pretrained
        # print memory()
        t = time.time()
        session = tf.Session(config=config)
        if architecture == 'DCGAN':
            x_true, z, x_generated, d_true1, d_true, d_generated1, d_generated, g_loss, d_loss, optimizer, g_vars, \
            g_train, d_vars, d_train = \
                build_model(fix_first_layers_gen_b=False, fix_last_layer_gen_b=False,
                            fix_2last_layer_gen_b=False,
                            fix_first_layers_disc_b=False, fix_last_layer_disc_b=False,
                            fix_2last_layer_disc_b=False)
        else:  # WGANGP
            x_true, z, x_generated, d_true, d_generated, epsilon, x_hat, d_hat, gradients, ddx, d_regularizer, g_loss, \
            wasserstein_dist, d_loss, optimizer, g_vars, g_train, d_vars, d_train = \
                build_model(fix_first_layers_gen_b=False, fix_last_layer_gen_b=False,
                            fix_2last_layer_gen_b=False,
                            fix_first_layers_disc_b=False, fix_last_layer_disc_b=False,
                            fix_2last_layer_disc_b=False)
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(sess=session, save_path=load_path_extended)
        # print memory()

        if run == 0:
            generate_image('before_perturbation')
        im_before = session.run(x_generated, feed_dict={z: fixed_noise}).squeeze()

        # perturb the weights
        trainable_vars = tf.trainable_variables(scope='generator')
        for v in trainable_vars:
            # print v.name
            if v.name in ['generator/Conv2d_transpose_2/weights:0', 'generator/Conv2d_transpose_2/biases:0']:
                if perturbe_last_layer:
                    if perturb_distribution == 'uniform':
                        std = np.std(session.run(v))
                        session.run(tf.assign(v, value=v + tf.random_uniform(v.shape, minval=-std*standard_deviation_factor,
                                                                             maxval=std*standard_deviation_factor)))
                    else:
                        std = np.std(session.run(v))
                        session.run(tf.assign(v, value=v + tf.random_normal(v.shape, mean=0.0,
                                                                            stddev=std * standard_deviation_factor)))
            else:
                if perturb_distribution == 'uniform':
                    std = np.std(session.run(v))
                    session.run(tf.assign(v, value=v + tf.random_uniform(v.shape, minval=-std * standard_deviation_factor,
                                                                         maxval=std * standard_deviation_factor)))
                else:
                    std = np.std(session.run(v))
                    session.run(tf.assign(v, value=v + tf.random_normal(v.shape, mean=0.0,
                                                                        stddev=std * standard_deviation_factor)))

        im_after = session.run(x_generated, feed_dict={z: fixed_noise}).squeeze()
        err = np.sum(np.square(im_before - im_after))/float(fixed_noise_size)
        pixel_wise_error[run] = err
        if err > highest_error:
            highest_error = err
            generate_image('after_perturbation_{}'.format(run))
        print 'completed run: {}'.format(run)
        del im_before, im_after, std, err
        session.close()
        tf.reset_default_graph()
        print 'time: {}'.format(time.time() - t)

    out = '\nhighest error: {}, at {}'.format(highest_error, np.argmax(pixel_wise_error))
    out += '\nmin error:     {}'.format(np.min(pixel_wise_error))
    out += '\nmean error:    {}'.format(np.mean(pixel_wise_error))
    print out

    if settings.send_email:
        subject = 'perturb GAN finished'
        body = out
        send_email.send_email(subject=subject, body=body,
                              file_names=[samples_dir + 'after_perturbation_{}.png'.format(np.argmax(pixel_wise_error)),
                                          samples_dir + 'before_perturbation.png'])

    return pixel_wise_error


if __name__ == '__main__':
    e = perturbed(standard_deviation_factor=0.5, perturb_distribution='uniform', perturbe_last_layer=True, runs=10000,
                  pretrained_model='DCGAN')
    # e = perturbed(standard_deviation_factor=.5, perturb_distribution='normal', perturbe_last_layer=True, runs=100,
    #               pretrained_model='DCGAN')

