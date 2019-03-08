"""
author: Florian Krach
used parts of the code of the following implementations:
- Minimal implementation of Wasserstein GAN for MNIST, https://github.com/adler-j/minimal_wgan
- Improved Wasserstein-GAN, https://github.com/igul222/improved_wgan_training
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
    N_CPUS_PARALLEL = 12  # set to None to use value of settings.py
else:
    N_CPUS_TF = 1  # set to None to use value of settings.py
    N_CPUS_PARALLEL = 2  # set to None to use value of settings.py


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
def perturbed_retrain(epochs_new=1000, standard_deviation_factor_new=1,
                      fix_last_layer_gen_new=False, fix_2last_layer_gen_new=True,
                      fix_last_layer_disc_new=False, fix_2last_layer_disc_new=True,
                      pretrained_model='DCGAN', perturb_BN=False,
                      load_saved=True):
    """
    - this function is used to take a pretrained network, identified by 'pretrained_model' and adds a noise to the
      weights of the fixed layers, then trains number of 'epochs' (only non-fixed layers can be trained)
    - the function computes losses and auto-saves the model every 100 steps and automatically resumes training where it
      stopped (when load_saved=True)

    :param epochs_new:
    :param standard_deviation_factor_new:
    :param fix_last_layer_gen_new:
    :param fix_2last_layer_gen_new:
    :param fix_last_layer_disc_new:
    :param fix_2last_layer_disc_new:
    :param pretrained_model: 'WGANGP' or 'DCGAN', defaults to 'DCGAN'
    :param perturb_BN: whether also the weights of the batchnormalization are perturbed
    :param load_saved:
    :return:
    """

    # -------------------------------------------------------
    # setting for sending emails
    send = settings.send_email

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
    directory = 'perturbed_retraining/' + str(standard_deviation_factor_new) + '_' + \
                str(fix_last_layer_gen_new) + '_' + str(fix_2last_layer_gen_new) + '_' + \
                str(fix_last_layer_disc_new) + '_' + str(fix_2last_layer_disc_new) + '_' + \
                str(pretrained_model) + '_' + str(perturb_BN) + '/'
    samples_dir = directory + 'samples/'
    model_dir = directory + 'model/'

    # create directories if they don't exist
    if not os.path.isdir('perturbed_retraining/'):
        call(['mkdir', 'perturbed_retraining/'])

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
    def build_model(fix_first_layers_gen_b=True, fix_last_layer_gen_b=fix_last_layer_gen_new,
                    fix_2last_layer_gen_b=fix_2last_layer_gen_new,
                    fix_first_layers_disc_b=True, fix_last_layer_disc_b=fix_last_layer_disc_new,
                    fix_2last_layer_disc_b=fix_2last_layer_disc_new
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
    fixed_noise = np.random.normal(size=(fixed_noise_size, input_dim)).astype('float32')

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
    if architecture == 'DCGAN':
        training_progress = pd.DataFrame(data=None, index=None, columns=['epoch', 'time', 'd_loss'])
    else:  # WGAN-GP
        training_progress = pd.DataFrame(data=None, index=None, columns=['epoch', 'time', 'Wasserstein_dist', 'd_loss'])

    # restore the model that was already trained with perturbed_train and saved:
    if load_saved:
        # build model
        if architecture == 'DCGAN':
            x_true, z, x_generated, d_true1, d_true, d_generated1, d_generated, g_loss, d_loss, optimizer, g_vars, \
            g_train, d_vars, d_train = build_model()
        else:  # WGANGP
            x_true, z, x_generated, d_true, d_generated, epsilon, x_hat, d_hat, gradients, ddx, d_regularizer, g_loss, \
            wasserstein_dist, d_loss, optimizer, g_vars, g_train, d_vars, d_train = build_model()

        # create saver and reload the already trained
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(sess=session, save_path=model_dir+'saved_model')

        # get the number of epochs already trained and the training progress
        epochs_trained = int(np.loadtxt(fname=model_dir+'epochs.csv'))
        tp_app = pd.read_csv(filepath_or_buffer=directory+'training_progress.csv', index_col=0, header=0)
        training_progress = pd.concat([training_progress, tp_app], axis=0, ignore_index=True)
        print 'loaded training progress, and the model, which was already trained for {} epochs'.format(epochs_trained)
        print training_progress
        print

    # load the pretrained model from pretrained models, perturb the fixed-layer weights
    else:
        print 'load the pretrained model with architecture: {}'.format(architecture)
        print
        epochs_trained = 0

        # first load the model s.t. all variables are trainable
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

        # restore the model that was pretrained
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(sess=session, save_path=load_path_extended)
        generate_image('before_perturbation')

        # perturb the weights
        trainable_vars = tf.trainable_variables()
        for v in trainable_vars:
            if perturb_BN:
                std = np.std(session.run(v))
                session.run(tf.assign(v, value=v + tf.random_normal(v.shape, mean=0.0,
                                                                    stddev=std * standard_deviation_factor_new)))
            elif 'BatchNorm' not in v.name:
                std = np.std(session.run(v))
                session.run(tf.assign(v, value=v+tf.random_normal(v.shape, mean=0.0,
                                                                      stddev=std*standard_deviation_factor_new)))
        saver.save(sess=session, save_path=model_dir+'saved_model')
        print
        print 'perturbed weights were saved'

        # load new session, so that no conflict with names in the name_scopes
        session.close()
        tf.reset_default_graph()
        session = tf.Session(config=config)

        # load the model with the perturbed weights, but now s.t. the correct variables are trainable
        if architecture == 'DCGAN':
            x_true, z, x_generated, d_true1, d_true, d_generated1, d_generated, g_loss, d_loss, optimizer, g_vars, \
            g_train, d_vars, d_train = build_model()
        else:  # WGANGP
            x_true, z, x_generated, d_true, d_generated, epsilon, x_hat, d_hat, gradients, ddx, d_regularizer, g_loss, \
            wasserstein_dist, d_loss, optimizer, g_vars, g_train, d_vars, d_train = build_model()

        # restore the model that was pretrained
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(sess=session, save_path=model_dir+'saved_model')
        print 'reloaded perturbed weights model, now with correct trainable weights'
        print
        generate_image('after_perturbation')

    # -------------------------------------------------------
    # if the network is already trained completely, set send to false
    if epochs_trained == epochs_new:
        send = False

    # -------------------------------------------------------
    # FK: print and get model summary
    n_params_gen = model_summary(scope='generator')[0]
    print
    n_params_disc = model_summary(scope='discriminator')[0]
    print

    # -------------------------------------------------------
    # FK: print model config to file
    model_config = [['standard_deviation_factor_new',
                     'fix_last_layer_gen_new', 'fix_2last_layer_gen_new',
                     'fix_last_layer_disc_new', 'fix_2last_layer_disc_new',
                     'pretrained_model', 'perturb_BN',
                     'n_trainable_params_gen', 'n_trainable_params_disc'],
                    [standard_deviation_factor_new,
                     fix_last_layer_gen_new, fix_2last_layer_gen_new,
                     fix_last_layer_disc_new, fix_2last_layer_disc_new,
                     pretrained_model, perturb_BN,
                     n_params_gen, n_params_disc]]
    model_config = np.transpose(model_config)
    model_config = pd.DataFrame(data=model_config)
    model_config.to_csv(path_or_buf=directory+'model_config.csv')
    print 'saved model configuration'
    print

    # -------------------------------------------------------
    # FK: get the MNIST data loader
    train_gen, dev_gen, test_gen = mnist.load(batch_size, batch_size)

    # create a infinite generator
    def inf_train_gen():
        while True:
            for images, targets in train_gen():
                yield images

    gen = inf_train_gen()

    # -------------------------------------------------------
    # training loop
    t = time.time()  # get start time

    for i in xrange(epochs_new-epochs_trained):
        z_train = np.random.randn(batch_size, input_dim)
        session.run(g_train, feed_dict={z: z_train})

        # loop for critic training
        for j in xrange(critic_iters):
            # FK: insert the following 3 lines s.t. not the same batch is used for all 5 discriminator updates
            batch = gen.next()
            images = batch.reshape([-1, 28, 28, 1])
            z_train = np.random.randn(batch_size, input_dim)
            session.run(d_train, feed_dict={x_true: images, z: z_train})

        # print the current epoch
        print('epoch={}/{}'.format(i+epochs_trained+1, epochs_new))

        # all 100 steps compute the losses and elapsed times, and generate images
        if (i + epochs_trained) % 100 == 99:
            # get time for last 100 epochs
            elapsed_time = time.time() - t

            # generate sample images from fixed noise
            generate_image(i+epochs_trained+1)
            print 'generated images'

            # save model
            saver.save(sess=session, save_path=model_dir+'saved_model')
            # save number of epochs trained
            np.savetxt(fname=model_dir+'epochs.csv', X=[i+epochs_trained+1])
            print 'saved model after training epoch {}'.format(i+epochs_trained+1)

            # compute and save losses on dev set
            if architecture == 'DCGAN':
                dev_d_loss = []
                for images_dev, _ in dev_gen():
                    images_dev = images_dev.reshape([-1, 28, 28, 1])
                    z_train_dev = np.random.randn(batch_size, input_dim)
                    _dev_d_loss = session.run(d_loss, feed_dict={x_true: images_dev, z: z_train_dev})
                    dev_d_loss.append(_dev_d_loss)
                tp_app = pd.DataFrame(data=[[i + epochs_trained + 1, elapsed_time, np.mean(dev_d_loss)]],
                                      index=None, columns=['epoch', 'time', 'd_loss'])
                training_progress = pd.concat([training_progress, tp_app], axis=0, ignore_index=True)
                training_progress.to_csv(path_or_buf=directory + 'training_progress.csv')
            else:  # WGAN-GP
                dev_W_dist = []
                dev_d_loss = []
                for images_dev, _ in dev_gen():
                    images_dev = images_dev.reshape([-1, 28, 28, 1])
                    z_train_dev = np.random.randn(batch_size, input_dim)
                    _dev_W_dist = session.run(wasserstein_dist, feed_dict={x_true: images_dev, z: z_train_dev})
                    _dev_d_loss = session.run(d_loss, feed_dict={x_true: images_dev, z: z_train_dev})
                    dev_W_dist.append(_dev_W_dist)
                    dev_d_loss.append(_dev_d_loss)
                tp_app = pd.DataFrame(data=[[i+epochs_trained+1, elapsed_time, np.mean(dev_W_dist), np.mean(dev_d_loss)]],
                                      index=None, columns=['epoch', 'time', 'Wasserstein_dist', 'd_loss'])
                training_progress = pd.concat([training_progress, tp_app], axis=0, ignore_index=True)
                training_progress.to_csv(path_or_buf=directory+'training_progress.csv')
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
        subject = 'GAN (MNIST) perturbed pretrained network retraining finished'
        body = 'to download the results of this model use (in the terminal):\n\n'
        body += 'scp -r fkrach@euler.ethz.ch:/cluster/home/fkrach/MasterThesis/MTCode1/'+directory+' .'
        files = [directory+'model_config.csv', directory+'training_progress.csv']
        send_email.send_email(subject=subject, body=body, file_names=files)

    return directory


def parallel_training(parameters=None, nb_jobs=-1):
    """
    :param parameters: an array of arrays with all the parameters
    :param nb_jobs: number of jobs that run parallel, -1 means all available cpus are used
    :return:
    """
    if parameters is None:
        parameters = [(10000, 1, False, True, False, True, 'DCGAN', False),
                      (10000, 1, False, True, False, True, 'WGANGP', False)]
    results = Parallel(n_jobs=nb_jobs)(delayed(perturbed_retrain)(epochs_new, standard_deviation_factor_new,
                                                                  fix_last_layer_gen_new, fix_2last_layer_gen_new,
                                                                  fix_last_layer_disc_new, fix_2last_layer_disc_new,
                                                                  pretrained_model, perturb_BN)
                                       for epochs_new, standard_deviation_factor_new,
                                           fix_last_layer_gen_new, fix_2last_layer_gen_new,
                                           fix_last_layer_disc_new, fix_2last_layer_disc_new,
                                           pretrained_model, perturb_BN in parameters)
    if settings.send_email:
        subject = 'GAN (MNIST) parallel training finished'
        body = 'to download all the results of this parallel run use (in the terminal):\n\n'
        for directory in results:
            body += 'scp -r fkrach@euler.ethz.ch:/cluster/home/fkrach/MasterThesis/MTCode1/'+directory+' .; '
        send_email.send_email(subject=subject, body=body, file_names=None)
    return 0





if __name__ == '__main__':
    # -------------------------------------------------------
    # parallel training
    param_array = settings.perturbed_param_array3
    if N_CPUS_PARALLEL is None:
        nb_jobs = settings.number_parallel_jobs
    else:
        nb_jobs = N_CPUS_PARALLEL

    parallel_training(parameters=param_array, nb_jobs=nb_jobs)

    # perturbed_retrain(epochs_new=1000, standard_deviation_factor_new=2,
    #                   fix_last_layer_gen_new=False, fix_2last_layer_gen_new=True,
    #                   fix_last_layer_disc_new=False, fix_2last_layer_disc_new=True,
    #                   pretrained_model='DCGAN',
    #                   load_saved=True)




