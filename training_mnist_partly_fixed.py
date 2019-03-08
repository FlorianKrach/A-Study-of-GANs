"""
author: Florian Krach
modified code from: Minimal implementation of Wasserstein GAN for MNIST, https://github.com/adler-j/minimal_wgan
also used parts from: Improved Wasserstein-GAN, https://github.com/igul222/improved_wgan_training
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


# FK: model summary of
def model_summary(scope, BN_layers_trainable=True):
    model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    vars2 = []
    for v in model_vars:
        if 'BatchNorm' not in v.name:
            vars2 += [v]
    if not BN_layers_trainable:
        model_vars = vars2
    return slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def leaky_relu(x):
    return tf.maximum(x, 0.2 * x)


def generator(z, n_features_first=N_FEATURES_FIRST, n_features_reduction_factor=2, fix_first_layers=False,
              fix_last_layer=False, fix_2last_layer=False, architecture='WGANGP', init_method=None):

    first_layers_trainable = not fix_first_layers
    last_layer_trainable = not fix_last_layer
    last2_layer_trainable = not fix_2last_layer

    if architecture == 'DCGAN':
        normalizer = layers.batch_norm
    else:  # WGAN-GP
        normalizer = None

    if init_method in ['He']:
        init = layers.variance_scaling_initializer()
    else:
        init = layers.xavier_initializer()

    # the layers use relu activations (default)
    with tf.variable_scope('generator'):
        z = layers.fully_connected(z, num_outputs=4*4*n_features_first, trainable=first_layers_trainable,
                                   normalizer_fn=normalizer, weights_initializer=init)
        z = tf.reshape(z, [-1, 4, 4, n_features_first])  # we use the dimensions as NHWC resp. BHWC

        z = layers.conv2d_transpose(z, num_outputs=int(n_features_first/n_features_reduction_factor), kernel_size=5,
                                    stride=2, trainable=first_layers_trainable, normalizer_fn=normalizer,
                                    weights_initializer=init)
        z = layers.conv2d_transpose(z, num_outputs=int(n_features_first/(n_features_reduction_factor**2)),
                                    kernel_size=5, stride=2, trainable=last2_layer_trainable, normalizer_fn=normalizer,
                                    weights_initializer=init)
        z = layers.conv2d_transpose(z, num_outputs=1, kernel_size=5, stride=2, activation_fn=tf.nn.sigmoid,
                                    trainable=last_layer_trainable, weights_initializer=init)
        return z[:, 2:-2, 2:-2, :]  # FK: of the 32x32 image leave away the outer border of 4 pixels to get a 28x28
        # image as in the training set of MNIST


def discriminator(x, reuse, n_features_first=N_FEATURES_FIRST, n_features_reduction_factor=2, fix_first_layers=False,
                  fix_last_layer=False, fix_2last_layer=False, architecture='WGANGP', init_method=None):

    first_layers_trainable = not fix_first_layers
    last_layer_trainable = not fix_last_layer
    last2_layer_trainable = not fix_2last_layer

    if architecture == 'DCGAN':
        normalizer = layers.batch_norm
    else:  # WGAN-GP
        normalizer = None

    if init_method in ['He']:
        init = layers.variance_scaling_initializer()
    else:
        init = layers.xavier_initializer()

    with tf.variable_scope('discriminator', reuse=reuse):
        x = layers.conv2d(x, num_outputs=int(n_features_first/(n_features_reduction_factor**2)), kernel_size=5,
                          stride=2, activation_fn=leaky_relu, trainable=first_layers_trainable,
                          normalizer_fn=normalizer, weights_initializer=init)
        x = layers.conv2d(x, num_outputs=int(n_features_first/n_features_reduction_factor), kernel_size=5, stride=2,
                          activation_fn=leaky_relu, trainable=first_layers_trainable, normalizer_fn=normalizer,
                          weights_initializer=init)
        x = layers.conv2d(x, num_outputs=n_features_first, kernel_size=5, stride=2,
                          activation_fn=leaky_relu, trainable=last2_layer_trainable, normalizer_fn=normalizer,
                          weights_initializer=init)

        x = layers.flatten(x)
        return layers.fully_connected(x, num_outputs=1, activation_fn=None, trainable=last_layer_trainable,
                                      weights_initializer=init)


# split up discriminator into two parts, to make gradient computation less expensive, if first layers are fixed
def discriminator1(x, reuse, n_features_first=N_FEATURES_FIRST, n_features_reduction_factor=2, fix_first_layers=False,
                   architecture='WGANGP', init_method=None, **kwargs):

    first_layers_trainable = not fix_first_layers

    if architecture == 'DCGAN':
        normalizer = layers.batch_norm
    else:  # WGAN-GP
        normalizer = None

    if init_method in ['He']:
        init = layers.variance_scaling_initializer()
    else:
        init = layers.xavier_initializer()

    with tf.variable_scope('discriminator1', reuse=reuse):
        x = layers.conv2d(x, num_outputs=int(n_features_first/(n_features_reduction_factor**2)), kernel_size=5,
                          stride=2, activation_fn=leaky_relu, trainable=first_layers_trainable,
                          normalizer_fn=normalizer, weights_initializer=init)
        return layers.conv2d(x, num_outputs=int(n_features_first/n_features_reduction_factor), kernel_size=5, stride=2,
                             activation_fn=leaky_relu, trainable=first_layers_trainable, normalizer_fn=normalizer,
                             weights_initializer=init)


def discriminator2(x, reuse, n_features_first=N_FEATURES_FIRST, fix_last_layer=False, fix_2last_layer=False,
                   architecture='WGANGP', init_method=None, **kwargs):

    last_layer_trainable = not fix_last_layer
    last2_layer_trainable = not fix_2last_layer

    if architecture == 'DCGAN':
        normalizer = layers.batch_norm
    else:  # WGAN-GP
        normalizer = None

    if init_method in ['He']:
        init = layers.variance_scaling_initializer()
    else:
        init = layers.xavier_initializer()

    with tf.variable_scope('discriminator2', reuse=reuse):
        x = layers.conv2d(x, num_outputs=n_features_first, kernel_size=5, stride=2,
                          activation_fn=leaky_relu, trainable=last2_layer_trainable, normalizer_fn=normalizer,
                          weights_initializer=init)

        x = layers.flatten(x)
        return layers.fully_connected(x, num_outputs=1, activation_fn=None, trainable=last_layer_trainable,
                                      weights_initializer=init)


# TODO: extra fully-connected?
# add BN init method using init method -> done
# changed weight initialization to a.) normals -> done, b.) normals that are orthogonal -> doesnt make sense,
def train(input_dim=INPUT_DIM, batch_size=BATCH_SIZE, n_features_first=N_FEATURES_FIRST, critic_iters=CRITIC_ITERS,
          lambda_reg=LAMBDA, learning_rate=1e-4, epochs=ITERS, fixed_noise_size=FIXED_NOISE_SIZE,
          n_features_reduction_factor=2,
          fix_first_layers_gen=True, fix_last_layer_gen=False, fix_2last_layer_gen=False,
          fix_first_layers_disc=True, fix_last_layer_disc=False, fix_2last_layer_disc=False,
          architecture='WGANGP', use_unfixed_gradient_only=False, extra_fully_connected_layer=False,
          init_method=None, BN_layers_trainable=True, different_optimizers=False,
          load_saved=True):
    """
    - this is the function to use to train a GAN model for MNIST, with the configuration given by the parameters
    - the function computes losses and auto-saves the model every 100 steps and automatically resumes training where it
    stopped (when load_saved=True)

    :param input_dim:
    :param batch_size:
    :param n_features_first:
    :param critic_iters:
    :param lambda_reg:
    :param learning_rate:
    :param epochs:
    :param fixed_noise_size:
    :param n_features_reduction_factor: integer, e.g.: 1: use same number of feature-maps everywhere, 2: half the number
           of feature-maps in every step
    :param fix_first_layers_gen:
    :param fix_last_layer_gen:
    :param fix_2last_layer_gen:
    :param fix_first_layers_disc:
    :param fix_last_layer_disc:
    :param fix_2last_layer_disc:
    :param architecture: right now only supports 'WGANGP' and 'DCGAN', defaults to 'WGANGP'
    :param use_unfixed_gradient_only: whether to compute the gradient of the whole discriminator or only of the layers
           which are trainable
    :param extra_fully_connected_layer: not supported yet, whether to insert an extra fully connected layer in gen and
           disc towards the end
    :param init_method: the method with which the variables are initialized, support: 'uniform', 'normal',
           'truncated_normal' (each using std given by xavier initializer), 'normal1', 'truncated_normal1' (each using
           std 1), 'normal_BN', 'uniform_BN', 'normal_BN_shift', 'He', defaults to 'uniform'
    :param BN_layers_trainable: shall the BN layers be trainable
    :param different_optimizers: shall discriminator and generator be trained with different optimizers
    :param load_saved:
    :return:
    """

    # -------------------------------------------------------
    # setting for sending emails and getting statistics
    send = settings.send_email
    get_stats = settings.wgan_mnist_get_statistics

    # -------------------------------------------------------
    # architecture default
    if architecture not in ['DCGAN']:
        architecture = 'WGANGP'
    if architecture == 'DCGAN':
        critic_iters = 1

    # -------------------------------------------------------
    # init_method default
    if init_method not in ['normal', 'truncated_normal', 'normal1', 'truncated_normal1', 'normal_BN', 'uniform_BN',
                           'normal_BN_shift', 'He']:
        init_method = 'uniform'

    # -------------------------------------------------------
    # create unique folder name
    directory = 'training/'+str(input_dim)+'_'+str(batch_size)+'_'+str(n_features_first)+'_'+str(critic_iters)+'_'+\
                str(lambda_reg)+\
                '_'+ str(learning_rate)+'_'+ str(fixed_noise_size)+'_'+str(n_features_reduction_factor)+'_'+\
                str(fix_first_layers_gen)+'_'+str(fix_last_layer_gen)+'_'+str(fix_2last_layer_gen)+'_'+\
                str(fix_first_layers_disc)+'_'+str(fix_last_layer_disc)+'_'+str(fix_2last_layer_disc)+'_'+\
                str(architecture)+'_'+str(use_unfixed_gradient_only)+'_'+str(extra_fully_connected_layer)+\
                '_'+str(init_method)+'_'+str(BN_layers_trainable)+'_'+str(different_optimizers)+'/'
    samples_dir = directory+'samples/'
    model_dir = directory+'model/'

    # create directories if they don't exist
    if not os.path.isdir('training/'):
        call(['mkdir', 'training/'])

    if not os.path.isdir(directory):
        load_saved=False
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
    config.intra_op_parallelism_threads = settings.number_cpus
    config.inter_op_parallelism_threads = settings.number_cpus
    session = tf.Session(config=config)

    # -------------------------------------------------------
    # convenience function to build the model
    def build_model(fix_first_layers_gen_b=fix_first_layers_gen, fix_last_layer_gen_b=fix_last_layer_gen,
                    fix_2last_layer_gen_b=fix_2last_layer_gen,
                    fix_first_layers_disc_b=fix_first_layers_disc, fix_last_layer_disc_b=fix_last_layer_disc,
                    fix_2last_layer_disc_b=fix_2last_layer_disc
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
                                fix_2last_layer=fix_2last_layer_gen_b, architecture=architecture,
                                init_method=init_method)

        # # without splitting the discriminator
        # d_true = discriminator(x_true, reuse=False, n_features_first=n_features_first,
        #                        n_features_reduction_factor=n_features_reduction_factor,
        #                        fix_first_layers=fix_first_layers_disc, fix_last_layer=fix_last_layer_disc,
        #                        fix_2last_layer=fix_2last_layer_disc,
        #                        init_method=init_method)
        #
        # d_generated = discriminator(x_generated, reuse=True, n_features_first=n_features_first,
        #                             n_features_reduction_factor=n_features_reduction_factor,
        #                             fix_first_layers=fix_first_layers_disc, fix_last_layer=fix_last_layer_disc,
        #                             fix_2last_layer=fix_2last_layer_disc,
        #                             init_method=init_method)

        # splitting the discriminator
        d_true1 = discriminator1(x_true, reuse=False, n_features_first=n_features_first,
                                 n_features_reduction_factor=n_features_reduction_factor,
                                 fix_first_layers=fix_first_layers_disc_b, fix_last_layer=fix_last_layer_disc_b,
                                 fix_2last_layer=fix_2last_layer_disc_b, architecture=architecture,
                                 init_method=init_method)
        d_true = discriminator2(d_true1, reuse=False, n_features_first=n_features_first,
                                n_features_reduction_factor=n_features_reduction_factor,
                                fix_first_layers=fix_first_layers_disc_b, fix_last_layer=fix_last_layer_disc_b,
                                fix_2last_layer=fix_2last_layer_disc_b, architecture=architecture,
                                init_method=init_method)

        d_generated1 = discriminator1(x_generated, reuse=True, n_features_first=n_features_first,
                                      n_features_reduction_factor=n_features_reduction_factor,
                                      fix_first_layers=fix_first_layers_disc_b, fix_last_layer=fix_last_layer_disc_b,
                                      fix_2last_layer=fix_2last_layer_disc_b, architecture=architecture,
                                      init_method=init_method)
        d_generated = discriminator2(d_generated1, reuse=True, n_features_first=n_features_first,
                                     n_features_reduction_factor=n_features_reduction_factor,
                                     fix_first_layers=fix_first_layers_disc_b, fix_last_layer=fix_last_layer_disc_b,
                                     fix_2last_layer=fix_2last_layer_disc_b, architecture=architecture,
                                     init_method=init_method)

        if architecture == 'DCGAN':
            with tf.name_scope('loss'):
                g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_generated,
                                                                                labels=tf.ones_like(d_generated)))
                d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_generated,
                                                                                labels=tf.zeros_like(d_generated))) +\
                         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_true,
                                                                                labels=tf.ones_like(d_true)))
                d_loss = d_loss/2.

            if different_optimizers:
                with tf.name_scope('g_optimizer'):
                    g_optimizer = tf.train.AdamOptimizer(learning_rate=2 * learning_rate, beta1=0.5)

                    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                    # make BN layers trainable or not, depending on BN_layers_trainable
                    g_vars2 = []
                    for v in g_vars:
                        if 'BatchNorm' not in v.name:
                            g_vars2 += [v]
                    if not BN_layers_trainable:
                        g_vars = g_vars2
                    g_train = g_optimizer.minimize(g_loss, var_list=g_vars)
                with tf.name_scope('d_optimizer'):
                    d_optimizer = tf.train.AdamOptimizer(learning_rate=2 * learning_rate, beta1=0.5)

                    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                    # make BN layers trainable or not, depending on BN_layers_trainable
                    d_vars2 = []
                    for v in d_vars:
                        if 'BatchNorm' not in v.name:
                            d_vars2 += [v]
                    if not BN_layers_trainable:
                        d_vars = d_vars2
                    d_train = d_optimizer.minimize(d_loss, var_list=d_vars)
            else:  # just one optimizer for discriminator and generator
                with tf.name_scope('optimizer'):
                    optimizer = tf.train.AdamOptimizer(learning_rate=2*learning_rate, beta1=0.5)
                    d_optimizer = optimizer
                    g_optimizer = optimizer

                    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                    # make BN layers trainable or not, depending on BN_layers_trainable
                    g_vars2 = []
                    for v in g_vars:
                        if 'BatchNorm' not in v.name:
                            g_vars2 += [v]
                    if not BN_layers_trainable:
                        g_vars = g_vars2
                    g_train = optimizer.minimize(g_loss, var_list=g_vars)

                    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                    # make BN layers trainable or not, depending on BN_layers_trainable
                    d_vars2 = []
                    for v in d_vars:
                        if 'BatchNorm' not in v.name:
                            d_vars2 += [v]
                    if not BN_layers_trainable:
                        d_vars = d_vars2
                    d_train = optimizer.minimize(d_loss, var_list=d_vars)

        else:  # WGAN-GP
            with tf.name_scope('regularizer'):
                epsilon = tf.random_uniform([batch_size, 1, 1, 1], 0.0, 1.0)
                x_hat = epsilon * x_true + (1 - epsilon) * x_generated

                # # without splitting the discriminator
                # d_hat = discriminator(x_hat, reuse=True, n_features_first=n_features_first,
                #                       n_features_reduction_factor=n_features_reduction_factor,
                #                       fix_first_layers=fix_first_layers_disc, fix_last_layer=fix_last_layer_disc,
                #                       fix_2last_layer=fix_2last_layer_disc)
                #
                # gradients = tf.gradients(d_hat, x_hat)[0]
                # ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
                # d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)

                # splitting the discriminator
                d_hat1 = discriminator1(x_hat, reuse=True, n_features_first=n_features_first,
                                        n_features_reduction_factor=n_features_reduction_factor,
                                        fix_first_layers=fix_first_layers_disc_b, fix_last_layer=fix_last_layer_disc_b,
                                        fix_2last_layer=fix_2last_layer_disc_b, architecture=architecture,
                                        init_method=init_method)
                d_hat = discriminator2(d_hat1, reuse=True, n_features_first=n_features_first,
                                       n_features_reduction_factor=n_features_reduction_factor,
                                       fix_first_layers=fix_first_layers_disc_b, fix_last_layer=fix_last_layer_disc_b,
                                       fix_2last_layer=fix_2last_layer_disc_b, architecture=architecture,
                                       init_method=init_method)

                if use_unfixed_gradient_only:
                    # TODO: is there a better way to compute the partly gradient
                    gradients = tf.gradients(d_hat, d_hat1)[0]
                    ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
                    ddx = tf.reduce_mean(ddx, axis=1, keepdims=True)
                else:
                    gradients = tf.gradients(d_hat, x_hat)[0]
                    ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
                d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)

            with tf.name_scope('loss'):
                g_loss = -tf.reduce_mean(d_generated)
                wasserstein_dist = tf.reduce_mean(d_true) - tf.reduce_mean(d_generated)
                d_loss = -wasserstein_dist + lambda_reg * d_regularizer

            if different_optimizers:
                with tf.name_scope('g_optimizer'):
                    g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0, beta2=0.9)

                    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                    g_train = g_optimizer.minimize(g_loss, var_list=g_vars)
                with tf.name_scope('d_optimizer'):
                    d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0, beta2=0.9)

                    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                    d_train = d_optimizer.minimize(d_loss, var_list=d_vars)
            else:  # just one optimizer for discriminator and generator
                with tf.name_scope('optimizer'):
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0, beta2=0.9)
                    # FK: TODO: beta1 = 0.5 in IWGAN, here 0 -> change? In experiments (only 1000 epochs) it seemed better with 0
                    d_optimizer = optimizer
                    g_optimizer = optimizer

                    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                    g_train = optimizer.minimize(g_loss, var_list=g_vars)

                    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                    d_train = optimizer.minimize(d_loss, var_list=d_vars)

        # initialize variables using uniform xavier init method, see tensorflow documentation
        session.run(tf.global_variables_initializer())

        if architecture == 'DCGAN':
            return x_true, z, x_generated, d_true1, d_true, d_generated1, d_generated, g_loss, d_loss, g_optimizer, \
                   d_optimizer, g_vars, g_train, d_vars, d_train
        else:  # WGANGP
            return x_true, z, x_generated, d_true1, d_true, d_generated1, d_generated, epsilon, x_hat, d_hat, \
                   gradients, ddx, d_regularizer, \
                   g_loss, wasserstein_dist, d_loss, g_optimizer, d_optimizer, g_vars, g_train, d_vars, d_train

    # -------------------------------------------------------
    # build the model
    if (init_method in ['uniform', 'He']) or load_saved:
        if architecture == 'DCGAN':
            x_true, z, x_generated, d_true1, d_true, d_generated1, d_generated, g_loss, d_loss, g_optimizer, \
            d_optimizer, g_vars, g_train, d_vars, d_train = build_model()
        else:  # WGANGP
            x_true, z, x_generated, d_true1, d_true, d_generated1, d_generated, epsilon, x_hat, d_hat, gradients, ddx,\
            d_regularizer, g_loss, wasserstein_dist, d_loss, g_optimizer, d_optimizer, g_vars, g_train, d_vars, \
            d_train = build_model()
    else:  # not load_saved and not 'uniform'
        # build model with all variables trainable to be able to change weights
        if architecture == 'DCGAN':
            x_true, z, x_generated, d_true1, d_true, d_generated1, d_generated, g_loss, d_loss, g_optimizer, \
            d_optimizer, g_vars, g_train, d_vars, d_train = build_model(False, False, False, False, False, False)
        else:  # WGANGP
            x_true, z, x_generated, d_true1, d_true, d_generated1, d_generated, epsilon, x_hat, d_hat, gradients, ddx,\
            d_regularizer, g_loss, wasserstein_dist, d_loss, g_optimizer, d_optimizer, g_vars, g_train, d_vars, \
            d_train = build_model(False, False, False, False, False, False)

        # change the weights as wanted
        saver = tf.train.Saver(max_to_keep=1)
        trainable_vars = tf.trainable_variables()
        if get_stats:
            import matplotlib.pyplot as plt
        for v in trainable_vars:
            print 'change weights of: '+str(v.name)
            # TODO: is it good to change also BN weights??
            weights = session.run(v)
            # if 'BatchNorm' in v.name: #delete
            #     print 'BN weights:' #delete
            #     print weights #delete
            #     print #delete
            if init_method == 'normal':  # using xavier init method, see tensorflow documentation
                max_abs_val = np.max(np.abs(weights))
                session.run(tf.assign(v, value=tf.random_normal(v.shape, mean=0.0, stddev=max_abs_val/np.sqrt(3))))
            elif init_method == 'truncated_normal':  # using xavier init method, see tensorflow documentation
                max_abs_val = np.max(np.abs(weights))
                session.run(tf.assign(v, value=tf.truncated_normal(v.shape, mean=0.0, stddev=max_abs_val/np.sqrt(3))))
            elif init_method == 'normal1':
                session.run(tf.assign(v, value=tf.random_normal(v.shape, mean=0.0, stddev=1.0)))
            elif init_method == 'truncated_normal1':
                session.run(tf.assign(v, value=tf.truncated_normal(v.shape, mean=0.0, stddev=1.0)))
            elif init_method == 'uniform_BN':
                max_abs_val = np.max(np.abs(weights))
                if 'BatchNorm' in v.name:
                    session.run(tf.assign(v, value=tf.random_uniform(v.shape, minval=-last_val, maxval=last_val)))
                last_val = max_abs_val
            elif init_method == 'normal_BN':
                max_abs_val = np.max(np.abs(weights))
                if 'BatchNorm' in v.name:
                    session.run(tf.assign(v, value=tf.random_normal(v.shape, mean=0.0, stddev=last_val/np.sqrt(3))))
                else:
                    session.run(tf.assign(v, value=tf.random_normal(v.shape, mean=0.0, stddev=max_abs_val/np.sqrt(3))))
                last_val = max_abs_val
            elif init_method == 'normal_BN_shift':
                max_abs_val = np.max(np.abs(weights))
                if 'BatchNorm' in v.name:
                    session.run(tf.assign(v, value=tf.random_normal(v.shape, mean=-last_val, stddev=last_val)))
                else:
                    session.run(
                        tf.assign(v, value=tf.random_normal(v.shape, mean=0.0, stddev=max_abs_val / np.sqrt(3))))
                last_val = max_abs_val

            if get_stats:
                weights_new = session.run(v)
                f = plt.figure()
                plt.hist(np.reshape(weights_new, newshape=(-1,)), bins=100, density=True)
                f.savefig(fname=directory + v.name.replace('/', '_').replace(':','') + '.png')
                plt.close(f)

        saver.save(sess=session, save_path=model_dir + 'saved_model')
        print
        print 'weights were initialized with: '+init_method
        print

        # load new session, so that no conflict with names in the name_scopes
        session.close()
        tf.reset_default_graph()
        session = tf.Session(config=config)

        # load the model with the perturbed weights, but now s.t. the correct variables are trainable
        if architecture == 'DCGAN':
            x_true, z, x_generated, d_true1, d_true, d_generated1, d_generated, g_loss, d_loss, g_optimizer, \
            d_optimizer, g_vars, g_train, d_vars, d_train = build_model()
        else:  # WGANGP
            x_true, z, x_generated, d_true1, d_true, d_generated1, d_generated, epsilon, x_hat, d_hat, gradients, ddx,\
            d_regularizer, g_loss, wasserstein_dist, d_loss, g_optimizer, d_optimizer, g_vars, g_train, d_vars, \
            d_train = build_model()

        # restore the model with the correctly initialized weights
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(sess=session, save_path=model_dir + 'saved_model')
        print 'loaded model with weights initialized with: '+init_method
        print

    # -------------------------------------------------------
    # FK: For saving samples, taken from IWGAN
    fixed_noise = np.random.normal(size=(fixed_noise_size, input_dim)).astype('float32')

    def generate_image(frame):
        samples = session.run(x_generated, feed_dict={z: fixed_noise}).squeeze()
        # print samples.shape
        save_images.save_images(
            samples.reshape((fixed_noise_size, 28, 28)),
            samples_dir + 'iteration_{}.png'.format(frame)
        )

    # -------------------------------------------------------
    # FK: for saving the model create a saver
    saver = tf.train.Saver(max_to_keep=1)
    epochs_trained = 0
    if architecture == 'DCGAN':
        training_progress = pd.DataFrame(data=None, index=None, columns=['epoch', 'time', 'd_loss'])
    else:  # WGAN-GP
        training_progress = pd.DataFrame(data=None, index=None, columns=['epoch', 'time', 'Wasserstein_dist', 'd_loss'])

    # restore the model:
    if load_saved:
        saver.restore(sess=session, save_path=model_dir+'saved_model')
        epochs_trained = int(np.loadtxt(fname=model_dir+'epochs.csv'))
        tp_app = pd.read_csv(filepath_or_buffer=directory+'training_progress.csv', index_col=0, header=0)
        training_progress = pd.concat([training_progress, tp_app], axis=0, ignore_index=True)
        print 'loaded training progress, and the model, which was already trained for {} epochs'.format(epochs_trained)
        print training_progress
        print

    # if the network is already trained completely, set send to false
    if epochs_trained == epochs:
        send = False

    # -------------------------------------------------------
    # FK: print and get model summary
    n_params_gen = model_summary(scope='generator', BN_layers_trainable=BN_layers_trainable)[0]
    print
    n_params_disc = model_summary(scope='discriminator', BN_layers_trainable=BN_layers_trainable)[0]
    print

    # -------------------------------------------------------
    # FK: print model config to file
    model_config = [['input_dim', 'batch_size', 'n_features_first', 'critic_iters',
                     'lambda_reg', 'learning_rate', 'fixed_noise_size', 'n_features_reduction_factor',
                     'fix_first_layers_gen', 'fix_last_layer_gen', 'fix_2last_layer_gen',
                     'fix_first_layers_disc', 'fix_last_layer_disc', 'fix_2last_layer_disc',
                     'architecture', 'use_unfixed_gradient_only', 'extra_fully_connected_layer', 'init_method',
                     'BN_layers_trainable', 'different_optimizers',
                     'n_trainable_params_gen', 'n_trainable_params_disc'],
                    [input_dim, batch_size, n_features_first, critic_iters,
                     lambda_reg, learning_rate, fixed_noise_size, n_features_reduction_factor,
                     fix_first_layers_gen, fix_last_layer_gen, fix_2last_layer_gen,
                     fix_first_layers_disc, fix_last_layer_disc, fix_2last_layer_disc,
                     architecture, use_unfixed_gradient_only, extra_fully_connected_layer, init_method,
                     BN_layers_trainable, different_optimizers,
                     n_params_gen, n_params_disc]]
    model_config = np.transpose(model_config)
    model_config = pd.DataFrame(data=model_config)
    model_config.to_csv(path_or_buf=directory+'model_config.csv')
    print 'saved model configuration'
    print

    # -------------------------------------------------------
    # FK: get the MNIST data loader
    train_gen, dev_gen, test_gen = mnist.load(batch_size, batch_size)

    # create an infinite generator
    def inf_train_gen():
        while True:
            for images, targets in train_gen():
                yield images

    gen = inf_train_gen()

    # -------------------------------------------------------
    # training loop
    t = time.time()  # get start time

    # for average times:
    if get_stats:
        t1s = np.zeros((epochs-epochs_trained))
        t2s = np.zeros((epochs-epochs_trained))
        t3s = np.zeros((epochs-epochs_trained))
        t4s = np.zeros((epochs-epochs_trained))

    for i in xrange(epochs-epochs_trained):
        z_train = np.random.randn(batch_size, input_dim)
        if get_stats:
            tt1 = time.time()
        session.run(g_train, feed_dict={z: z_train})
        if get_stats:
            tt1 = time.time() - tt1

        # loop for critic training
        for j in xrange(critic_iters):
            # FK: insert the following 3 lines s.t. not the same batch is used for all 5 discriminator updates
            if get_stats:
                tt = time.time()
            batch = gen.next()
            images = batch.reshape([-1, 28, 28, 1])
            z_train = np.random.randn(batch_size, input_dim)
            if get_stats:
                print '\ncomputation time to get true batch and random vector: {}'.format(time.time()-tt)
                tt = time.time()
            session.run(d_train, feed_dict={x_true: images, z: z_train})
            if get_stats:
                t1 = time.time() - tt + tt1
                t1s[i] = t1
                print 'computation time to train for 1 epoch (minimize disc and gen one step): t1 = {}'.format(t1)
                tt = time.time()
                session.run(d_loss, feed_dict={x_true: images, z: z_train})
                session.run(g_loss, feed_dict={z: z_train})
                t2 = time.time() -tt
                t2s[i] = t2
                print 'computation time to compute the disc. and gen. loss once: t2 = {}'.format(t2)
                tt = time.time()
                session.run(x_generated, feed_dict={z: z_train})
                t3 = time.time() - tt
                t3s[i] = t3
                print 'computation time to compute x_generated: t3 = {}'.format(t3)
                if architecture == 'WGANGP':
                    tt = time.time()
                    session.run(d_regularizer, feed_dict={x_true: images, z: z_train})
                    t4 = time.time() - tt
                    t4s[i] = t4
                    print 'computation time to compute gradient regularization term: t4 = {}'.format(t4)
                print 't1/t2 = {}'.format(t1/t2)
                # list_ = session.run(g_optimizer.compute_gradients(g_loss, var_list=g_vars), feed_dict={z: z_train})
                # print 'number of gradients computed: {}'.format(2*len(list_))

        # print the current epoch
        print('epoch={}/{}'.format(i+epochs_trained+1, epochs))

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

    # average times:
    if get_stats:
        print '\n\naverage times over {} epochs:'.format(epochs - epochs_trained)
        print 'computation time to train for 1 epoch (minimize disc and gen one step): t1 = {}'.format(np.mean(t1s))
        print 'computation time to compute the disc. and gen. loss once: t2 = {}'.format(np.mean(t2s))
        print 'computation time to compute x_generated: t3 = {}'.format(np.mean(t3s))
        if architecture == 'WGANGP':
            print 'computation time to compute gradient regularization term: t4 = {}'.format(np.mean(t4s))
        print 't1/t2 = {}'.format(np.mean(t1s) / np.mean(t2s))
        print

    # -------------------------------------------------------
    # after training close the session
    session.close()
    tf.reset_default_graph()

    # -------------------------------------------------------
    # when training is done send email
    if send:
        subject = 'GAN (MNIST) training finished'
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
        parameters = [(INPUT_DIM, BATCH_SIZE, N_FEATURES_FIRST, CRITIC_ITERS, LAMBDA, 1e-4, ITERS, FIXED_NOISE_SIZE, 2,
                       True, False, False, True, False, False, 'WGANGP', False, False, None, True, False),
                      (INPUT_DIM, BATCH_SIZE, N_FEATURES_FIRST, CRITIC_ITERS, LAMBDA, 1e-4, ITERS, FIXED_NOISE_SIZE, 2,
                       True, True, False, True, False, False, 'WGANGP', False, False, None, True, False)]
    results = Parallel(n_jobs=nb_jobs)(delayed(train)(input_dim, batch_size, n_features_first, critic_iters, lambda_reg,
                                                      learning_rate, epochs, fixed_noise_size,
                                                      n_features_reduction_factor, fix_first_layers_gen,
                                                      fix_last_layer_gen, fix_2last_layer_gen, fix_first_layers_disc,
                                                      fix_last_layer_disc, fix_2last_layer_disc,
                                                      architecture, use_unfixed_gradient_only,
                                                      extra_fully_connected_layer, init_method,
                                                      BN_layers_trainable, different_optimizers)
                                       for input_dim, batch_size, n_features_first, critic_iters, lambda_reg,
                                           learning_rate, epochs, fixed_noise_size,
                                           n_features_reduction_factor, fix_first_layers_gen,
                                           fix_last_layer_gen, fix_2last_layer_gen, fix_first_layers_disc,
                                           fix_last_layer_disc, fix_2last_layer_disc, architecture,
                                           use_unfixed_gradient_only, extra_fully_connected_layer,
                                           init_method, BN_layers_trainable, different_optimizers in parameters)

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
    param_array = settings.param_array7
    nb_jobs = settings.number_parallel_jobs

    parallel_training(parameters=param_array, nb_jobs=nb_jobs)

    # train(n_features_reduction_factor=2, use_unfixed_gradient_only=False, architecture='DCGAN',
    #       fix_first_layers_gen=True, fix_first_layers_disc=True, fix_2last_layer_disc=True, fix_2last_layer_gen=True,
    #       # fix_first_layers_disc=False, fix_first_layers_gen=False,
    #       BN_layers_trainable=False, different_optimizers=False,
    #       init_method='He',
    #       # epochs=100,
    #       load_saved=True)













