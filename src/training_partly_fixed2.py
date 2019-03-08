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
BN = True

if settings.euler:
    N_CPUS_TF = 1  # set to None to use value of settings.py
    N_CPUS_PARALLEL = 4  # set to None to use value of settings.py
else:
    N_CPUS_TF = 3  # set to None to use value of settings.py
    N_CPUS_PARALLEL = 1  # set to None to use value of settings.py


# FK: model summary of
def model_summary(var_list):
    return slim.model_analyzer.analyze_vars(var_list, print_info=True)


def leaky_relu(x):
    return tf.maximum(x, 0.2 * x)


def generator(z, n_features_first=N_FEATURES_FIRST, n_features_reduction_factor=2, fix_layer_1=False,
              fix_layer_2=False, fix_layer_3=False, fix_layer_4=False, architecture='WGANGP', init_method=None):

    if architecture == 'DCGAN' and BN:
        normalizer = layers.batch_norm
    else:  # WGAN-GP
        normalizer = None

    if init_method in ['He']:
        init = layers.variance_scaling_initializer()
    elif init_method == 'normal':
        init = layers.xavier_initializer(uniform=False)
    else:
        init = layers.xavier_initializer()

    # the layers use relu activations (default)
    with tf.variable_scope('generator'):
        z = layers.fully_connected(z, num_outputs=4*4*n_features_first, trainable=(not fix_layer_1),
                                   normalizer_fn=normalizer, weights_initializer=init)
        z = tf.reshape(z, [-1, 4, 4, n_features_first])  # we use the dimensions as NHWC resp. BHWC

        z = layers.conv2d_transpose(z, num_outputs=int(n_features_first/n_features_reduction_factor), kernel_size=5,
                                    stride=2, trainable=(not fix_layer_2), normalizer_fn=normalizer,
                                    weights_initializer=init)
        z = layers.conv2d_transpose(z, num_outputs=int(n_features_first/(n_features_reduction_factor**2)),
                                    kernel_size=5, stride=2, trainable=(not fix_layer_3), normalizer_fn=normalizer,
                                    weights_initializer=init)
        z = layers.conv2d_transpose(z, num_outputs=1, kernel_size=5, stride=2, activation_fn=tf.nn.sigmoid,
                                    trainable=(not fix_layer_4), weights_initializer=init)
        return z[:, 2:-2, 2:-2, :]  # FK: of the 32x32 image leave away the outer border of 4 pixels to get a 28x28
        # image as in the training set of MNIST


def discriminator(x, reuse, n_features_first=N_FEATURES_FIRST, n_features_reduction_factor=2, fix_layer_1=False,
                  fix_layer_2=False, fix_layer_3=False, fix_layer_4=False, architecture='WGANGP', init_method=None):

    if architecture == 'DCGAN' and BN:
        normalizer = layers.batch_norm
    else:  # WGAN-GP
        normalizer = None

    if init_method in ['He']:
        init = layers.variance_scaling_initializer()
    elif init_method == 'normal':
        init = layers.xavier_initializer(uniform=False)
    else:
        init = layers.xavier_initializer()

    with tf.variable_scope('discriminator', reuse=reuse):
        x = layers.conv2d(x, num_outputs=int(n_features_first/(n_features_reduction_factor**2)), kernel_size=5,
                          stride=2, activation_fn=leaky_relu, trainable=(not fix_layer_1),
                          normalizer_fn=normalizer, weights_initializer=init)
        x = layers.conv2d(x, num_outputs=int(n_features_first/n_features_reduction_factor), kernel_size=5, stride=2,
                          activation_fn=leaky_relu, trainable=(not fix_layer_2), normalizer_fn=normalizer,
                          weights_initializer=init)
        x = layers.conv2d(x, num_outputs=n_features_first, kernel_size=5, stride=2,
                          activation_fn=leaky_relu, trainable=(not fix_layer_3), normalizer_fn=normalizer,
                          weights_initializer=init)

        x = layers.flatten(x)
        return layers.fully_connected(x, num_outputs=1, activation_fn=None, trainable=(not fix_layer_4),
                                      weights_initializer=init)


def train(input_dim=INPUT_DIM, batch_size=BATCH_SIZE, n_features_first=N_FEATURES_FIRST, critic_iters=CRITIC_ITERS,
          lambda_reg=LAMBDA, learning_rate=1e-4, iterations=ITERS, fixed_noise_size=FIXED_NOISE_SIZE,
          n_features_reduction_factor=2,
          gen_fix_layer_1=False, gen_fix_layer_2=False, gen_fix_layer_3=False, gen_fix_layer_4=False,
          disc_fix_layer_1=False, disc_fix_layer_2=False, disc_fix_layer_3=False, disc_fix_layer_4=False,
          architecture='DCGAN', init_method='He', BN_layers_trainable=True,
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
    :param iterations:
    :param fixed_noise_size:
    :param n_features_reduction_factor: integer, e.g.: 1: use same number of feature-maps everywhere, 2: half the number
           of feature-maps in every step
    :param architecture: right now only supports 'WGANGP' and 'DCGAN', defaults to 'DCGAN'
    :param init_method: the method with which the variables are initialized, support: 'uniform', 'normal',
           'truncated_normal' (each using std given by xavier initializer), 'normal1', 'truncated_normal1' (each using
           std 1), 'normal_BN', 'uniform_BN', 'normal_BN_shift', 'He', defaults to 'He'
    :param BN_layers_trainable: shall the BN layers be trainable
    :param load_saved:
    :return:
    """

    # -------------------------------------------------------
    # setting for sending emails and getting statistics
    send = settings.send_email
    get_stats = settings.get_statistics

    # -------------------------------------------------------
    # architecture default
    if architecture not in ['WGANGP']:
        architecture = 'DCGAN'
    if architecture == 'DCGAN':
        lambda_reg = None

    # -------------------------------------------------------
    # init_method default
    if init_method not in ['normal', 'truncated_normal', 'normal1', 'truncated_normal1', 'normal_BN', 'uniform_BN',
                           'normal_BN_shift', 'He', 'LayerDistribution']:
        init_method = 'uniform'

    # -------------------------------------------------------
    # create unique folder name
    dir1 = 'partly_fixed2/'
    directory = dir1+str(input_dim)+'_'+str(batch_size)+'_'+str(n_features_first)+'_'+str(critic_iters)+'_'+\
                str(lambda_reg)+'_'+str(learning_rate)+'_'+str(n_features_reduction_factor)+'_'+\
                str(gen_fix_layer_1)+'_'+str(gen_fix_layer_2)+'_'+str(gen_fix_layer_3)+'_'+str(gen_fix_layer_4)+'_' + \
                str(disc_fix_layer_1) + '_' + str(disc_fix_layer_2) + '_' + str(disc_fix_layer_3) + '_' + \
                str(disc_fix_layer_4) + '_' + \
                str(architecture)+'_'+str(init_method)+'_'+str(BN_layers_trainable)+'_'+str(BN)+'/'
    samples_dir = directory+'samples/'
    model_dir = directory+'model/'

    # create directories if they don't exist
    if not os.path.isdir(dir1):
        call(['mkdir', dir1])

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
    if N_CPUS_TF is None:
        number_cpus_tf = settings.number_cpus
    else:
        number_cpus_tf = N_CPUS_TF
    config.intra_op_parallelism_threads = number_cpus_tf
    config.inter_op_parallelism_threads = number_cpus_tf
    session = tf.Session(config=config)

    # -------------------------------------------------------
    # convenience function to build the model
    def build_model(gen_fix_layer_1_b=gen_fix_layer_1, gen_fix_layer_2_b=gen_fix_layer_2,
                    gen_fix_layer_3_b=gen_fix_layer_3, gen_fix_layer_4_b=gen_fix_layer_4,
                    disc_fix_layer_1_b=disc_fix_layer_1, disc_fix_layer_2_b=disc_fix_layer_2,
                    disc_fix_layer_3_b=disc_fix_layer_3, disc_fix_layer_4_b=disc_fix_layer_4):
        with tf.name_scope('placeholders'):
            x_true = tf.placeholder(tf.float32, [None, 28, 28, 1])
            z = tf.placeholder(tf.float32, [None, input_dim])

        x_generated = generator(z, n_features_first=n_features_first,
                                n_features_reduction_factor=n_features_reduction_factor,
                                fix_layer_1=gen_fix_layer_1_b, fix_layer_2=gen_fix_layer_2_b,
                                fix_layer_3=gen_fix_layer_3_b, fix_layer_4=gen_fix_layer_4_b,
                                architecture=architecture,
                                init_method=init_method)

        d_true = discriminator(x_true, reuse=False, n_features_first=n_features_first,
                               n_features_reduction_factor=n_features_reduction_factor,
                               fix_layer_1=disc_fix_layer_1_b, fix_layer_2=disc_fix_layer_2_b,
                               fix_layer_3=disc_fix_layer_3_b, fix_layer_4=disc_fix_layer_4_b,
                               architecture=architecture,
                               init_method=init_method)

        d_generated = discriminator(x_generated, reuse=True, n_features_first=n_features_first,
                                    n_features_reduction_factor=n_features_reduction_factor,
                                    fix_layer_1=disc_fix_layer_1_b, fix_layer_2=disc_fix_layer_2_b,
                                    fix_layer_3=disc_fix_layer_3_b, fix_layer_4=disc_fix_layer_4_b,
                                    architecture=architecture,
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

            with tf.name_scope('g_optimizer'):
                g_optimizer = tf.train.AdamOptimizer(learning_rate=2 * learning_rate, beta1=0.5)

                g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                # make BN layers trainable or not, depending on BN_layers_trainable
                g_vars2 = []
                not_to_include = []
                if gen_fix_layer_1_b:
                    not_to_include += ['generator/fully_connected/BatchNorm/beta:0']
                if gen_fix_layer_2_b:
                    not_to_include += ['generator/Conv2d_transpose/BatchNorm/beta:0']
                if gen_fix_layer_3_b:
                    not_to_include += ['generator/Conv2d_transpose_1/BatchNorm/beta:0']
                if disc_fix_layer_1_b:
                    not_to_include += ['discriminator/Conv/BatchNorm/beta:0']
                if disc_fix_layer_2_b:
                    not_to_include += ['discriminator/Conv_1/BatchNorm/beta:0']
                if disc_fix_layer_3_b:
                    not_to_include += ['discriminator/Conv_2/BatchNorm/beta:0']
                for v in g_vars:
                    if v.name not in not_to_include:
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
                    if v.name not in not_to_include:
                        d_vars2 += [v]
                if not BN_layers_trainable:
                    d_vars = d_vars2
                d_train = d_optimizer.minimize(d_loss, var_list=d_vars)

        else:  # WGAN-GP
            with tf.name_scope('regularizer'):
                epsilon = tf.random_uniform([batch_size, 1, 1, 1], 0.0, 1.0)
                x_hat = epsilon * x_true + (1 - epsilon) * x_generated

                d_hat = discriminator(x_hat, reuse=True, n_features_first=n_features_first,
                                      n_features_reduction_factor=n_features_reduction_factor,
                                      fix_layer_1=disc_fix_layer_1_b, fix_layer_2=disc_fix_layer_2_b,
                                      fix_layer_3=disc_fix_layer_3_b, fix_layer_4=disc_fix_layer_4_b,
                                      architecture=architecture,
                                      init_method=init_method)

                gradients = tf.gradients(d_hat, x_hat)[0]
                ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
                d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)

            with tf.name_scope('loss'):
                g_loss = -tf.reduce_mean(d_generated)
                wasserstein_dist = tf.reduce_mean(d_true) - tf.reduce_mean(d_generated)
                d_loss = -wasserstein_dist + lambda_reg * d_regularizer

            with tf.name_scope('g_optimizer'):
                g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0, beta2=0.9)

                g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                g_train = g_optimizer.minimize(g_loss, var_list=g_vars)
            with tf.name_scope('d_optimizer'):
                d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0, beta2=0.9)

                d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                d_train = d_optimizer.minimize(d_loss, var_list=d_vars)

        # initialize variables using uniform xavier init method, see tensorflow documentation
        session.run(tf.global_variables_initializer())

        if architecture == 'DCGAN':
            return x_true, z, x_generated, g_loss, d_loss, g_train, d_train, g_vars, d_vars
        else:  # WGANGP
            return x_true, z, x_generated, g_loss, wasserstein_dist, d_loss, g_train, d_train, g_vars, d_vars

    # -------------------------------------------------------
    # build the model
    if (init_method in ['uniform', 'He', 'normal']) or load_saved:
        if architecture == 'DCGAN':
            x_true, z, x_generated, g_loss, d_loss, g_train, d_train, g_vars, d_vars = build_model()
        else:  # WGANGP
            x_true, z, x_generated, g_loss, wasserstein_dist, d_loss, g_train, d_train, g_vars, d_vars = build_model()
    else:  # not load_saved and not 'uniform'
        # build model with all variables trainable to be able to change weights
        if architecture == 'DCGAN':
            x_true, z, x_generated, g_loss, d_loss, g_train, d_train, \
            g_vars, d_vars = build_model(False, False, False, False,False, False, False, False)
        else:  # WGANGP
            x_true, z, x_generated, g_loss, wasserstein_dist, d_loss, g_train, \
            d_train, g_vars, d_vars = build_model(False, False, False, False, False, False, False, False)

        # change the weights as wanted
        saver = tf.train.Saver(max_to_keep=1)
        trainable_vars = tf.trainable_variables()
        if get_stats:
            import matplotlib.pyplot as plt
        for v in trainable_vars:
            print 'change weights of: '+str(v.name)
            weights = session.run(v)
            # if 'BatchNorm' in v.name: #delete
            #     print 'BN weights:' #delete
            #     print weights #delete
            #     print #delete

            if init_method == 'truncated_normal':  # using xavier init method, see tensorflow documentation
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
            elif init_method == 'LayerDistribution':
                if v.name == 'generator/fully_connected/weights:0':
                    session.run(tf.assign(v, value=tf.random_normal(v.shape, mean=0.0, stddev=0.037907723)))
                elif v.name == 'generator/Conv2d_transpose/weights:0':
                    session.run(tf.assign(v, value=tf.random_normal(v.shape, mean=-0.007851141, stddev=0.034838371)))
                elif v.name == 'generator/Conv2d_transpose_1/weights:0':
                    session.run(tf.assign(v, value=tf.random_normal(v.shape, mean=-0.001966879, stddev=0.037020162)))
                elif v.name == 'generator/Conv2d_transpose_2/weights:0':
                    session.run(tf.assign(v, value=tf.random_normal(v.shape, mean=-0.121885814, stddev=0.294095486)))
                elif v.name == 'discriminator/Conv/weights:0':
                    session.run(tf.assign(v, value=tf.random_normal(v.shape, mean=-0.005809855, stddev=0.044240803)))
                elif v.name == 'discriminator/Conv_1/weights:0':
                    session.run(tf.assign(v, value=tf.random_normal(v.shape, mean=-0.000329115, stddev=0.03293338)))
                elif v.name == 'discriminator/Conv_2/weights:0':
                    session.run(tf.assign(v, value=tf.random_normal(v.shape, mean=-0.000697783, stddev=0.028810507)))
                elif v.name == 'discriminator/fully_connected/weights:0':
                    session.run(tf.assign(v, value=tf.random_normal(v.shape, mean=0.000849896, stddev=0.074863143)))

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
            x_true, z, x_generated, g_loss, d_loss, g_train, d_train, g_vars, d_vars = build_model()
        else:  # WGANGP
            x_true, z, x_generated, g_loss, wasserstein_dist, d_loss, g_train, d_train, g_vars, d_vars = build_model()

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
    iterations_trained = 0
    if architecture == 'DCGAN':
        training_progress = pd.DataFrame(data=None, index=None, columns=['iteration', 'time', 'd_loss'])
    else:  # WGAN-GP
        training_progress = pd.DataFrame(data=None, index=None,
                                         columns=['iteration', 'time', 'Wasserstein_dist', 'd_loss'])

    # restore the model:
    if load_saved:
        saver.restore(sess=session, save_path=model_dir+'saved_model')
        iterations_trained = int(np.loadtxt(fname=model_dir+'iterations.csv'))
        tp_app = pd.read_csv(filepath_or_buffer=directory+'training_progress.csv', index_col=0, header=0)
        training_progress = pd.concat([training_progress, tp_app], axis=0, ignore_index=True)
        print 'loaded training progress, and the model, which was already trained for {} iterations'.format(iterations_trained)
        print training_progress
        print

    # if the network is already trained completely, set send to false
    if iterations_trained == iterations:
        send = False

    # -------------------------------------------------------
    # FK: print and get model summary
    n_params_gen = model_summary(var_list=g_vars)[0]
    print
    n_params_disc = model_summary(var_list=d_vars)[0]
    print

    # -------------------------------------------------------
    # FK: print model config to file
    model_config = [['input_dim', 'batch_size', 'n_features_first', 'critic_iters',
                     'lambda_reg', 'learning_rate', 'fixed_noise_size', 'n_features_reduction_factor',
                     'gen_fix_layer_1', 'gen_fix_layer_2', 'gen_fix_layer_3', 'gen_fix_layer_4',
                     'disc_fix_layer_1', 'disc_fix_layer_2', 'disc_fix_layer_3', 'disc_fix_layer_4',
                     'architecture', 'init_method', 'BN_layers_trainable',
                     'n_trainable_params_gen', 'n_trainable_params_disc'],
                    [input_dim, batch_size, n_features_first, critic_iters,
                     lambda_reg, learning_rate, fixed_noise_size, n_features_reduction_factor,
                     gen_fix_layer_1, gen_fix_layer_2, gen_fix_layer_3, gen_fix_layer_4,
                     disc_fix_layer_1, disc_fix_layer_2, disc_fix_layer_3, disc_fix_layer_4,
                     architecture, init_method, BN_layers_trainable,
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
    print model_config
    print
    t = time.time()  # get start time

    # for average times:
    if get_stats:
        t1s = np.zeros((iterations-iterations_trained))
        t2s = np.zeros((iterations-iterations_trained))
        t3s = np.zeros((iterations-iterations_trained))
        t4s = np.zeros((iterations-iterations_trained))

    for i in xrange(iterations-iterations_trained):
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
                print 'computation time to train for 1 iteration (minimize disc and gen one step): t1 = {}'.format(t1)
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
                print 't1/t2 = {}'.format(t1/t2)
                # list_ = session.run(g_optimizer.compute_gradients(g_loss, var_list=g_vars), feed_dict={z: z_train})
                # print 'number of gradients computed: {}'.format(2*len(list_))

        # print the current iteration
        print('iteration={}/{}'.format(i+iterations_trained+1, iterations))

        # all 100 steps compute the losses and elapsed times, and generate images
        if (i + iterations_trained) % 100 == 99:
            # get time for last 100 iterations
            elapsed_time = time.time() - t

            # generate sample images from fixed noise
            generate_image(i+iterations_trained+1)
            print 'generated images'

            # compute and save losses on dev set
            if architecture == 'DCGAN':
                dev_d_loss = []
                for images_dev, _ in dev_gen():
                    images_dev = images_dev.reshape([-1, 28, 28, 1])
                    z_train_dev = np.random.randn(batch_size, input_dim)
                    _dev_d_loss = session.run(d_loss, feed_dict={x_true: images_dev, z: z_train_dev})
                    dev_d_loss.append(_dev_d_loss)
                tp_app = pd.DataFrame(data=[[i + iterations_trained + 1, elapsed_time, np.mean(dev_d_loss)]],
                                      index=None, columns=['iteration', 'time', 'd_loss'])
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
                tp_app = pd.DataFrame(data=[[i+iterations_trained+1, elapsed_time, np.mean(dev_W_dist), np.mean(dev_d_loss)]],
                                      index=None, columns=['iteration', 'time', 'Wasserstein_dist', 'd_loss'])
                training_progress = pd.concat([training_progress, tp_app], axis=0, ignore_index=True)
                training_progress.to_csv(path_or_buf=directory+'training_progress.csv')
            print 'saved training progress'
            print

            # save model
            saver.save(sess=session, save_path=model_dir + 'saved_model')
            # save number of iterations trained
            np.savetxt(fname=model_dir + 'iterations.csv', X=[i + iterations_trained + 1])
            print 'saved model after training iteration {}'.format(i + iterations_trained + 1)

            # fix new start time
            t = time.time()

    # average times:
    if get_stats:
        print '\n\naverage times over {} iterations:'.format(iterations - iterations_trained)
        print 'computation time to train for 1 iteration (minimize disc and gen one step): t1 = {}'.format(np.mean(t1s))
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
        subject = 'GAN (MNIST) partly fixed training finished'
        body = 'to download the results of this model use (in the terminal):\n\n'
        body += 'scp -r fkrach@euler.ethz.ch:/cluster/home/fkrach/MasterThesis/MTCode1/'+directory+' .'
        files = [directory+'model_config.csv', directory+'training_progress.csv',
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
        parameters = [(INPUT_DIM, BATCH_SIZE, N_FEATURES_FIRST, CRITIC_ITERS, LAMBDA, 1e-4, ITERS, FIXED_NOISE_SIZE, 2,
                       False, False, False, False, False, False, False, False, 'DCGAN', 'He', True)]
    results = Parallel(n_jobs=nb_jobs)(delayed(train)(input_dim, batch_size, n_features_first, critic_iters,
                                                      lambda_reg, learning_rate, iterations, fixed_noise_size,
                                                      n_features_reduction_factor,
                                                      gen_fix_layer_1, gen_fix_layer_2,
                                                      gen_fix_layer_3, gen_fix_layer_4,
                                                      disc_fix_layer_1, disc_fix_layer_2,
                                                      disc_fix_layer_3, disc_fix_layer_4,
                                                      architecture, init_method, BN_layers_trainable)
                                       for input_dim, batch_size, n_features_first, critic_iters,
                                           lambda_reg, learning_rate, iterations, fixed_noise_size,
                                           n_features_reduction_factor,
                                           gen_fix_layer_1, gen_fix_layer_2,
                                           gen_fix_layer_3, gen_fix_layer_4,
                                           disc_fix_layer_1, disc_fix_layer_2,
                                           disc_fix_layer_3, disc_fix_layer_4,
                                           architecture, init_method, BN_layers_trainable in parameters)

    if settings.send_email:
        subject = 'partly fixed parallel training finished'
        body = 'to download all the results of this parallel run use (in the terminal):\n\n'
        for directory in results:
            body += 'scp -r fkrach@euler.ethz.ch:/cluster/home/fkrach/MasterThesis/MTCode1/'+directory+' .; '
        send_email.send_email(subject=subject, body=body, file_names=None)
    return 0


if __name__ == '__main__':
    # -------------------------------------------------------
    # parallel training
    if N_CPUS_PARALLEL is None:
        nb_jobs = settings.number_parallel_jobs
    else:
        nb_jobs = N_CPUS_PARALLEL

    param_array = settings.partly_fixed_param_array3

    parallel_training(parameters=param_array, nb_jobs=nb_jobs)

    # train(input_dim=128, batch_size=50, n_features_first=N_FEATURES_FIRST, critic_iters=1,
    #       lambda_reg=None, learning_rate=1e-4, iterations=20000, fixed_noise_size=FIXED_NOISE_SIZE,
    #       n_features_reduction_factor=2,
    #       gen_fix_layer_1=True, gen_fix_layer_2=True, gen_fix_layer_3=True, gen_fix_layer_4=False,
    #       disc_fix_layer_1=True, disc_fix_layer_2=True, disc_fix_layer_3=True, disc_fix_layer_4=False,
    #       architecture='DCGAN', init_method='LayerDistribution', BN_layers_trainable=True,
    #       load_saved=True)
