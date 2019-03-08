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


def subset_train(input_dim=INPUT_DIM, batch_size=BATCH_SIZE, n_features_first=N_FEATURES_FIRST,
                 critic_iters=CRITIC_ITERS, lambda_reg=LAMBDA, learning_rate=1e-4, epochs1=ITERS, epochs2 = ITERS,
                 fixed_noise_size=FIXED_NOISE_SIZE, n_features_reduction_factor=2,
                 fix_last_layer_gen=False, fix_2last_layer_gen=True,
                 fix_last_layer_disc=False, fix_2last_layer_disc=True,
                 architecture='DCGAN', number_digits=5, which_digits=None,
                 second_train_on_others_only=False, perturb_factor=0,
                 load_saved=True):
    """
    - this function first trains a GAN on the mnist digits in 'which_digits' or on 'number_digits' randomly chosen ones,
      for 'epochs1', then fixes all but wanted last layers and trains on full dataset for 'epochs2'
    - the function computes losses and auto-saves the model every 100 steps and automatically resumes training where it
      stopped (when load_saved=True)

    :param input_dim:
    :param batch_size:
    :param n_features_first:
    :param critic_iters:
    :param lambda_reg:
    :param learning_rate:
    :param epochs1:
    :param epochs2:
    :param fixed_noise_size:
    :param n_features_reduction_factor: integer, e.g.: 1: use same number of feature-maps everywhere, 2: half the number
           of feature-maps in every step
    :param fix_last_layer_gen:
    :param fix_2last_layer_gen:
    :param fix_last_layer_disc:
    :param fix_2last_layer_disc:
    :param architecture: right now only supports 'WGANGP' and 'DCGAN', defaults to 'WGANGP'
    :param number_digits:
    :param which_digits:
    :param second_train_on_others_only:
    :param perturb_factor:
    :param load_saved:
    :return:
    """

    # -------------------------------------------------------
    # setting for sending emails
    send = settings.send_email

    # -------------------------------------------------------
    # architecture default
    if architecture not in ['DCGAN']:
        architecture = 'WGANGP'
    if architecture == 'DCGAN':
        critic_iters = 1

    # -------------------------------------------------------
    # mnist digits to use in first run
    if which_digits is None:
        all_digits = range(10)
        np.random.shuffle(all_digits)
        assert(number_digits <= 10)
        which_digits = all_digits[0:number_digits]
    which_digits_str = ''
    for digit in np.sort(which_digits):
        which_digits_str += str(digit)

    # -------------------------------------------------------
    # create unique folder name
    directory = 'subset_training/'+str(input_dim)+'_'+str(batch_size)+'_'+str(n_features_first)+'_'+str(critic_iters)+\
                '_'+str(lambda_reg)+\
                '_'+ str(learning_rate)+'_'+ str(fixed_noise_size)+'_'+str(n_features_reduction_factor)+'_'+\
                str(fix_last_layer_gen)+'_'+str(fix_2last_layer_gen)+'_'+\
                str(fix_last_layer_disc)+'_'+str(fix_2last_layer_disc)+'_'+\
                str(architecture)+'_'+str(second_train_on_others_only)+'_'+str(perturb_factor)+'_'+str(which_digits_str)+'/'
    samples_dir1 = directory+'samples1/'
    samples_dir2 = directory + 'samples2/'
    model_dir1 = directory+'model1/'
    model_dir2 = directory + 'model2/'

    # create directories if they don't exist
    if not os.path.isdir('subset_training/'):
        call(['mkdir', 'subset_training/'])

    if not os.path.isdir(directory):
        load_saved=False
        print 'make new directory:', directory
        print
        call(['mkdir', directory])
        call(['mkdir', samples_dir1])
        call(['mkdir', samples_dir2])
        call(['mkdir', model_dir1])
        call(['mkdir', model_dir2])

    # if directories already exist, but model wasn't saved so far, set load_saved to False
    if load_saved:
        load_saved2 = True
    else:
        load_saved2 = False
    if 'training_progress1.csv' not in os.listdir(directory):
        load_saved = False
        load_saved2 = False
    if 'training_progress2.csv' not in os.listdir(directory):
        load_saved2 = False

    # -------------------------------------------------------
    # initialize a TF session
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = settings.number_cpus
    config.inter_op_parallelism_threads = settings.number_cpus
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
                                                                                labels=tf.zeros_like(d_generated))) + \
                         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_true,
                                                                                labels=tf.ones_like(d_true)))
                d_loss = d_loss / 2.

            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=2 * learning_rate, beta1=0.5)

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
            return x_true, z, x_generated, d_true, d_generated, g_loss, d_loss, optimizer, \
                   g_vars, g_train, d_vars, d_train
        else:  # WGANGP
            return x_true, z, x_generated, d_true, d_generated, epsilon, x_hat, d_hat, gradients, ddx, d_regularizer, \
                   g_loss, wasserstein_dist, d_loss, optimizer, g_vars, g_train, d_vars, d_train

    # -------------------------------------------------------
    # FK: For saving samples, taken from IWGAN
    fixed_noise = np.random.normal(size=(fixed_noise_size, input_dim)).astype('float32')

    def generate_image(frame, samples_dir):
        samples = session.run(x_generated, feed_dict={z: fixed_noise}).squeeze()
        # print samples.shape
        save_images.save_images(
            samples.reshape((fixed_noise_size, 28, 28)),
            samples_dir + 'iteration_{}.png'.format(frame)
        )

    # -------------------------------------------------------
    # build the 1. model
    if architecture == 'DCGAN':
        x_true, z, x_generated, d_true, d_generated, g_loss, d_loss, optimizer, g_vars, \
        g_train, d_vars, d_train = build_model()
    else:  # WGANGP
        x_true, z, x_generated, d_true, d_generated, epsilon, x_hat, d_hat, gradients, ddx, d_regularizer, g_loss, \
        wasserstein_dist, d_loss, optimizer, g_vars, g_train, d_vars, d_train = build_model()

    # -------------------------------------------------------
    # FK: for saving the model create a saver
    saver = tf.train.Saver(max_to_keep=1)
    epochs_trained1 = 0
    if architecture == 'DCGAN':
        training_progress1 = pd.DataFrame(data=None, index=None, columns=['epoch', 'time', 'd_loss'])
    else:  # WGAN-GP
        training_progress1 = pd.DataFrame(data=None, index=None, columns=['epoch', 'time', 'Wasserstein_dist', 'd_loss'])

    # restore the 1. model:
    if load_saved:
        saver.restore(sess=session, save_path=model_dir1+'saved_model')
        epochs_trained1 = int(np.loadtxt(fname=model_dir1+'epochs.csv'))
        tp_app = pd.read_csv(filepath_or_buffer=directory+'training_progress1.csv', index_col=0, header=0)
        training_progress1 = pd.concat([training_progress1, tp_app], axis=0, ignore_index=True)
        print 'loaded training progress, and the model, which was already trained for {} epochs'.format(epochs_trained1)
        print training_progress1
        print

    # -------------------------------------------------------
    # FK: print and get model summary
    n_params_gen = model_summary(scope='generator')[0]
    print
    n_params_disc = model_summary(scope='discriminator')[0]
    print

    # -------------------------------------------------------
    # FK: print model config to file
    model_config = [['input_dim', 'batch_size', 'n_features_first', 'critic_iters',
                     'lambda_reg', 'learning_rate', 'fixed_noise_size', 'n_features_reduction_factor',
                     'fix_last_layer_gen', 'fix_2last_layer_gen',
                     'fix_last_layer_disc', 'fix_2last_layer_disc',
                     'architecture', 'second_train_on_others_only', 'perturb_factor', 'which_digits',
                     'n_trainable_params_gen', 'n_trainable_params_disc'],
                    [input_dim, batch_size, n_features_first, critic_iters,
                     lambda_reg, learning_rate, fixed_noise_size, n_features_reduction_factor,
                     fix_last_layer_gen, fix_2last_layer_gen,
                     fix_last_layer_disc, fix_2last_layer_disc,
                     architecture, second_train_on_others_only, perturb_factor, which_digits_str,
                     n_params_gen, n_params_disc]]
    model_config = np.transpose(model_config)
    model_config = pd.DataFrame(data=model_config)
    model_config.to_csv(path_or_buf=directory+'model_config.csv')
    print 'saved model configuration'
    print

    # -------------------------------------------------------
    # FK: get the MNIST data loader with digits in which_digits
    train_gen, dev_gen, test_gen = mnist.load(batch_size, batch_size, which_numbers=which_digits)

    # create a infinite generator
    def inf_train_gen():
        while True:
            for images, targets in train_gen():
                yield images

    gen = inf_train_gen()

    # -------------------------------------------------------
    # convenience training function
    def training_func(epochs, epochs_trained, model_dir, training_progress, training_progress_file_name, samples_dir):
        t = time.time()  # get start time

        for i in xrange(epochs-epochs_trained):
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
            print('epoch={}/{}'.format(i+epochs_trained+1, epochs))

            # all 100 steps compute the losses and elapsed times, and generate images
            if (i + epochs_trained) % 100 == 99:
                # get time for last 100 epochs
                elapsed_time = time.time() - t

                # generate sample images from fixed noise
                generate_image(i+epochs_trained+1, samples_dir)
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
                training_progress.to_csv(path_or_buf=directory + training_progress_file_name)
                print 'saved training progress'
                print

                # fix new start time
                t = time.time()

    # -------------------------------------------------------
    # train the 1. model
    training_func(epochs=epochs1, epochs_trained=epochs_trained1, model_dir=model_dir1,
                  training_progress=training_progress1, training_progress_file_name='training_progress1.csv',
                  samples_dir=samples_dir1)

    # -------------------------------------------------------
    # perturbe the weights if wanted
    # TODO: might it be better to only perturb the weights of the generator (instead of gen. and discriminator)?
    if perturb_factor > 0:
        trainable_vars = tf.trainable_variables()
        for v in trainable_vars:
            if 'BatchNorm' not in v.name:
                std = np.std(session.run(v))
                session.run(tf.assign(v, value=v + tf.random_normal(v.shape, mean=0.0,
                                                                    stddev=std * perturb_factor)))
        saver = tf.train.Saver(max_to_keep=2)
        saver.save(sess=session, save_path=model_dir1 + 'perturbed_model')
        print
        print 'perturbed weights were saved'
        print

    # -------------------------------------------------------
    # load new session, so that no conflict with names in the name_scopes
    session.close()
    tf.reset_default_graph()
    session = tf.Session(config=config)

    # -------------------------------------------------------
    # build the 2. model
    if architecture == 'DCGAN':
        x_true, z, x_generated, d_true, d_generated, g_loss, d_loss, optimizer, g_vars, \
        g_train, d_vars, d_train = build_model(fix_first_layers_gen_b=True, fix_last_layer_gen_b=fix_last_layer_gen,
                                               fix_2last_layer_gen_b=fix_2last_layer_gen,
                                               fix_first_layers_disc_b=True, fix_last_layer_disc_b=fix_last_layer_disc,
                                               fix_2last_layer_disc_b=fix_2last_layer_disc)
    else:  # WGANGP
        x_true, z, x_generated, d_true, d_generated, epsilon, x_hat, d_hat, gradients, ddx, d_regularizer, g_loss, \
        wasserstein_dist, d_loss, optimizer, g_vars,\
        g_train, d_vars, d_train = build_model(fix_first_layers_gen_b=True, fix_last_layer_gen_b=fix_last_layer_gen,
                                               fix_2last_layer_gen_b=fix_2last_layer_gen,
                                               fix_first_layers_disc_b=True, fix_last_layer_disc_b=fix_last_layer_disc,
                                               fix_2last_layer_disc_b=fix_2last_layer_disc)

    # -------------------------------------------------------
    # FK: for saving the 2. model create a saver
    saver = tf.train.Saver(max_to_keep=1)
    epochs_trained2 = 0
    if architecture == 'DCGAN':
        training_progress2 = pd.DataFrame(data=None, index=None, columns=['epoch', 'time', 'd_loss'])
    else:  # WGAN-GP
        training_progress2 = pd.DataFrame(data=None, index=None, columns=['epoch', 'time', 'Wasserstein_dist', 'd_loss'])

    # restore the 2. model:
    if load_saved2:
        saver.restore(sess=session, save_path=model_dir2+'saved_model')
        epochs_trained2 = int(np.loadtxt(fname=model_dir2+'epochs.csv'))
        tp_app = pd.read_csv(filepath_or_buffer=directory+'training_progress2.csv', index_col=0, header=0)
        training_progress2 = pd.concat([training_progress2, tp_app], axis=0, ignore_index=True)
        print 'loaded training progress, and the model, which was already trained for {} epochs'.format(epochs_trained2)
        print training_progress2
        print
    # or load the 1. model if training of 2. not yet startet
    else:
        if perturb_factor > 0:
            saver.restore(sess=session, save_path=model_dir1 + 'perturbed_model')
            print 'loaded the 1. model, now with only last layers trainable, with perturbed weights'
            print
        else:
            saver.restore(sess=session, save_path=model_dir1+'saved_model')
            print 'loaded the 1. model, now with only last layers trainable'
            print

    # -------------------------------------------------------
    # FK: print and get model summary of 2. model
    n_params_gen2 = model_summary(scope='generator')[0]
    print
    n_params_disc2 = model_summary(scope='discriminator')[0]
    print

    # -------------------------------------------------------
    # print model config to file with 2. model
    model_config = [['input_dim', 'batch_size', 'n_features_first', 'critic_iters',
                     'lambda_reg', 'learning_rate', 'fixed_noise_size', 'n_features_reduction_factor',
                     'fix_last_layer_gen', 'fix_2last_layer_gen',
                     'fix_last_layer_disc', 'fix_2last_layer_disc',
                     'architecture', 'second_train_on_others_only', 'perturb_factor', 'which_digits',
                     'n_trainable_params_gen', 'n_trainable_params_disc',
                     'n_trainable_params_gen2', 'n_trainable_params_disc2'],
                    [input_dim, batch_size, n_features_first, critic_iters,
                     lambda_reg, learning_rate, fixed_noise_size, n_features_reduction_factor,
                     fix_last_layer_gen, fix_2last_layer_gen,
                     fix_last_layer_disc, fix_2last_layer_disc,
                     architecture, second_train_on_others_only, perturb_factor, which_digits_str,
                     n_params_gen, n_params_disc,
                     n_params_gen2, n_params_disc2]]
    model_config = np.transpose(model_config)
    model_config = pd.DataFrame(data=model_config)
    model_config.to_csv(path_or_buf=directory + 'model_config.csv')
    print 'saved model configuration'
    print

    # -------------------------------------------------------
    # get the MNIST data loader with all digits
    if second_train_on_others_only:
        all_digits = range(10)
        digits = [digit for digit in all_digits if digit not in which_digits]
    else:
        digits = None
    train_gen, dev_gen, test_gen = mnist.load(batch_size, batch_size, which_numbers=digits)

    # create a infinite generator
    def inf_train_gen():
        while True:
            for images, targets in train_gen():
                yield images

    gen = inf_train_gen()

    # -------------------------------------------------------
    # train the 2. model
    training_func(epochs=epochs2, epochs_trained=epochs_trained2, model_dir=model_dir2,
                  training_progress=training_progress2, training_progress_file_name='training_progress2.csv',
                  samples_dir=samples_dir2)

    # -------------------------------------------------------
    # after training close the session
    session.close()
    tf.reset_default_graph()

    # -------------------------------------------------------
    # when training is done send email
    if send:
        subject = 'GAN (MNIST) training on mnist digits subset finished'
        body = 'to download the results of this model use (in the terminal):\n\n'
        body += 'scp -r fkrach@euler.ethz.ch:/cluster/home/fkrach/MasterThesis/MTCode1/'+directory+' .'
        files = [directory+'model_config.csv', directory+'training_progress1.csv', directory+'training_progress2.csv']
        send_email.send_email(subject=subject, body=body, file_names=files)

    return directory


def parallel_training(parameters=None, nb_jobs=-1):
    """
    :param parameters: an array of arrays with all the parameters
    :param nb_jobs: number of jobs that run parallel, -1 means all available cpus are used
    :return:
    """
    if parameters is None:
        parameters = [(INPUT_DIM, BATCH_SIZE, N_FEATURES_FIRST, CRITIC_ITERS, LAMBDA, 1e-4, ITERS, ITERS,
                       FIXED_NOISE_SIZE, 2, False, True, False, True, 'WGANGP', 5, None, False, 0),
                      (INPUT_DIM, BATCH_SIZE, N_FEATURES_FIRST, CRITIC_ITERS, LAMBDA, 1e-4, ITERS, ITERS,
                       FIXED_NOISE_SIZE, 2, False, True, False, True, 'DCGAN', 5, None, False, 0)]
    results = Parallel(n_jobs=nb_jobs)(delayed(subset_train)(input_dim, batch_size, n_features_first, critic_iters,
                                                             lambda_reg, learning_rate, epochs1, epochs2,
                                                             fixed_noise_size, n_features_reduction_factor,
                                                             fix_last_layer_gen, fix_2last_layer_gen,
                                                             fix_last_layer_disc, fix_2last_layer_disc,
                                                             architecture, number_digits, which_digits,
                                                             second_train_on_others_only, perturb_factor)
                                       for input_dim, batch_size, n_features_first, critic_iters,
                                           lambda_reg, learning_rate, epochs1, epochs2,
                                           fixed_noise_size, n_features_reduction_factor,
                                           fix_last_layer_gen, fix_2last_layer_gen,
                                           fix_last_layer_disc, fix_2last_layer_disc,
                                           architecture, number_digits, which_digits,
                                           second_train_on_others_only, perturb_factor in parameters)
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
    param_array = settings.subset_param_array1
    nb_jobs = settings.number_parallel_jobs

    parallel_training(nb_jobs=nb_jobs, parameters=param_array)

    # subset_train(epochs1=100, epochs2=100, architecture='DCGAN', perturb_factor=1, second_train_on_others_only=True)