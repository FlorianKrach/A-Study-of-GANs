"""
author: Florian Krach
used parts of the code of the following implementations:
- Minimal implementation of Wasserstein GAN for MNIST, https://github.com/adler-j/minimal_wgan
- SWG, https://github.com/ishansd/swg
- Improved Wasserstein-GAN, https://github.com/igul222/improved_wgan_training
"""

"""
this function was the first implementation of JLSWGN on mnist, but is now outdated and was not updated any more
use JLSWGN_mnist.py or JLSWGN_32.py instead
"""


import numpy as np
import math
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
ITERS = 20000  # How many generator iterations to train for
FIXED_NOISE_SIZE = 128
JL_MEAN = 0
JL_STD = 1


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
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print'memory use: ' + str(memoryUse)


def generator(z, n_features_first=N_FEATURES_FIRST, n_features_reduction_factor=2, BN=True,
              init_method='He'):

    first_layers_trainable = True
    last_layer_trainable = True
    last2_layer_trainable = True

    if BN:
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


def train(input_dim=INPUT_DIM, batch_size=BATCH_SIZE, n_features_first=N_FEATURES_FIRST,
          learning_rate=1e-4, epochs=ITERS, fixed_noise_size=FIXED_NOISE_SIZE,
          n_features_reduction_factor=2,
          architecture='JLSWGN',
          init_method='He', BN=True, JL_dim=None, JL_error=0.5, n_projections=10000,
          load_saved=True):
    """
    - this is the function to use to train a Johnson-Lindenstrauss Generative Network model which uses the sliced
      Wasserstein-2 distance as objective funtion (JLSWGN) for MNIST, with the configuration given by the parameters
    - the function computes losses and auto-saves the model every 100 steps and automatically resumes training where it
      stopped (when load_saved=True)

    :param input_dim: the dimension of the latent space -> Z
    :param batch_size: the batch size, should be a divisor of 50k
    :param n_features_first: the number of feature maps in the first step of the generator
    :param epochs: the number of epochs to train for (in fact this number should be 50k/batch_size*true_epochs)
    :param fixed_noise_size: the number of pictures that is generated during training for visual progress
    :param n_features_reduction_factor: integer, e.g.: 1: use same number of feature-maps everywhere, 2: half the number
           of feature-maps in every step
    :param architecture: right now only supports 'JLSWGN', 'SWGN', defaults to 'JLSWGN'
    :param init_method: the method with which the variables are initialized, support: 'uniform', 'He', defaults to 'He'
    :param BN: shall batch normalization be used
    :param JL_dim: the target dimension of the JL mapping
    :param JL_error: the max pairwise distance deviation error of the JL mapping, only applies when JL_dim=None
    :param n_projections: number of random projections in sliced Wasserstein-2 distance
    :param load_saved: whether an already existing training progress shall be loaded to continue there (if one exists)
    :return:
    """

    picture_size = 28*28*1

    # -------------------------------------------------------
    # setting for sending emails and getting statistics
    send = settings.send_email

    # -------------------------------------------------------
    # architecture default
    use_JL = True
    if architecture not in ['SWGN']:
        architecture = 'JLSWGN'
    if architecture == 'SWGN':
        use_JL = False
        JL_error = None
        JL_dim = None

    # -------------------------------------------------------
    # init_method default
    if init_method not in ['uniform']:
        init_method = 'He'

    # -------------------------------------------------------
    # JL_dim:
    if JL_dim is None:
        if JL_error is None and use_JL:
            use_JL = False
            architecture = 'SWGN'
            print
            print 'architecture changed to SWGN, since JL_dim and JL_error were None'
            print
        elif JL_error is not None:
            JL_dim = int(math.ceil(8*np.log(2*batch_size)/(JL_error**2)))
            # this uses the constant given on the Wikipedia page of "Johnson-Lindenstrauss Lemma"
    else:
        JL_error = np.round(np.sqrt(8*np.log(2*batch_size)/JL_dim), decimals=4)

    if use_JL and JL_dim >= picture_size:
        use_JL = False
        architecture='SWGN'
        JL_error = None
        JL_dim = None
        print
        print 'JL mapping is not used, since the target dimension was chosen bigger than the input dimension'
        print

    # -------------------------------------------------------
    # create unique folder name
    directory = 'JLSWGN/'+str(input_dim)+'_'+str(batch_size)+'_'+str(n_features_first)+'_'+\
                str(learning_rate)+'_'+str(n_features_reduction_factor)+'_'+\
                str(architecture)+'_'+str(init_method)+'_'+str(BN)+'_'+str(JL_dim)+'_'+str(JL_error)+'_'+\
                str(n_projections)+'/'
    samples_dir = directory+'samples/'
    model_dir = directory+'model/'

    # create directories if they don't exist
    if not os.path.isdir('JLSWGN/'):
        call(['mkdir', 'JLSWGN/'])

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
    def build_model():
        """
        - function to build the model
        """
        with tf.name_scope('placeholders'):
            x_true = tf.placeholder(tf.float32, [None, 28, 28, 1])
            z = tf.placeholder(tf.float32, [None, input_dim])
            if use_JL:
                JL = tf.placeholder(tf.float32, [picture_size, JL_dim])
                P_non_normalized = tf.placeholder(tf.float32, [JL_dim, n_projections])
            else:
                JL = None
                P_non_normalized = tf.placeholder(tf.float32, [picture_size, n_projections])

        x_generated = generator(z, n_features_first=n_features_first,
                                n_features_reduction_factor=n_features_reduction_factor,
                                BN=BN,
                                init_method=init_method)

        # define loss
        with tf.name_scope('loss'):
            # first flatten the pictures
            x_true_flattened = tf.reshape(x_true, shape=[-1, picture_size])
            x_generated_flattened = tf.reshape(x_generated, shape=[-1, picture_size])

            # then apply the Johnson-Lindenstrauss map, if wanted
            if use_JL:
                JL_true = tf.matmul(x_true_flattened, JL)
                JL_gen = tf.matmul(x_generated_flattened, JL)
            else:
                JL_true = x_true_flattened
                JL_gen = x_generated_flattened

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

        with tf.name_scope('optimizer'):
            generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
            g_train = g_optimizer.minimize(generator_loss, var_list=generator_vars)

        # initialize variables using init_method
        session.run(tf.global_variables_initializer())

        return x_true, z, x_generated, JL, P_non_normalized, P, generator_loss, g_optimizer, g_train

    # -------------------------------------------------------
    # build the model
    x_true, z, x_generated, JL, P_non_normalized, P, generator_loss, g_optimizer, g_train = build_model()

    # -------------------------------------------------------
    # For saving samples, taken from IWGAN
    fixed_noise = np.random.normal(size=(fixed_noise_size, input_dim)).astype('float32')

    def generate_image(frame):
        samples = session.run(x_generated, feed_dict={z: fixed_noise}).squeeze()
        # print samples.shape
        save_images.save_images(
            samples.reshape((fixed_noise_size, 28, 28)),
            samples_dir + 'iteration_{}.png'.format(frame)
        )

    # -------------------------------------------------------
    # for saving the model create a saver
    saver = tf.train.Saver(max_to_keep=1)
    epochs_trained = 0
    tp_columns = ['epoch', 'time_for_epochs', 'JLSW2_loss', 'time_for_loss']
    training_progress = pd.DataFrame(data=None, index=None, columns=tp_columns)

    # restore the model:
    if load_saved:
        saver.restore(sess=session, save_path=model_dir + 'saved_model')
        epochs_trained = int(np.loadtxt(fname=model_dir + 'epochs.csv'))
        tp_app = pd.read_csv(filepath_or_buffer=directory + 'training_progress.csv', index_col=0, header=0)
        training_progress = pd.concat([training_progress, tp_app], axis=0, ignore_index=True)
        print 'loaded training progress, and the model, which was already trained for {} epochs'.format(epochs_trained)
        print training_progress
        print

    # if the network is already trained completely, set send to false
    if epochs_trained == epochs:
        send = False

    # -------------------------------------------------------
    # print and get model summary
    n_params_gen = model_summary(scope='generator')[0]
    print

    # -------------------------------------------------------
    # FK: print model config to file
    model_config = [['input_dim', 'batch_size', 'n_features_first',
                     'learning_rate', 'fixed_noise_size', 'n_features_reduction_factor',
                     'architecture', 'init_method',
                     'BN', 'JL_dim', 'JL_error', 'n_projections',
                     'n_trainable_params_gen'],
                    [input_dim, batch_size, n_features_first,
                     learning_rate, fixed_noise_size, n_features_reduction_factor,
                     architecture, init_method,
                     BN, JL_dim, JL_error, n_projections,
                     n_params_gen]]
    model_config = np.transpose(model_config)
    model_config = pd.DataFrame(data=model_config)
    model_config.to_csv(path_or_buf=directory + 'model_config.csv')
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

    for i in xrange(epochs - epochs_trained):
        # print the current epoch
        print('iteration={}/{}'.format(i + epochs_trained + 1, epochs))

        batch = gen.next()
        images = batch.reshape([-1, 28, 28, 1])
        z_train = np.random.randn(batch_size, input_dim)
        if use_JL:
            JL_train = np.random.randn(picture_size, JL_dim)
            P_train = np.random.randn(JL_dim, n_projections)
            session.run(g_train, feed_dict={x_true: images, z: z_train, JL: JL_train, P_non_normalized: P_train})
        else:
            P_train = np.random.randn(picture_size, n_projections)
            session.run(g_train, feed_dict={x_true: images, z: z_train, P_non_normalized: P_train})

        # memory()

        # all 100 steps compute the losses and elapsed times, and generate images
        if (i + epochs_trained) % 100 == 99:
            # get time for last 100 epochs
            elapsed_time = time.time() - t

            # generate sample images from fixed noise
            generate_image(i + epochs_trained + 1)
            print 'generated images'

            # save model
            saver.save(sess=session, save_path=model_dir + 'saved_model')
            # save number of epochs trained
            np.savetxt(fname=model_dir + 'epochs.csv', X=[i + epochs_trained + 1])
            print 'saved model after training epoch {}'.format(i + epochs_trained + 1)

            # compute and save losses on dev set
            t = time.time()
            dev_d_loss = []
            for images_dev, _ in dev_gen():
                images_dev = images_dev.reshape([-1, 28, 28, 1])
                z_train_dev = np.random.randn(batch_size, input_dim)
                if use_JL:
                    JL_train_dev = np.random.randn(picture_size, JL_dim)
                    P_train_dev = np.random.randn(JL_dim, n_projections)
                    _dev_d_loss = session.run(generator_loss, feed_dict={x_true: images_dev, z: z_train_dev,
                                                                         JL: JL_train_dev,
                                                                         P_non_normalized: P_train_dev})
                else:
                    P_train_dev = np.random.randn(picture_size, n_projections)
                    _dev_d_loss = session.run(generator_loss, feed_dict={x_true: images_dev, z: z_train_dev,
                                                                         P_non_normalized: P_train_dev})
                dev_d_loss.append(_dev_d_loss)
            t_loss = time.time() - t
            tp_app = pd.DataFrame(data=[[i + epochs_trained + 1, elapsed_time, np.mean(dev_d_loss), t_loss]],
                                  index=None, columns=tp_columns)
            training_progress = pd.concat([training_progress, tp_app], axis=0, ignore_index=True)
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
        subject = 'JLSWGN (MNIST) training finished'
        body = 'to download the results of this model use (in the terminal):\n\n'
        body += 'scp -r fkrach@euler.ethz.ch:/cluster/home/fkrach/MasterThesis/MTCode1/' + directory + ' .'
        files = [directory + 'model_config.csv', directory + 'training_progress.csv']
        send_email.send_email(subject=subject, body=body, file_names=files)

    return directory


def parallel_training(parameters=None, nb_jobs=-1):
    """
    :param parameters: an array of arrays with all the parameters
    :param nb_jobs: number of jobs that run parallel, -1 means all available cpus are used
    :return:
    """
    if parameters is None:
        parameters = [(INPUT_DIM, BATCH_SIZE, N_FEATURES_FIRST, 1e-4, ITERS, FIXED_NOISE_SIZE, 2,
                       'JLSWGN', 'He', True, 28*28/2, None, 10000),
                      (INPUT_DIM, BATCH_SIZE, N_FEATURES_FIRST, 1e-4, ITERS, FIXED_NOISE_SIZE, 2,
                       'SWGN', 'He', True, None, None, 10000)]
    results = Parallel(n_jobs=nb_jobs)(delayed(train)(input_dim, batch_size, n_features_first, learning_rate,
                                                      epochs, fixed_noise_size, n_features_reduction_factor,
                                                      architecture, init_method, BN, JL_dim, JL_error, n_projections)
                                       for input_dim, batch_size, n_features_first, learning_rate,
                                           epochs, fixed_noise_size, n_features_reduction_factor,
                                           architecture, init_method, BN, JL_dim, JL_error, n_projections in parameters)

    if settings.send_email:
        subject = 'JLSWGN (MNIST) parallel training finished'
        body = 'to download all the results of this parallel run use (in the terminal):\n\n'
        for directory in results:
            body += 'scp -r fkrach@euler.ethz.ch:/cluster/home/fkrach/MasterThesis/MTCode1/'+directory+' .; '
        send_email.send_email(subject=subject, body=body, file_names=None)
    return 0




if __name__ == '__main__':
    # -------------------------------------------------------
    # parallel training
    param_array = settings.JLSWGN_param_array2
    nb_jobs = settings.number_parallel_jobs

    parallel_training(parameters=param_array, nb_jobs=nb_jobs)
    # parallel_training(nb_jobs=nb_jobs)

    # train(n_features_reduction_factor=2, init_method='He', epochs=10000, load_saved=True, architecture='JLSWGN',
    #       JL_dim=28*28/2)





