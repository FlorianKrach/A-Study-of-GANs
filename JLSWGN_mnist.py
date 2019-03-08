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
import preprocessing_mnist


# FK: variable definitions
BATCH_SIZE = 250
INPUT_DIM = 128
PICTURE_SIZE_POWER = 5  # the picture width and height are 2^PICTURE_SIZE_POWER
N_FEATURES_FIRST = 256  # FK: has to be divisible by 4
ITERS = 20000  # How many generator iterations to train for
FIXED_NOISE_SIZE = 128
JL_MEAN = 0
JL_STD = 1
NPY = False  # whether to use the .npy version of the data. This is much faster, but needs more memory (about 3 times)

STEP_SIZE_LOSS_COMPUTATION = 100  # after which number of steps losses are computed and samples generated
START_COMPUTING_LOSS = 0

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
              BN=True, power=5, init_method='He'):

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
        # first layer (fully connected) -> [B, 4, 4, n_features]
        z = layers.fully_connected(z, num_outputs=4*4*n_features_first, trainable=True,
                                   normalizer_fn=normalizer, weights_initializer=init)
        z = tf.reshape(z, [-1, 4, 4, n_features_first])  # we use the dimensions as NHWC resp. BHWC

        # middle layers (convolutions) -> [B, 4*(2**(power-3)), 4*(2**(power-3)), n_features2]
        for i in range(power-3):
            n_out = max(int(n_features_first/(n_features_reduction_factor**(i+1))), min_features)
            z = layers.conv2d_transpose(z, num_outputs=n_out, kernel_size=5,
                                        stride=2, trainable=True, normalizer_fn=normalizer,
                                        weights_initializer=init)

        # last layer (convolution) -> [B, (2**power), (2**power), 1] -> [B, (2**power)**2]
        z = layers.conv2d_transpose(z, num_outputs=1, kernel_size=5, stride=2, activation_fn=tf.nn.sigmoid,
                                    trainable=True, weights_initializer=init)
        size = 2**power
        return tf.reshape(z, shape=[-1, size*size])


def train(input_dim=INPUT_DIM, batch_size=BATCH_SIZE,
          learning_rate=1e-4, epochs=ITERS, fixed_noise_size=FIXED_NOISE_SIZE,
          n_features_first=N_FEATURES_FIRST, n_features_reduction_factor=2, min_features=64,
          architecture='JLSWGN', init_method='He', BN=False,
          JL_dim=None, JL_error=None, n_projections=10000,
          power=5, image_enlarge_method='zoom', order=1,
          load_saved=True):
    """
    - this is the function to use to train a Johnson-Lindenstrauss Generative Network model which uses the sliced
      Wasserstein-2 distance as objective funtion (JLSWGN) for MNIST (that is artificially enlarged, to get a higher
      dimensional data set), with the configuration given by the parameters
    - the function computes losses and auto-saves the model every 100 steps and automatically resumes training where it
      stopped (when load_saved=True)

    :param input_dim: the dimension of the latent space -> Z
    :param batch_size: the batch size, should be a divisor of 50k
    :param learning_rate:
    :param n_features_first: the number of feature maps in the first step of the generator
    :param epochs: the number of iterations to train for (this should be: 50k/batch_size*true_epochs)
    :param fixed_noise_size: the number of pictures that is generated during training for visual progress
    :param n_features_reduction_factor: integer, e.g.: 1: use same number of feature-maps everywhere, 2: half the number
           of feature-maps in every step
    :param min_features: the minimal number of features (if the reduction of features would give something smaller, it
           is set to this number)
    :param architecture: right now only supports 'JLSWGN', 'SWGN', defaults to 'JLSWGN'
    :param init_method: the method with which the variables are initialized, support: 'uniform', 'He', defaults to 'He'
    :param BN: shall batch normalization be used
    :param JL_dim: the target dimension of the JL mapping
    :param JL_error: the max pairwise distance deviation error of the JL mapping, only applies when JL_dim=None
    :param n_projections: number of random projections in sliced Wasserstein-2 distance
    :param power: int, this specifies the picture size as 2**power, training data in this size is produced automatically
           using preprocessing_mnist.py (if not existent), default: 'zoom'
    :param image_enlarge_method: whether to 'zoom' the MNIST data or to only 'enlarge' the black border
    :param order: only needed when 'zoom', the degree of the spline interpolation used for zooming, int in [0, 5],
           default: 1
    :param load_saved: whether an already existing training progress shall be loaded to continue there (if one exists)
    :return:
    """

    # -------------------------------------------------------
    # setting for sending emails and getting statistics
    send = settings.send_email

    # -------------------------------------------------------
    # picture size
    assert type(power) == int
    size = 2**power
    picture_size = size*size
    picture_dim = [-1, size, size]

    # -------------------------------------------------------
    # image enlarge method default
    if image_enlarge_method in ['enlarge', 'enlarge_border', 'border']:
        image_enlarge_method = 'border'
        order = None
    else:
        image_enlarge_method = 'zoom'
        if (order is None) or (int(order) not in range(6)):
            order = 1
        else:
            order = int(order)

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
            print '\narchitecture changed to SWGN, since JL_dim and JL_error were None\n'
        elif JL_error is not None:
            JL_dim = int(math.ceil(8 * np.log(2 * batch_size) / (JL_error ** 2)))
            # this uses the constant given on the Wikipedia page of "Johnson-Lindenstrauss Lemma"
    else:
        JL_error = np.round(np.sqrt(8 * np.log(2 * batch_size) / JL_dim), decimals=4)

    if use_JL and JL_dim >= picture_size:
        use_JL = False
        architecture = 'SWGN'
        JL_error = None
        JL_dim = None
        print '\nJL mapping is not used, since the target dimension was chosen bigger than the input dimension\n'

    print '\nJL_dim = {}'.format(JL_dim)
    print 'JL_error = {}\n'.format(JL_error)

    # -------------------------------------------------------
    # create unique folder name
    dir1 = 'JLSWGN_mnist/'
    directory = dir1 + str(size) + '_' + str(image_enlarge_method) + '_' + str(order) + '_' + \
                str(input_dim) + '_' + str(batch_size) + '_' + str(learning_rate) + '_' + \
                str(n_features_first) + '_' + str(n_features_reduction_factor) + '_' + str(min_features) + '_' + \
                str(architecture) + '_' + str(init_method) + '_' + str(BN) + '_' + \
                str(JL_dim) + '_' + str(JL_error) + '_' + \
                str(n_projections) + '/'
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
        with tf.name_scope('placeholders'):
            real_data_int = tf.placeholder(tf.uint8, [None, picture_size])  # uint8 with int values in [0, 255]
            x_true = tf.cast(real_data_int, tf.float32) / 255.  # float with values in [0,1]
            z = tf.placeholder(tf.float32, [None, input_dim])
            if use_JL:
                JL = tf.placeholder(tf.float32, [picture_size, JL_dim])
                P_non_normalized = tf.placeholder(tf.float32, [JL_dim, n_projections])
                P_non_normalized_SWD = tf.placeholder(tf.float32, [picture_size, n_projections])
            else:
                JL = None
                P_non_normalized = tf.placeholder(tf.float32, [picture_size, n_projections])
                P_non_normalized_SWD = None

        x_generated = generator(z, n_features_first=n_features_first,
                                n_features_reduction_factor=n_features_reduction_factor, min_features=min_features,
                                BN=BN, power=power,
                                init_method=init_method)

        # define loss (big part taken from SWG)
        with tf.name_scope('loss'):
            # apply the Johnson-Lindenstrauss map, if wanted, to the flattened arrays
            if use_JL:
                JL_true = tf.matmul(x_true, JL)
                JL_gen = tf.matmul(x_generated, JL)
            else:
                JL_true = x_true
                JL_gen = x_generated

            # next project the samples (images). After being transposed, we have tensors
            # of the format: [[projected_image1_proj1, projected_image2_proj1, ...],
            #                 [projected_image1_proj2, projected_image2_proj2, ...],
            #                 ...]
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

            # get for JLSWGN the sliced Wasserstein distance (SWD) (since SWD and JLSWD are not comparable)
            if use_JL:
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
            else:
                SWD = generator_loss

        with tf.name_scope('optimizer'):
            generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
            g_train = g_optimizer.minimize(generator_loss, var_list=generator_vars)

        # initialize variables using init_method
        session.run(tf.global_variables_initializer())

        return real_data_int, z, x_generated, JL, P_non_normalized, P_non_normalized_SWD, SWD, g_train

    # -------------------------------------------------------
    # build the model
    real_data_int, z, x_generated, JL, P_non_normalized, P_non_normalized_SWD, SWD, g_train = build_model()

    # -------------------------------------------------------
    # For creating and saving samples (taken from IWGAN)
    fixed_noise = np.random.normal(size=(fixed_noise_size, input_dim)).astype('float32')

    def generate_image(frame):
        samples = session.run(x_generated, feed_dict={z: fixed_noise})
        samples = (samples * 255.99).astype('uint8')  # transform linearly from [0,1] to int[0,255]
        samples = samples.reshape(picture_dim)
        save_images.save_images(samples, samples_dir + 'iteration_{}.png'.format(frame))

    # -------------------------------------------------------
    # get the dataset as infinite generator
    mem = memory()

    data_dir = '../data/MNIST/'
    if image_enlarge_method == 'zoom':
        if NPY:
            data_file = data_dir + 'mnist{}_zoom_{}_train.npy'.format(size, order)
        else:
            data_file = data_dir + 'mnist{}_zoom_{}.pkl.gz'.format(size, order)
        if not os.path.isfile(data_file):
            print '\ndata set not found, creating now ...'
            preprocessing_mnist.zoom(size=size, interpolation_order=order, npy=NPY)
        data_file = data_dir + 'mnist{}_zoom_{}'.format(size, order)
    else:
        if NPY:
            data_file = data_dir + 'mnist{}_border_train.npy'.format(size)
        else:
            data_file = data_dir + 'mnist{}_border.pkl.gz'.format(size)
        if not os.path.isfile(data_file):
            print '\ndata set not found, creating now ...'
            preprocessing_mnist.enlarge_border(size=size, npy=NPY)
        data_file = data_dir + 'mnist{}_border'.format(size)

    print 'load data ...'
    train_gen, n_train_samples, dev_gen, n_dev_samples = preprocessing_mnist.load(data_file, batch_size, npy=NPY)
    print 'number train samples: {}'.format(n_train_samples)
    print 'number dev samples: {}\n'.format(n_dev_samples)

    def inf_train_gen():
        while True:
            for images, _ in train_gen():
                yield images

    gen = inf_train_gen()

    print 'memory usage before loading data (GB): {}'.format(mem)
    mem = memory()
    print 'memory usage after loading data (GB): {}\n'.format(mem)

    # -------------------------------------------------------
    # for saving the model create a saver
    saver = tf.train.Saver(max_to_keep=1)
    epochs_trained = 0
    tp_columns = ['iteration', 'time_for_iterations', 'SWD_approximation', 'time_for_SWD', 'used_memory_GB']
    training_progress = pd.DataFrame(data=None, index=None, columns=tp_columns)

    # restore the model:
    if load_saved:
        saver.restore(sess=session, save_path=model_dir + 'saved_model')
        epochs_trained = int(np.loadtxt(fname=model_dir + 'epochs.csv'))
        tp_app = pd.read_csv(filepath_or_buffer=directory + 'training_progress.csv', index_col=0, header=0)
        training_progress = pd.concat([training_progress, tp_app], axis=0, ignore_index=True)
        print 'loaded training progress, and the model, which was already trained for {} iterations'.format(epochs_trained)
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
    model_config = [['data_set', 'input_dim', 'batch_size', 'learning_rate', 'fixed_noise_size',
                     'n_features_first', 'n_features_reduction_factor', 'min_features',
                     'architecture', 'init_method',
                     'BN', 'JL_dim', 'JL_error', 'n_projections',
                     'n_trainable_params_gen'],
                    [data_file[:-7], input_dim, batch_size, learning_rate, fixed_noise_size,
                     n_features_first, n_features_reduction_factor, min_features,
                     architecture, init_method,
                     BN, JL_dim, JL_error, n_projections,
                     n_params_gen]]
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

    for i in xrange(epochs - epochs_trained):
        # print the current iteration
        print('iteration={}/{}'.format(i + epochs_trained + 1, epochs))

        images = gen.next()
        z_train = np.random.randn(batch_size, input_dim)
        if use_JL:
            JL_train = np.random.randn(picture_size, JL_dim)
            P_train = np.random.randn(JL_dim, n_projections)
            session.run(g_train, feed_dict={real_data_int: images, z: z_train, JL: JL_train, P_non_normalized: P_train})
        else:
            P_train = np.random.randn(picture_size, n_projections)
            session.run(g_train, feed_dict={real_data_int: images, z: z_train, P_non_normalized: P_train})

        mem = memory()
        if not settings.euler:
            print'memory use (GB): {}'.format(mem)

        # all STEP_SIZE_LOSS_COMPUTATION steps compute the losses and elapsed times, and generate images, and save model
        if (i + epochs_trained) % STEP_SIZE_LOSS_COMPUTATION == (STEP_SIZE_LOSS_COMPUTATION-1):
            # get time for last STEP_SIZE_LOSS_COMPUTATION epochs
            elapsed_time = time.time() - t

            # generate sample images from fixed noise
            generate_image(i + epochs_trained + 1)
            print 'generated images'

            # compute and save losses on dev set, starting after ? iterations
            if i + epochs_trained + 1 >= START_COMPUTING_LOSS:
                t = time.time()
                dev_SWD = []
                print 'compute SWD ...'
                j = 0
                for images_dev, _ in dev_gen():
                    if not settings.euler:
                        # progress bar
                        sys.stdout.write('\r>> Compute SWD %.1f%%' % (float(j)/float(n_dev_samples/batch_size) * 100.0))
                        sys.stdout.flush()
                        j += 1
                    z_train_dev = np.random.randn(batch_size, input_dim)
                    P_train_dev = np.random.randn(picture_size, n_projections)
                    if use_JL:
                        _dev_SWD = session.run(SWD, feed_dict={real_data_int: images_dev, z: z_train_dev,
                                                               P_non_normalized_SWD: P_train_dev})
                    else:
                        _dev_SWD = session.run(SWD, feed_dict={real_data_int: images_dev, z: z_train_dev,
                                                               P_non_normalized: P_train_dev})
                    dev_SWD.append(_dev_SWD)
                dev_SWD = np.mean(dev_SWD)
                t_loss = time.time() - t
            else:
                dev_SWD = None
                t_loss = None

            tp_app = pd.DataFrame(data=[[i + epochs_trained + 1, elapsed_time, dev_SWD, t_loss, mem]],
                                  index=None, columns=tp_columns)
            training_progress = pd.concat([training_progress, tp_app], axis=0, ignore_index=True)

            # save model
            saver.save(sess=session, save_path=model_dir + 'saved_model')
            # save number of epochs trained
            np.savetxt(fname=model_dir + 'epochs.csv', X=[i + epochs_trained + 1])
            print 'saved model after training epoch {}'.format(i + epochs_trained + 1)
            # save training progress
            training_progress.to_csv(path_or_buf=directory + 'training_progress.csv')

            print 'saved training progress\n'

            # fix new start time
            t = time.time()

    # -------------------------------------------------------
    # after training close the session
    session.close()
    tf.reset_default_graph()

    # -------------------------------------------------------
    # when training is done send email
    if send:
        subject = 'JLSWGN ({}) training finished'.format(data_file[:-7])
        body = 'to download the results of this model use (in the terminal):\n\n'
        body += 'scp -r fkrach@euler.ethz.ch:/cluster/home/fkrach/MasterThesis/MTCode1/' + directory + ' .'
        files = [directory + 'model_config.csv', directory + 'training_progress.csv',
                 samples_dir + 'iteration_{}.png'.format(epochs)]
        send_email.send_email(subject=subject, body=body, file_names=files)

    return directory


def parallel_training(parameters=None, nb_jobs=-1):
    """
    :param parameters: an array of arrays with all the parameters
    :param nb_jobs: number of jobs that run parallel, -1 means all available cpus are used
    :return:
    """
    if parameters is None:
        parameters = [(INPUT_DIM, BATCH_SIZE, 1e-4, ITERS, FIXED_NOISE_SIZE, N_FEATURES_FIRST, 2, 64,
                       'JLSWGN', 'He', True, 32*32*3/4, None, 10000, 5, 'zoom', 1),
                      (INPUT_DIM, BATCH_SIZE, 1e-4, ITERS, FIXED_NOISE_SIZE, N_FEATURES_FIRST, 2, 64,
                       'JLSWGN', 'He', True, None, None, 10000, 5, 'zoom', 1)]
    results = Parallel(n_jobs=nb_jobs)(delayed(train)(input_dim, batch_size, learning_rate, epochs,
                                                      fixed_noise_size, n_features_first, n_features_reduction_factor,
                                                      min_features, architecture, init_method, BN, JL_dim, JL_error,
                                                      n_projections, power, image_enlarge_method, order)
                                       for input_dim, batch_size, learning_rate, epochs,
                                           fixed_noise_size, n_features_first, n_features_reduction_factor,
                                           min_features, architecture, init_method, BN, JL_dim, JL_error,
                                           n_projections, power, image_enlarge_method, order in parameters)

    if settings.send_email:
        subject = 'JLSWGN MNIST parallel training finished'
        body = 'to download all the results of this parallel run use (in the terminal):\n\n'
        for directory in results:
            body += 'scp -r fkrach@euler.ethz.ch:/cluster/home/fkrach/MasterThesis/MTCode1/'+directory+' .; '
        send_email.send_email(subject=subject, body=body, file_names=None)
    return 0


if __name__ == '__main__':
    # -------------------------------------------------------
    # parallel training
    param_array = settings.JLSWGN_MNIST_param_array1_0
    nb_jobs = settings.number_parallel_jobs


    # parallel_training(parameters=param_array, nb_jobs=nb_jobs)
    # parallel_training(nb_jobs=nb_jobs)

    train(n_features_first=256, init_method='He', epochs=10000, load_saved=True, architecture='JLSWGN',
          JL_dim=None, learning_rate=1e-4, batch_size=100, n_projections=10000,
          power=5, order=1, image_enlarge_method='zoom')


