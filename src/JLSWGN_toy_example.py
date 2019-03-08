"""
author: Florian Krach
"""

import numpy as np
import math
import tensorflow as tf
from tensorflow.contrib import layers
import settings
import send_email
import os, sys
from subprocess import call
import tensorflow.contrib.slim as slim
import pandas as pd
import time
from joblib import Parallel, delayed


# FK: variable definitions
BATCH_SIZE = 50
INPUT_DIM = 128
N_FEATURES_FIRST = 256  # FK: has to be divisible by 4
CRITIC_ITERS = 1  # number of critic iters per gen iter
ITERS = 20000  # How many generator iterations to train for
FIXED_NOISE_SIZE = 1024
JL_MEAN = 0
JL_STD = 1
TEST_DATA_SIZE = 1000

GENERATE_SAMPLES = False

if settings.euler:
    N_CPUS_TF = 1  # set to None to use value of settings.py
    N_CPUS_PARALLEL = 6  # set to None to use value of settings.py
else:
    N_CPUS_TF = 2  # set to None to use value of settings.py
    N_CPUS_PARALLEL = 2  # set to None to use value of settings.py



# FK: model summary of
def model_summary(scope):
    model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    return slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def generator(z, out_dim, n_features_first=N_FEATURES_FIRST, BN=True, init_method='normal'):

    if BN:
        normalizer = layers.batch_norm
    else:  # WGAN-GP
        normalizer = None

    if init_method in ['He']:
        init = layers.variance_scaling_initializer()
    elif init_method == 'normal':
        init = layers.xavier_initializer(uniform=False)
    else:
        init = layers.xavier_initializer(uniform=True)

    # the layers use relu activations (default)
    with tf.variable_scope('generator'):
        z = layers.fully_connected(z, num_outputs=n_features_first, normalizer_fn=normalizer, weights_initializer=init)
        # we use the dimensions as BxDim
        z = layers.fully_connected(z, num_outputs=n_features_first, normalizer_fn=normalizer, weights_initializer=init)
        z = layers.fully_connected(z, num_outputs=out_dim, normalizer_fn=normalizer, weights_initializer=init,
                                   activation_fn=tf.nn.sigmoid)
        return z


def generate_target(filename, number_samples=10000, N=1000, sigma=0.1):
    """
    - this function randomly chooses a mean and produces samples of a normal distribution with this mean and variance
      matrix I*sigma**2, where I is the identity matrix in N dimensions, then clips the samples to [0,1]^N
    :param filename: the prefix of the file, dimension and sample number are added to make it unique
    :param number_samples: the number of samples to generate
    :param N: the dimension of the target space
    :param sigma: the standard deviation of the distribution
    :return:
    """

    # first chose a mean
    mean = np.random.uniform(low=0.2, high=0.8, size=N).reshape((N, 1))
    # print mean.reshape((-1))

    # generate samples
    samples = np.random.normal(loc=0, scale=sigma, size=N*number_samples).reshape((N, number_samples))
    samples = mean + samples
    samples = np.clip(samples, a_min=0, a_max=1)
    samples = np.transpose(samples)

    # store the samples
    np.save(file=filename, arr=samples)
    print "stored {} samples of a normal distribution in {} dimensions".format(number_samples, N)
    print

    return samples


def compute_intra_set_SWD(path='', batch_size=50, L=10000, N=1000):
    data = np.load(path)
    l, dim = data.shape

    # SWD
    x_true = tf.placeholder(tf.float32, [None, dim])
    x_generated = tf.placeholder(tf.float32, [None, dim])
    P_non_normalized_SWD = tf.placeholder(tf.float32, [dim, L])

    P_SWD = tf.nn.l2_normalize(P_non_normalized_SWD, axis=0)
    projected_true_SWD = tf.transpose(tf.matmul(x_true, P_SWD))
    projected_fake_SWD = tf.transpose(tf.matmul(x_generated, P_SWD))
    sorted_true_SWD, true_indices_SWD = tf.nn.top_k(input=projected_true_SWD, k=batch_size)
    sorted_fake_SWD, fake_indices_SWD = tf.nn.top_k(input=projected_fake_SWD, k=batch_size)
    flat_true_SWD = tf.reshape(sorted_true_SWD, [-1])
    rows = np.asarray(
        [batch_size * np.floor(i * 1.0 / batch_size) for i in range(L * batch_size)])
    rows = rows.astype(np.int32)
    flat_idx_SWD = tf.reshape(fake_indices_SWD, [-1, 1]) + np.reshape(rows, [-1, 1])
    shape = tf.constant([batch_size * L])
    rearranged_true_SWD = tf.reshape(tf.scatter_nd(flat_idx_SWD, flat_true_SWD, shape),
                                     [L, batch_size])
    SWD = tf.reduce_mean(tf.square(projected_fake_SWD - rearranged_true_SWD))

    # tf Session
    config = tf.ConfigProto()
    if N_CPUS_TF is None:
        number_cpus_tf = settings.number_cpus
    else:
        number_cpus_tf = N_CPUS_TF
    config.intra_op_parallelism_threads = number_cpus_tf
    config.inter_op_parallelism_threads = number_cpus_tf
    session = tf.Session(config=config)

    # compute distance
    inter_SWD = []
    for i in range(N):
        # progress bar
        sys.stdout.write('\r>> Compute inter set SWD %.1f%%' % (float(i) / float(N) * 100.0))
        sys.stdout.flush()

        ind1 = np.random.randint(0, l, batch_size)
        ind2 = np.random.randint(0, l, batch_size)
        P = np.random.randn(dim, L)
        _SWD = session.run(SWD, feed_dict={x_true: data[ind1], x_generated: data[ind2],
                                           P_non_normalized_SWD: P})
        inter_SWD.append(_SWD)

    return np.mean(inter_SWD)


def train(input_dim=INPUT_DIM, batch_size=BATCH_SIZE, n_features_first=N_FEATURES_FIRST,
          learning_rate=1e-4, epochs=ITERS, fixed_noise_size=FIXED_NOISE_SIZE,
          architecture='JLSWGN',
          init_method='normal', BN=True, JL_dim=None, JL_error=0.32, n_projections=10000,
          target_dim=1000, target_number_samples=10000, target_sigma=0.01,
          run=None,
          load_saved=True):
    """
    - this is the function to use to train a Johnson-Lindenstrauss Generative Network model which uses the sliced
      Wasserstein-2 distance as objective funtion (JLSWGN) on a toy data set which is generated first. These are samples
      from a random normal distribution in a very high dimensional space
    - the function computes losses and auto-saves the model every 100 steps and automatically resumes training where it
      stopped (when load_saved=True)

    :param input_dim: the dimension of the latent space -> Z
    :param batch_size: the batch size, should be a divisor of 50k
    :param learning_rate:
    :param n_features_first: the number of feature maps in the first step of the generator
    :param epochs: the number of epochs to train for (in fact this number should be 50k/batch_size*true_epochs)
    :param fixed_noise_size: the number of pictures that is generated during training for visual progress
    :param architecture: right now only supports 'JLSWGN', 'SWGN', defaults to 'JLSWGN'
    :param init_method: the method with which the variables are initialized, support: 'normal', 'He', defaults to first
    :param BN: shall batch normalization be used
    :param JL_dim: the target dimension of the JL mapping
    :param JL_error: the max pairwise distance deviation error of the JL mapping, only applies when JL_dim=None
    :param n_projections: number of random projections in sliced Wasserstein-2 distance
    :param load_saved: whether an already existing training progress shall be loaded to continue there (if one exists)
    :param target_dim: the dimension of the target space
    :param target_number_samples: the number of samples from the target distribution that are generated and
                                  used for training
    :param target_sigma: standard deviation for the generated data
    :param run: which run of the same experiment, either None or integer
    :return:
    """

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
    if init_method not in ['normal', 'uniform']:
        init_method = 'He'

    # -------------------------------------------------------
    # JL_dim:
    if JL_dim is None:
        if JL_error is None and use_JL:
            use_JL=False
            architecture = 'SWGN'
            print
            print 'architecture changed to SWGN, since JL_dim and JL_error were None'
            print
        elif JL_error is not None:
            JL_dim = int(math.ceil(8*np.log(2*batch_size)/(JL_error**2)))
            # this uses the constant given on the Wikipedia page of "Johnson-Lindenstrauss Lemma"
    else:
        JL_error = np.round(np.sqrt(8*np.log(2*batch_size)/JL_dim), decimals=4)

    if use_JL and JL_dim >= target_dim:
        use_JL = False
        architecture='SWGN'
        JL_error = None
        JL_dim = None
        print
        print 'JL mapping is not used, since the target dimension was chosen bigger than the input dimension'
        print

    # -------------------------------------------------------
    # create unique folder name
    dir1 = 'JLSWGN_toy/'
    if run is not None:
        dir1 += '{}/'.format(run)
    directory = dir1 + str(input_dim) + '_' + str(batch_size) + '_' + str(n_features_first) + '_' + \
                str(learning_rate) + '_' + \
                str(architecture) + '_' + str(init_method) + '_' + str(BN) + '_' + str(JL_dim) + '_' + \
                str(JL_error) + '_' + \
                str(n_projections) + '_' + str(target_dim) + '_' + str(target_number_samples) +  '_' + \
                str(target_sigma) + '/'
    samples_dir = directory + 'samples/'
    model_dir = directory + 'model/'

    # create directories if they don't exist
    try:
        os.makedirs(dir1)
    except OSError:
        pass

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
    # create the target data set if it doesn't exist yet otherwise load it
    filename = 'target_data_{}_{}_{}.npy'.format(target_dim, target_number_samples, target_sigma)

    if settings.euler:
        if run is not None:
            path_name = '/cluster/scratch/fkrach/toy_data/{}/'.format(run)
        else:
            path_name = '/cluster/scratch/fkrach/toy_data/'
        try:
            os.makedirs(path_name)
        except OSError:
            pass
        if filename not in os.listdir(path_name):
            print 'generate target data on scratch directory ...'
            target_data = generate_target(path_name + filename,
                                          number_samples=target_number_samples + TEST_DATA_SIZE, N=target_dim,
                                          sigma=target_sigma)
        else:
            print 'load target data from scratch directory ...'
            target_data = np.load(path_name + filename)
    else:
        if filename not in os.listdir(dir1):
            print 'generate target data ...'
            target_data = generate_target(dir1 + filename,
                                          number_samples=target_number_samples + TEST_DATA_SIZE, N=target_dim,
                                          sigma=target_sigma)
        else:
            print 'load target data ...'
            target_data = np.load(dir1 + filename)

    target_data_test = target_data[-TEST_DATA_SIZE:]
    target_data = target_data[:-TEST_DATA_SIZE]

    # print 'target data:'
    # print target_data
    # print 'target test data:'
    # print target_data_test
    # print

    # -------------------------------------------------------
    # convenience function to build the model
    def build_model():
        """
        - function to build the model
        """
        with tf.name_scope('placeholders'):
            x_true = tf.placeholder(tf.float32, [None, target_dim])
            z = tf.placeholder(tf.float32, [None, input_dim])
            if use_JL:
                JL = tf.placeholder(tf.float32, [target_dim, JL_dim])
                P_non_normalized = tf.placeholder(tf.float32, [JL_dim, n_projections])
                P_non_normalized_SWD = tf.placeholder(tf.float32, [target_dim, n_projections])
            else:
                JL = None
                P_non_normalized = tf.placeholder(tf.float32, [target_dim, n_projections])
                P_non_normalized_SWD = None

        x_generated = generator(z, target_dim, n_features_first=n_features_first, BN=BN,
                                init_method=init_method)

        # define loss
        with tf.name_scope('loss'):
            # apply the Johnson-Lindenstrauss map, if wanted
            if use_JL:
                JL_true = tf.matmul(x_true, JL)/np.sqrt(target_dim)
                JL_gen = tf.matmul(x_generated, JL)/np.sqrt(target_dim)
            else:
                JL_true = x_true
                JL_gen = x_generated

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
            rows = np.asarray(
                [batch_size * np.floor(i * 1.0 / batch_size) for i in range(n_projections * batch_size)])
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

        # initialize variables using uniform xavier init method, see tensorflow documentation
        session.run(tf.global_variables_initializer())

        return x_true, z, x_generated, JL, P_non_normalized, P_non_normalized_SWD, SWD, g_train

    # -------------------------------------------------------
    # build the model
    x_true, z, x_generated, JL, P_non_normalized, P_non_normalized_SWD, SWD, g_train = build_model()

    # -------------------------------------------------------
    # For saving samples, taken from IWGAN
    fixed_noise = np.random.normal(size=(fixed_noise_size, input_dim)).astype('float32')

    if not settings.euler:
        import matplotlib.pyplot as plt
    else:
        # For saving plots on the cluster,
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt

    def generate_plot(frame, axis=[[0, 1]]):
        """
        this function plots 2D projections of the target data and the generated data, where the projection plane is
        specified by the axis

        :param frame:
        :param axis: an array of 2-tuples containing the projection axis (so only the standard basis can be used)
        :return:
        """
        gen_samples = session.run(x_generated, feed_dict={z: fixed_noise})
        for axis_ in axis:
            f = plt.figure()
            plt.scatter(target_data_test[:, axis_[0]], target_data_test[:, axis_[1]], s=1, c='r', label='Target')
            plt.scatter(gen_samples[:, axis_[0]], gen_samples[:, axis_[1]], s=1, c='b', label='Generated')
            plt.legend()
            plt.title('Projected points originally in dimension {}'.format(target_dim))
            plt.xlabel(axis_[0])
            plt.ylabel(axis_[1])
            f.savefig(samples_dir+'iteration_{}_axis_{}_{}.png'.format(frame, axis_[0], axis_[1]))
            plt.close(f)

    # -------------------------------------------------------
    # for saving the model create a saver
    saver = tf.train.Saver(max_to_keep=1)
    epochs_trained = 0
    tp_columns = ['epoch', 'time_for_epochs', 'SWD', 'time_for_SWD', 'W2_loss_approximation', 'time_for_W2_loss']
    training_progress = pd.DataFrame(data=None, index=None, columns=tp_columns)

    # restore the model:
    if load_saved:
        saver.restore(sess=session, save_path=model_dir + 'saved_model')
        epochs_trained = int(np.loadtxt(fname=model_dir + 'epochs.csv'))
        tp_app = pd.read_csv(filepath_or_buffer=directory + 'training_progress.csv', index_col=0, header=0)
        training_progress = pd.concat([training_progress, tp_app], axis=0, ignore_index=True)
        print 'loaded training progress, and the model, which was already trained for {} epochs'.format(
            epochs_trained)
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
                     'learning_rate', 'fixed_noise_size',
                     'architecture', 'init_method',
                     'BN', 'JL_dim', 'JL_error', 'n_projections',
                     'target_dim', 'target_number_samples', 'target_sigma',
                     'n_trainable_params_gen'],
                    [input_dim, batch_size, n_features_first,
                     learning_rate, fixed_noise_size,
                     architecture, init_method,
                     BN, JL_dim, JL_error, n_projections,
                     target_dim, target_number_samples, target_sigma,
                     n_params_gen]]
    model_config = np.transpose(model_config)
    model_config = pd.DataFrame(data=model_config)
    model_config.to_csv(path_or_buf=directory + 'model_config.csv')
    print 'saved model configuration'
    print

    # -------------------------------------------------------
    # FK: get an infinite target data generator

    def inf_train_gen():
        while True:
            np.random.shuffle(target_data)
            for batch in xrange(target_number_samples/batch_size):
                yield target_data[batch*batch_size:(batch+1)*batch_size]

    gen = inf_train_gen()

    def test_gen():
        np.random.shuffle(target_data_test)
        for batch in xrange(TEST_DATA_SIZE/batch_size):
            yield target_data_test[batch * batch_size:(batch + 1) * batch_size]

    # -------------------------------------------------------
    # training loop
    t = time.time()  # get start time

    for i in xrange(epochs - epochs_trained):
        # print the current epoch
        print('epoch={}/{}'.format(i + epochs_trained + 1, epochs))

        batch = gen.next()
        z_train = np.random.randn(batch_size, input_dim)
        if use_JL:
            JL_train = np.random.randn(target_dim, JL_dim)
            P_train = np.random.randn(JL_dim, n_projections)
            session.run(g_train, feed_dict={x_true: batch, z: z_train, JL: JL_train, P_non_normalized: P_train})
        else:
            P_train = np.random.randn(target_dim, n_projections)
            session.run(g_train, feed_dict={x_true: batch, z: z_train, P_non_normalized: P_train})

        # all 100 steps compute the losses and elapsed times, and generate images
        if (i + epochs_trained) % 100 == 99:
            # get time for last 100 epochs
            elapsed_time = time.time() - t

            # generate sample images from fixed noise
            if GENERATE_SAMPLES:
                axis = np.random.randint(low=0, high=target_dim, size=(10, 2)).tolist()
                generate_plot(i + epochs_trained + 1, axis=axis)
                print 'generated plots'

            # compute and save losses on dev set
            t = time.time()
            dev_SWD = []
            print 'compute SWD ...'
            j = 0
            for data_dev in test_gen():
                if not settings.euler:
                    # progress bar
                    sys.stdout.write('\r>> Compute SWD %.1f%%' % (float(j) / float(TEST_DATA_SIZE / batch_size) * 100.0))
                    sys.stdout.flush()
                j += 1

                z_train_dev = np.random.randn(batch_size, input_dim)
                P_train_dev = np.random.randn(target_dim, n_projections)
                if use_JL:
                    _dev_SWD = session.run(SWD, feed_dict={x_true: data_dev, z: z_train_dev,
                                                           P_non_normalized_SWD: P_train_dev})
                else:
                    _dev_SWD = session.run(SWD, feed_dict={x_true: data_dev, z: z_train_dev,
                                                           P_non_normalized: P_train_dev})
                dev_SWD.append(_dev_SWD)
            dev_SWD = np.mean(dev_SWD)
            t_SWD = time.time() - t

            # also calculate an approximation of the wasserstein-2 distance (in fact the distance between 2 gaussian
            # distributions with same covariance matrix, which is given by the distance of their means)
            t = time.time()
            z_train_dev = np.random.randn(TEST_DATA_SIZE, input_dim)
            dev_W2_loss = np.mean(target_data_test, axis=0) - np.mean(
                session.run(x_generated, feed_dict={z: z_train_dev}), axis=0)
            dev_W2_loss = np.mean(dev_W2_loss**2)
            t_loss = time.time() - t

            tp_app = pd.DataFrame(data=[[i + epochs_trained + 1, elapsed_time, dev_SWD, t_SWD, dev_W2_loss, t_loss]],
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
    print 'session closed'
    print

    # -------------------------------------------------------
    # when training is done send email
    if send:
        subject = 'JLSWGN (TOY) training finished'
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
        ITERS = 1
        parameters = [(INPUT_DIM, BATCH_SIZE, N_FEATURES_FIRST, 1e-2, ITERS, FIXED_NOISE_SIZE,
                       'JLSWGN', 'normal', True, 28*28/2, None, 10000, 1000, 10000, 0.01),
                      (INPUT_DIM, BATCH_SIZE, N_FEATURES_FIRST, 1e-2, ITERS, FIXED_NOISE_SIZE,
                       'SWGN', 'normal', True, None, None, 10000, 1000, 10000, 0.01)]
    results = Parallel(n_jobs=nb_jobs)(delayed(train)(input_dim, batch_size, n_features_first, learning_rate,
                                                      epochs, fixed_noise_size,
                                                      architecture, init_method, BN, JL_dim, JL_error, n_projections,
                                                      target_dim, target_number_samples, target_sigma, run)
                                       for input_dim, batch_size, n_features_first, learning_rate,
                                           epochs, fixed_noise_size,
                                           architecture, init_method, BN, JL_dim, JL_error, n_projections,
                                           target_dim, target_number_samples, target_sigma, run in parameters)

    if settings.send_email:
        subject = 'JLSWGN (TOY) parallel training finished'
        body = 'to download all the results of this parallel run use (in the terminal):\n\n'
        for directory in results:
            body += 'scp -r fkrach@euler.ethz.ch:/cluster/home/fkrach/MasterThesis/MTCode1/'+directory+' .; '
        send_email.send_email(subject=subject, body=body, file_names=None)
    return 0



if __name__ == '__main__':
    # -------------------------------------------------------
    # parallel training
    param_array = settings.JLSWGN_toy_param_array3
    if N_CPUS_PARALLEL is None:
        nb_jobs = settings.number_parallel_jobs
    else:
        nb_jobs = N_CPUS_PARALLEL

    # -----------------------------------
    # # first generate all data
    # if not os.path.isdir('JLSWGN_toy/'):
    #     call(['mkdir', 'JLSWGN_toy/'])
    #
    # for param in param_array:
    #     target_dim = param[12]
    #     target_number_samples = param[13]
    #     target_sigma = param[14]
    #
    #     # create the target data set if it doesn't exist yet otherwise load it
    #     filename = 'target_data_{}_{}_{}.npy'.format(target_dim, target_number_samples, target_sigma)
    #
    #     if settings.euler:
    #         if filename not in os.listdir('/cluster/scratch/fkrach/'):
    #             print 'generate target data on scratch directory ...'
    #             generate_target('/cluster/scratch/fkrach/' + filename,
    #                             number_samples=target_number_samples + TEST_DATA_SIZE,
    #                             N=target_dim, sigma=target_sigma)
    #
    #     else:
    #         if filename not in os.listdir('JLSWGN_toy/'):
    #             print 'generate target data ...'
    #             generate_target('JLSWGN_toy/' + filename, number_samples=target_number_samples + TEST_DATA_SIZE,
    #                             N=target_dim, sigma=target_sigma)
    # -----------------------------------

    # parallel_training(parameters=param_array, nb_jobs=nb_jobs)
    # parallel_training(nb_jobs=nb_jobs)

    train(init_method='He', epochs=4000, load_saved=True, architecture='JLSWGN', batch_size=50, n_features_first=256,
          JL_dim=None,
          JL_error=0.32,
          learning_rate=1e-2, target_dim=10000, target_sigma=0.01)


    # -----------------------------------
    # # intra_set SWD
    # path = 'JLSWGN_toy/2/target_data_1000_10000_0.01.npy'
    # SWD = compute_intra_set_SWD(path=path, batch_size=50, L=10000, N=1000)
    # print 'inter_set_SWD for file: {}'.format(path)
    # print SWD



