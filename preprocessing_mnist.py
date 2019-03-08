"""
author: Florian Krach
"""

import numpy as np
import scipy.ndimage as image
import gzip
import cPickle as pickle
import settings
import save_images
import urllib
import os


filepath = settings.filepath_mnist
data_dir = '../data/MNIST/'
url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

# download MNIST data if not existent
if not os.path.isfile(filepath):
    print "Couldn't find MNIST dataset in given filepath, downloading..."
    urllib.urlretrieve(url, filepath)


def zoom(size, interpolation_order=1, npy=False):
    """
    :param size: size of the new images
    :param interpolation_order: order of the spline used for interpolation, in 0-5
    :return:
    """
    print
    print 'create mnist data that is zoomed (and interpolated with spline of order={}), s.t. every image has size:' \
          ' {}x{}'.format(interpolation_order, size, size)
    with gzip.open(filepath, 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)

    images, targets = train_data
    images1, targets1 = dev_data
    images = (np.reshape(images, (-1, 28, 28)) * 255.99).astype('uint8')
    images1 = (np.reshape(images1, (-1, 28, 28)) * 255.99).astype('uint8')

    factor = size / 28.

    images = image.zoom(images, zoom=(1, factor, factor), order=interpolation_order)
    images1 = image.zoom(images1, zoom=(1, factor, factor), order=interpolation_order)
    images = (np.reshape(images, (len(images), -1))).astype('uint8')
    images1 = (np.reshape(images1, (len(images1), -1))).astype('uint8')

    if not npy:
        data = (images, images1)
        filename = data_dir + 'mnist{}_zoom_{}.pkl.gz'.format(size, interpolation_order)
        print 'saving file ...'
        with gzip.open(filename=filename, mode='wb') as f:
            pickle.dump(data, file=f)
        print 'done\n'
    else:
        print 'saving files ...'
        filename = data_dir + 'mnist{}_zoom_{}_train.npy'.format(size, interpolation_order)
        filename1 = data_dir + 'mnist{}_zoom_{}_test.npy'.format(size, interpolation_order)
        np.save(file=filename, arr=images)
        np.save(file=filename1, arr=images1)
        print 'done\n'

    return filename


def enlarge_border(size, npy=False):
    print
    print 'create mnist data with larger black border, s.t. every image has size: {}x{}'.format(size, size)
    with gzip.open(filepath, 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)

    images, targets = train_data
    images1, targets1 = dev_data

    new = np.zeros((len(images), size, size)).astype('uint8')
    new1 = np.zeros((len(images1), size, size)).astype('uint8')

    p = (size - 28) / 2

    new[:, p:-p, p:-p] = (np.reshape(images, (-1, 28, 28))*255.99).astype('uint8')
    new1[:, p:-p, p:-p] = (np.reshape(images1, (-1, 28, 28))*255.99).astype('uint8')
    new = np.reshape(new, (len(new), -1))
    new1 = np.reshape(new1, (len(new1), -1))

    if not npy:
        data = (new, new1)
        filename = data_dir + 'mnist{}_border.pkl.gz'.format(size)
        print 'saving file ...'
        with gzip.open(filename=filename, mode='wb') as f:
            pickle.dump(data, file=f)
        print 'done\n'
    else:
        print 'saving files ...'
        filename = data_dir + 'mnist{}_border_train.npy'.format(size)
        filename1 = data_dir + 'mnist{}_border_test.npy'.format(size)
        np.save(file=filename, arr=new)
        np.save(file=filename1, arr=new1)
        print 'done\n'

    return filename


# ------------------------------------------
# functions for loading the data
def mnist_generator(images, batch_size, targets):
    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(targets)

        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], targets[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(filename, batch_size, npy=False):
    if not npy:
        with gzip.open(filename+'.pkl.gz', 'rb') as f:
            train_data, dev_data = pickle.load(f)
    else:
        train_data = np.load(filename+'_train.npy')
        dev_data = np.load(filename+'_test.npy')

    with gzip.open(filepath, 'rb') as f:
        _train_data, _dev_data, _test_data = pickle.load(f)
    _, targets = _train_data
    _, targets1 = _dev_data

    return (mnist_generator(train_data, batch_size, targets), len(train_data),
            mnist_generator(dev_data, batch_size, targets1), len(dev_data))



if __name__ == '__main__':
    # size = 64
    # enlarge_border(size)
    # with gzip.open(data_dir+'mnist{}_border.pkl.gz'.format(size), 'rb') as f:
    #     train_data, dev_data = pickle.load(f)
    #
    # image_batch = np.reshape(train_data[:64], (-1, size, size))
    # save_images.save_images(image_batch, save_path='test_image4.png')

    # size = 128
    # fp = zoom(size, 1, npy=True)
    # # with gzip.open(fp, 'rb') as f:
    # #     train_data, dev_data = pickle.load(f)
    #
    # train_data = np.load(fp)
    #
    # image_batch = np.reshape(train_data[:64], (-1, size, size))
    # save_images.save_images(image_batch, save_path='test_image1.png')

    train, l_train, dev, l_dev = load(data_dir+'mnist128_zoom_1', 100, npy=False)
    data = train()
    im, targ = data.next()
    save_images.save_images(im, save_path='test_image2.png')
    print targ

    # import tensorflow as tf
    # o = tf.one_hot(targ, depth=10)
    # sess = tf.Session()
    # print sess.run(o)