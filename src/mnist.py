"""
author: Florian Krach
modified version of the code from: Improved Wasserstein-GAN, https://github.com/igul222/improved_wgan_training
"""

import numpy as np

import os
import urllib
import gzip
import cPickle as pickle

import settings


def mnist_generator(data, batch_size, n_labelled, limit=None, which_numbers=None):
    """
    :param data:
    :param batch_size:
    :param n_labelled:
    :param limit:
    :param which_numbers: an arry conaining the digits to use, so a subset of {0,1,...9}
    :return:
    """
    images, targets = data

    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(targets)
    if limit is not None:
        print "WARNING ONLY FIRST {} MNIST DIGITS".format(limit)
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]

    if which_numbers is not None:
        print 'use only the digits {} for training'.format(which_numbers)
        index = []
        for i in range(len(targets)):
            if targets[i] in which_numbers:
                index += [i]

        # if len(ind) is not multiple of batch_size, delete some of the indices
        n_delete = len(index) % batch_size
        index = index[:-n_delete]
        images = images[index, :]
        targets = targets[index]

    if n_labelled is not None:
        labelled = np.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(targets)

        if n_labelled is not None:
            np.random.set_state(rng_state)
            np.random.shuffle(labelled)

        image_batches = images.reshape(-1, batch_size, 28, 28)  # FK: edited shape from (-1, batch_size, 784)
        target_batches = targets.reshape(-1, batch_size)

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in xrange(len(image_batches)):
                yield (np.copy(image_batches[i]), np.copy(target_batches[i]), np.copy(labelled))

        else:

            for i in xrange(len(image_batches)):
                yield (np.copy(image_batches[i]), np.copy(target_batches[i]))

    return get_epoch


def load(batch_size, test_batch_size, n_labelled=None, which_numbers=None):
    filepath = settings.filepath_mnist  # FK: changed
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print "Couldn't find MNIST dataset in given filepath, downloading..."
        urllib.urlretrieve(url, filepath)

    with gzip.open(filepath, 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)

    return (
        mnist_generator(train_data, batch_size, n_labelled, which_numbers=which_numbers),
        mnist_generator(dev_data, test_batch_size, n_labelled, which_numbers=which_numbers),
        mnist_generator(test_data, test_batch_size, n_labelled, which_numbers=which_numbers)
    )


if __name__ == '__main__':
    data, _, _ = load(50, 50, which_numbers=(0,1,2,3))
    print data().next()
    img = data().next()[0][0]
    import save_images
    save_images.save_images(img.reshape([1,28,28]), 'test.png')