"""
author: Florian Krach
"""

import numpy as np
import math
import tensorflow as tf
from tensorflow.contrib import layers
import save_images
import settings
import send_email
import os
from subprocess import call
import tensorflow.contrib.slim as slim
import pandas as pd
import time
from joblib import Parallel, delayed
import cifar10, celeba,preprocessing_mnist


def JL_reconstruction(data='mnist', JL_dim=32*32/2, batch_size=100, seed=None):
    # -------------------------------------------------------
    # get the dataset as infinite generator
    if seed is not None:
        np.random.seed(seed)

    if data == 'cifar10':
        data_dir = settings.filepath_cifar10
        train_gen, dev_gen = cifar10.load(batch_size, data_dir=data_dir)
        picture_size = 32 * 32 * 3
    elif data == 'celebA32':
        data_dir = settings.filepath_celebA32
        train_gen, dev_gen = celeba.load(batch_size, data_dir=data_dir, black_white=False)
        picture_size = 32 * 32 * 3
    elif data == 'mnist':
        filename = '../data/MNIST/mnist32_zoom_1'
        train_gen, n_samples_train, dev_gen, n_samples_test = preprocessing_mnist.load(filename, batch_size, npy=True)
        picture_size = 32*32
    elif data == 'celebA32_bw':
        data_dir = settings.filepath_celebA32
        train_gen, dev_gen = celeba.load(batch_size, data_dir=data_dir, black_white=True)
        picture_size = 32 * 32

    # -------------------------------------------------------
    # make directories
    dir1 = 'JL_reconstruction/'
    path = dir1 + data + '/'
    if not os.path.isdir(dir1):
        call(['mkdir', dir1])
    if not os.path.isdir(path):
        call(['mkdir', path])

    # -------------------------------------------------------
    # JL mapping
    A = np.random.randn(JL_dim, picture_size) / np.sqrt(picture_size)
    ATA = np.matmul(np.transpose(A), A)

    # JL error
    JL_error = np.round(np.sqrt(8 * np.log(2 * batch_size) / JL_dim), decimals=4)
    print '\ndata dimension: {}'.format(picture_size)
    print 'JL dimension:   {}'.format(JL_dim)
    print 'batch size:     {}'.format(batch_size)
    print 'JL error:       {}\n'.format(JL_error)

    # -------------------------------------------------------
    # encode and decode data
    im = train_gen().next()[0]
    im1 = im / 255.99

    reconstruction = np.matmul(im1, ATA) #/ float(picture_size)
    reconstruction = (255.99*np.clip(reconstruction, 0, 1)).astype('uint8')

    # reconstruction = np.matmul(im, ATA)  # / float(picture_size)
    # reconstruction = (np.clip(reconstruction, 0, 255)).astype('uint8')

    save_images.save_images(im, save_path=path+'true_images.png')
    save_images.save_images(reconstruction, save_path=path+'JL_reconstructed_image.png')

    im_d = np.zeros((100, picture_size))
    for i in range(batch_size):
        A = np.random.randn(JL_dim, picture_size) / np.sqrt(picture_size)
        ATA = np.matmul(np.transpose(A), A)
        reconstruction = np.matmul(im1[i].reshape((1, picture_size)), ATA)  # / float(picture_size)
        reconstruction = (255.99 * np.clip(reconstruction, 0, 1)).astype('uint8')
        im_d[i] = reconstruction.reshape((picture_size,))
    im_d = im_d.astype('uint8')
    save_images.save_images(im_d, save_path=path + 'different_JL_reconstructed_image.png')


if __name__ == '__main__':
    JL_reconstruction(data='mnist', JL_dim=32*32/1, batch_size=100)




