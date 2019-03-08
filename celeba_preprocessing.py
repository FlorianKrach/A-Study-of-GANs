"""
author: Florian Krach

this file is used to preprocess the CelebA images and has to be run before using the data set
"""

import PIL.Image as image  # for reading and editing images in e.g. jpeg format
from scipy.ndimage import imread  # for reading an e.g. jpeg image into a np array
import os
import sys
import numpy as np


# first download the img_align_celeba.zip file (Aligned and Cropped Images) from the Google Drive link at
# http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# then unzip the file and place it in the path below (or change the path)
path = '../data/img_align_celeba/'
path2 = '../data/celebA/'

size = 64  # size of picture: 32 or 64 or 128
n_pictures = 60000  # number of pictures to preprocess, or None if all
n_train = None  # number of training images or None (then 5/6*n_pictures)
black_white = False  # true or false


# -------------------------------------------------------
path2 = path2[:-1] + str(size) + '/'
print path2
if not os.path.isdir(path2):
    os.makedirs(path2)

if n_pictures is None:
    n_pictures = len(os.listdir(path))
if n_train is None:
    n_train = int(n_pictures * 5 / 6)


# -------------------------------------------------------
# preprocess the images
print 'Preprocessing: cropping and scaling images'
for j, f in enumerate(os.listdir(path)):
    if j < n_pictures:
        sys.stdout.write('\r>> Processing CelebA %.1f%%' % (float(j) / float(n_pictures) * 100.0))
        sys.stdout.flush()

        im = image.open(path+f)
        cropped = im.crop((30, 40, 30+138, 40+138))
        cropped.thumbnail((size, size), image.ANTIALIAS)
        cropped.save(path2+f)
    else:
        break


# -------------------------------------------------------
# save the images as np array
if black_white:
    print '\nConverting to black and white np arrays with integer values in [0, 255] of size [N, {}, {}]'.format(size, size)
    images_train = np.empty(shape=[n_train, size, size])
    images_test = np.empty(shape=[n_pictures - n_train, size, size])
    for j, f in enumerate(os.listdir(path2)):
        sys.stdout.write('\r>> Converting %.1f%%' % (float(j) / float(n_pictures) * 100.0))
        sys.stdout.flush()
        im = imread(path2 + f, flatten=True)
        if j < n_train:
            images_train[j] = im
            # print im
            # print im.shape
        else:
            images_test[j - n_train] = im

    print 'writing files ...'
    np.save(path2 + 'train_bw', images_train)
    np.save(path2 + 'test_bw', images_test)
    print 'files saved:\n{}\n{}'.format(path2 + 'train_bw.npy', path2 + 'test_bw.npy')


else:
    print '\nConverting to np arrays with integer values in [0, 255] of size [N, {}, {}, 3]'.format(size, size)
    images_train = np.empty(shape=[n_train, size, size, 3])
    images_test = np.empty(shape=[n_pictures-n_train, size, size, 3])
    for j, f in enumerate(os.listdir(path2)):
        sys.stdout.write('\r>> Converting %.1f%%' % (float(j) / float(n_pictures) * 100.0))
        sys.stdout.flush()
        im = imread(path2+f)
        if j < n_train:
            images_train[j] = im
            # print im
            # print im.shape
        else:
            images_test[j-n_train] = im

    print 'writing files ...'
    np.save(path2+'train', images_train)
    np.save(path2 + 'test', images_test)
    print 'files saved:\n{}\n{}'.format(path2+'train.npy', path2+'test.npy')








