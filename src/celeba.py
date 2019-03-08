"""
author: Florian Krach
used parts of the code of the following implementations:
- Improved Wasserstein-GAN, https://github.com/igul222/improved_wgan_training
"""

import numpy as np


# first run the file celeba_preprocessing.py

def celebA_generator(filename, batch_size):
    images = np.load(filename).astype('uint8')
    images = images.reshape([len(images), -1])  # flatten the images, before [N, 32, 32, 3 or 1]

    def get_epoch():
        np.random.shuffle(images)

        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], True)

    return get_epoch


def load(batch_size, data_dir, black_white=True):
    if black_white:
        f1 = data_dir + 'train_bw.npy'
        f2 = data_dir + 'test_bw.npy'
    else:
        f1 = data_dir + 'train.npy'
        f2 = data_dir + 'test.npy'

    return (
        celebA_generator(f1, batch_size),
        celebA_generator(f2, batch_size)
    )

