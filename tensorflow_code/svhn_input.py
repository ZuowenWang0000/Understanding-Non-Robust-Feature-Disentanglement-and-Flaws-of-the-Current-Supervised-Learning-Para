"""
Utilities for importing the CIFAR10 dataset.

Each image in the dataset is a numpy array of shape (32, 32, 3), with the values
being unsigned integers (i.e., in the range 0,1,...,255).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import random
import sys
import tensorflow as tf
version = sys.version_info

import numpy as np
import scipy.io as scio
#from base import RNGDataFlow

SVHN_URL = "http://ufldl.stanford.edu/housenumber"

class SVHNData(object):
    """
    Unpickles the CIFAR10 dataset from a specified folder containing a pickled
    version following the format of Krizhevsky which can be found
    [here](https://www.cs.toronto.edu/~kriz/cifar.html).

    Inputs to constructor
    =====================

        - path: path to the pickled dataset. The training data must be pickled
        into five files named data_batch_i for i = 1, ..., 5, containing 10,000
        examples each, the test data
        must be pickled into a single file called test_batch containing 10,000
        examples, and the 10 class names must be
        pickled into a file called batches.meta. The pickled examples should
        be stored as a tuple of two objects: an array of 10,000 32x32x3-shaped
        arrays, and an array of their 10,000 true labels.

    """

    _Cache = {}

    def __init__(self, path):
        """
        Args:
            name (str): 'train', 'test', or 'extra'.
            data_dir (str): a directory containing the original {train,test,extra}_32x32.mat.
            shuffle (bool): shuffle the dataset.
        """

        train_filename = 'train'
        eval_filename = 'test'

        # Load train data
        train_images, train_labels = self._load_datafile(
            os.path.join(path, train_filename + '_32x32.mat'))

        # Load test data
        eval_images, eval_labels = self._load_datafile(
            os.path.join(path, eval_filename + '_32x32.mat'))

        self.train_data = Dataset(train_images, train_labels)
        self.eval_data = Dataset(eval_images, eval_labels)

    @staticmethod
    def _load_datafile(filename):
        print("Loading {} ...".format(filename))
        data = scio.loadmat(filename)
        images = data['X'].transpose(3, 0, 1, 2)
        labels = data['y'].reshape((-1))
        labels[labels == 10] = 0
        return images, labels 
        

    @staticmethod
    def get_per_pixel_mean():
        """
        Returns:
            a 32x32x3 image
        """
        a = SVHNDigit('train')
        b = SVHNDigit('test')
        c = SVHNDigit('extra')
        return np.concatenate((a.X, b.X, c.X)).mean(axis=0)


class Dataset(object):
    """
    Dataset object implementing a simple batching procedure.
    """
    def __init__(self, xs, ys):
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.n)

    def get_next_batch(self, batch_size, multiple_passes=False,
                       reshuffle_after_pass=True):
        epoch_done = False
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end],...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end],...]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            epoch_done = True
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        self.batch_start += actual_batch_size
        return batch_xs, batch_ys, epoch_done


class AugmentedSVHNData(object):
    """
    Data augmentation wrapper over a loaded dataset.

    Inputs to constructor
    =====================
        - raw_data: the loaded SVHN dataset, via the CIFAR10Data class
        - sess: current tensorflow session
    """
    def __init__(self, raw_data, sess):
        assert isinstance(raw_data, SVHNData)
        self.image_size = 32

        # create augmentation computational graph
        self.x_input_placeholder = tf.placeholder(tf.float32,
                                                  shape=[None, 32, 32, 3])

        # random transforamtion parameters
        flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img),
                            self.x_input_placeholder)

        self.augmented = flipped

        self.train_data = AugmentedDataset(raw_data.train_data, sess,
                                             self.x_input_placeholder,
                                             self.augmented)
        self.eval_data = AugmentedDataset(raw_data.eval_data, sess,
                                             self.x_input_placeholder,
                                             self.augmented, 1)
        # self.label_names = raw_cifar10data.label_names


class AugmentedDataset(object):
    """
    Dataset object with built-in data augmentation. When performing
    adversarial attacks, we cannot include data augmentation as part of the
    model. If we do the adversary will try to backprop through it.
    """
    def __init__(self, raw_datasubset, sess, x_input_placeholder,
                 augmented, to_cache=False):
        self.sess = sess
        self.raw_datasubset = raw_datasubset
        self.x_input_placeholder = x_input_placeholder
        self.augmented = augmented

        if to_cache:
            # Get the actual data from the raw
            self.xs = self.sess.run(self.augmented,
                                    feed_dict={self.x_input_placeholder:
                                                self.raw_datasubset.xs})
            self.n = self.xs.shape[0]
            self.ys = self.raw_datasubset.ys
        else:
            self.xs = None
            self.n = None
            self.ys = None

    def get_next_batch(self, batch_size, multiple_passes=False,
                       reshuffle_after_pass=True):
        raw_batch = self.raw_datasubset.get_next_batch(batch_size,
                                                       multiple_passes,
                                                       reshuffle_after_pass)
        epoch_done = raw_batch[2]
        #images = raw_batch[0].astype(np.float32)
        return (self.sess.run(
                     self.augmented,
                     feed_dict={self.x_input_placeholder:
                                    raw_batch[0]}), raw_batch[1], epoch_done)
