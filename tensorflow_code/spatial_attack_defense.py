"""
Implementation of a spatial attack.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from itertools import product, repeat
import random
import scipy
import tensorflow as tf
import numpy as np

from pgd_attack import LinfPGDAttack, SpatialPGDAttack

class SpatialAttack:
    def __init__(self, model, config, method=None, worstofk=None,
                 attack_limits=None, fo_epsilon=2.0, fo_step_size=2.,
                 fo_num_steps=5):
        self.model = model
        self.grid_store = []

        if config.use_linf:
            self.linf_attack = LinfPGDAttack(
                model, config, fo_epsilon, fo_step_size, fo_num_steps)
        else:
            self.linf_attack = None

        self.use_spatial = config.use_spatial
        if config.use_spatial:
            # Attack method
            if method == None:
                self.method = config.spatial_method
            else:
                self.method = method

            # Attack parameters
            if attack_limits == None:
                self.limits = config.spatial_limits
            else:
                self.limits = attack_limits

            if config.only_rotation:
                self.limits = [0, 0, self.limits[2]]

            if config.only_translation:
                self.limits = [self.limits[0], self.limits[1], 0]

            # Attack method parameters
            if self.method == 'grid':
                self.granularity = config.grid_granularity
            elif self.method == 'random':
                if worstofk == None:
                    self.random_tries = config.random_tries
                else:
                    self.random_tries = worstofk
            elif self.method == 'fo':
                self.fo_attack = SpatialPGDAttack(
                    model, config, fo_epsilon, fo_step_size, fo_num_steps)
            else:
                raise NotImplementedError

    def perturb(self, x_nat, y, max_func, sess):
        if not self.use_spatial:
            t = np.zeros([len(x_nat), 3])
            if self.linf_attack:
                x = self.linf_attack.perturb(x_nat, y, sess, trans=t)
            else:
                x = x_nat
            return x, t
        if self.method == 'grid':
            return self.perturb_grid(x_nat, y, sess, -1)
        elif self.method == 'fo':
            return self.fo_attack.perturb(x_nat, y, sess)
        else: # random
            return self.perturb_grid(x_nat, y, max_func, sess, self.random_tries)

    def perturb_grid(self, x_nat, y, max_func, sess, random_tries=-1):
        n = len(x_nat)
        if random_tries > 0:
            # subsampling this list from the grid is a bad idea, instead we
            # will randomize each example from the full continuous range
            grid = [(42, 42, 42) for _ in range(random_tries)] # dummy list
        else: # exhaustive grid
            grid = product(*list(np.linspace(-l, l, num=g)
                           for l, g in zip(self.limits, self.granularity)))

        worst_x = np.copy(x_nat)
        worst_t = np.zeros([n, 3])
        k = 0

        if self.linf_attack:
            raise NotImplementedError
        else:
            x = x_nat

        no_op = np.zeros([n, 3])
        # computing pre_softmax of f(x), notice f(x) is not true label y
        if max_func == "cce":
            pass
        else:
            nat_dict = {self.model.x_input: x,
                        self.model.y_input: y,
                        self.model.is_training: False,
                        self.model.transform: no_op}
            f_x_nat_presoftmax = sess.run(self.model.pre_softmax, feed_dict=nat_dict)

        for tx, ty, r in grid:
            if random_tries > 0:
                # randomize each example separately
                t = np.stack((np.random.uniform(-l, l, n) for l in self.limits), axis=1)
            else:
                t = np.stack(repeat([tx, ty, r], n))

            adv_dict = {self.model.x_input: x,
                        self.model.y_input: y,
                        self.model.is_training: False,
                        self.model.transform: t}

            if max_func == "cce":  # w.r.t. the cce, not regularizer
                adv_loss = sess.run(self.model.y_xent,
                                    feed_dict=adv_dict)  # shape (bsize,)

            elif max_func == "l2":  # w.r.t. the regularizer
                f_x_adv_presoftmax = sess.run(self.model.pre_softmax,
                                              feed_dict=adv_dict)  # shape (bsize,)
                adv_loss = self.l2_reg_loss(f_x_nat_presoftmax, f_x_adv_presoftmax)

            elif max_func == "kl":
                f_x_adv_presoftmax = sess.run(self.model.pre_softmax,
                                              feed_dict=adv_dict)  # shape (bsize,)
                adv_loss = self.kl_reg_loss(f_x_nat_presoftmax, f_x_adv_presoftmax)

            else:
                raise NotImplementedError

            adv_loss = np.asarray(adv_loss)
            # update indices if adv_loss is larger than previous max_adv_loss
            if k == 0:
                # in first iteration update all
                idx = np.ones(n).astype(bool)
            else:
                idx = adv_loss > max_adv_loss
            idx = np.expand_dims(idx, axis=-1)  # shape (bsize, 1)
            if k == 0:
                max_adv_loss = adv_loss
            else:
                max_adv_loss = np.maximum(adv_loss, max_adv_loss)
            worst_t = np.where(idx, t, worst_t)  # shape (bsize, 3)

            idx = np.expand_dims(idx, axis=-1)
            idx = np.expand_dims(idx, axis=-1)  # shape (bsize, 1, 1, 1)
            worst_x = np.where(idx, x, worst_x,)  # shape (bsize, 32, 32, 3)
            k += 1
        return worst_x, worst_t

    def l2_reg_loss(self, dist_a, dist_b):
        assert dist_a.shape == dist_b.shape
        return np.sum(np.square(dist_a - dist_b), axis=1)

    # pass the presoftmax in
    def kl_reg_loss(self, dist_nat, dist_adv):
        assert dist_nat.shape == dist_adv.shape
        #  compute KL-div of f(x) and f(x')
        epsilon = np.zeros(dist_nat.shape)
        epsilon.fill(1e-08)

        prob_adv = scipy.special.softmax(dist_adv, axis=1) + epsilon
        prob_nat = scipy.special.softmax(dist_nat, axis=1) + epsilon

        # scipy.stats.entropy calculate the KL divergence (although it's called entropy)
        return scipy.stats.entropy(np.transpose(prob_nat), np.transpose(prob_adv))

