"""
Implementation of a PGD attack bounded in L_infty.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class LinfPGDAttack:
  def __init__(self, model, config, epsilon, step_size, num_steps):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = config.random_start

    if config.loss_function == 'xent':
      loss = model.xent
    elif config.loss_function == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax
                                                    - 1e4 * label_mask, axis=1)
      loss = wrong_logit - correct_logit
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]


  def perturb(self, x_nat, y, sess, trans=None):
    """
    Given a set of examples (x_nat, y), returns a set of adversarial
    examples within epsilon of x_nat in l_infinity norm. An optional
    spatial perturbations can be given as (trans_x, trans_y, rot).
    """

    if self.rand:
        x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
        x = np.copy(x_nat)

    if trans is None:
        trans = np.zeros([len(x_nat), 3])

    no_op = np.zeros([len(x_nat), 3])
    f_x_dict = {self.model.x_input: x,
                self.model.y_input: y,
                self.model.is_training: False,
                self.model.transform: no_op}

    f_x = sess.run(self.model.predictions, feed_dict=f_x_dict)

    for i in range(self.num_steps):
        curr_dict = {self.model.x_input: x,
                     self.model.y_input: f_x,
                     self.model.transform: trans,
                     self.model.is_training: False}
        grad = sess.run(self.grad, feed_dict=curr_dict)

        x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

        x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
        x = np.clip(x, 0, 255) # ensure valid pixel range

    return x

class L2PGDAttack:
  def __init__(self, model, config, epsilon, step_size, num_steps):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = config.random_start

    if config.loss_function == 'xent':
      loss = model.xent
    elif config.loss_function == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax
                                                    - 1e4 * label_mask, axis=1)
      loss = wrong_logit - correct_logit
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]

  def uniform_weights(self, n_attacks, n_samples):
      x = np.random.uniform(size=(n_attacks, n_samples))
      y = np.maximum(-np.log(x), 1e-8)
      return y / np.sum(y, axis=0, keepdims=True)

  def init_delta(self, x, attack, weight):
      if not self.rand:
          return np.zeros_like(x)

      assert len(weight) == len(x)
      eps = (self.epsilon * weight).reshape(len(x), 1, 1, 1)

      # if attack["type"] == "linf":
      #     return np.random.uniform(-eps, eps, x.shape)
      r = np.random.randn(*x.shape)
      norm = np.linalg.norm(r.reshape(r.shape[0], -1), axis=-1).reshape(-1, 1, 1, 1)
      return (r / norm) * eps

  def delta_update(self, old_delta, g, x_adv, attack, x_min, x_max, weight, seed=None, t=None):
      assert len(weight) == len(x_adv)

      eps_w = self.epsilon * weight
      eps = eps_w.reshape(len(x_adv), 1, 1, 1)

      # if attack["type"] == "linf":
      #     a = attack.get('a', (2.5 * eps) / attack["k"])
      #     new_delta = old_delta + a * np.sign(g)
      #     new_delta = np.clip(new_delta, -eps, eps)
      #
      #     new_delta = np.clip(new_delta, x_min - (x_adv - old_delta), x_max - (x_adv - old_delta))
      #     return new_delta

      # elif attack["type"] == "l2":
          # a = attack.get('a', (2.5 * eps) / attack["k"])


      # TODO  check if k is num_steps?
      # "a" is step_size ,  "k" is number of steps
      # epsilon: the maximum allowed perturbation per pixel
      if self.step_size is None:
          a = 2.5 * self.epsilon / self.num_steps
      else:
          a = self.step_size
      bad_pos = ((x_adv == x_max) & (g > 0)) | ((x_adv == x_min) & (g < 0))
      g[bad_pos] = 0

      # g is gradient
      g = g.reshape(len(g), -1)
      # divided by l2 norm of gradient
      # np.maximum is to avoid error amplification (NaN)
      # some dimensionality stuffs are for multi attack, but also works for single attack
      g /= np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
      g = g.reshape(old_delta.shape)

      # a*g = step_size * gradient
      new_delta = old_delta + a * g
      new_delta_norm = np.linalg.norm(new_delta.reshape(len(new_delta), -1), axis=-1).reshape(-1, 1, 1, 1)
      new_delta = new_delta / np.maximum(new_delta_norm, 1e-8) * np.minimum(new_delta_norm, eps)
      new_delta = np.clip(new_delta, x_min - (x_adv - old_delta), x_max - (x_adv - old_delta))
      return new_delta


  def norm_perturb(self, x_nat, y, sess, norm_attacks, norm_weights, trans=None):
        if len(norm_attacks) == 0:
            return x_nat

        x_min = 0 #self.x_min
        x_max = 255 #self.x_max

        if trans is None:
            trans = np.zeros([len(x_nat), 3])

        iters = self.num_steps #[a["k"] for a in norm_attacks]
        # assert (np.all(np.asarray(iters) == iters[0]))

        deltas = np.asarray([self.init_delta(x_nat, attack, weight)
                             for attack, weight in zip(norm_attacks, norm_weights)])
        x_adv = np.clip(x_nat + np.sum(deltas, axis=0), 0, 255)


        # a seed that remains constant across attack iterations
        seed = np.random.randint(low=0, high=2**32-1)

        for i in range(iters):
            grad = sess.run(self.grad, feed_dict={self.model.x_input: x_adv,
                                                  self.model.y_input: y,
                                                  self.model.is_training: False,
                                                  self.model.transform: trans})

            deltas[i % len(norm_attacks)] = self.delta_update(deltas[i % len(norm_attacks)],
                                                         grad,
                                                         x_adv,
                                                         norm_attacks[i % len(norm_attacks)],
                                                         x_min, x_max,
                                                         norm_weights[i % len(norm_attacks)],
                                                         seed=seed, t=i+1)

            x_adv = np.clip(x_nat + np.sum(deltas, axis=0), x_min, x_max)

        return np.clip(x_nat + np.sum(deltas, axis=0), x_min, x_max)

  def perturb(self, x_nat, y, sess, trans=None):

    # if self.rand:
    #     r = np.random.randn(*x.shape)
    #     norm = np.linalg.norm(r.reshape(r.shape[0], -1), axis=-1).reshape(-1, 1, 1, 1)
    #     return (r / norm) * eps
    #     # x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    # else:
    #     x = np.copy(x_nat)

    #                 we only have 1 attack, this method is copied from some code studying the effect of ensemble attack


    if trans is None:
        trans = np.zeros([len(x_nat), 3])
        x_nat_no_aug = x_nat

    norm_attacks = [0]
    weights = self.uniform_weights(1, len(x_nat))
    # actually for num_attacks=1, weights will just be [1,1,1,1,1...]

    x = self.norm_perturb(x_nat, y, sess, norm_attacks, weights)


    # no_op = np.zeros([len(x_nat), 3])
    # f_x_dict = {self.model.x_input: x,
    #             self.model.y_input: y,
    #             self.model.is_training: False,
    #             self.model.transform: no_op}
    #
    # f_x = sess.run(self.model.predictions, feed_dict=f_x_dict)
    #
    # for i in range(self.num_steps):
    #     curr_dict = {self.model.x_input: x,
    #                  self.model.y_input: f_x,
    #                  self.model.transform: trans,
    #                  self.model.is_training: False}
    #     grad = sess.run(self.grad, feed_dict=curr_dict)
    #
    #     x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')
    #
    #     x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
    #     x = np.clip(x, 0, 255) # ensure valid pixel range

    return x, np.zeros([len(x_nat), 3])



class SpatialPGDAttack:
  def __init__(self, model, config, epsilon, step_size, num_steps, 
               attack_limits=None):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = config.random_start
    self.limits = config.spatial_limits
    # Attack parameters
    if attack_limits == None:
      self.limits = config.spatial_limits
    else:
      self.limits = attack_limits
    if config.loss_function == 'xent':
      loss = model.xent
    elif config.loss_function == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax
                                                    - 1e4 * label_mask, axis=1)
      loss = wrong_logit - correct_logit
    elif config.loss_function == 'reg_kl':
      loss = model.reg_loss
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.transform)[0]


  def perturb(self, x_nat, y, sess):
    """
    Given a set of examples (x_nat, y), returns a set of adversarial
    examples within epsilon of x_nat in l_infinity norm. An optional
    spatial perturbations can be given as (trans_x, trans_y, rot).
    """
    # This is Tsipras code
    if self.rand:
      # For random restart
      n = len(x_nat)
      t = np.stack((np.random.uniform(-l, l, n) for l in self.limits),
                  axis=1)
    else:
      t = np.zeros([len(x_nat), 3])

    lim_arr = np.array(self.limits)

    # x_input needs to contain but original exampes and adversarial ones
    # i.e. first transformations are noop, then adv. t
    n = len(x_nat)
    x_in = np.concatenate((x_nat, x_nat), axis=0)
    y = np.concatenate((y, y), axis=0)
    noop = np.zeros([n, 3])
    
    for i in range(self.num_steps):
        t_in = np.concatenate((noop, t), axis=0)
        curr_dict = {self.model.x_input: x_in,
                     self.model.y_input: y,
                     self.model.is_training: False,
                     self.model.transform: t_in}
        grad = sess.run(self.grad, feed_dict=curr_dict)
        grad_adv = grad[n:2*n]
        t = np.add(t, [self.step_size] * np.sign(grad_adv), out=t, casting='unsafe')
        t = np.clip(t, -lim_arr, lim_arr)

    x = x_nat

    return x, t

  # def perturb(self, x_nat, y, sess):
  #   """
  #   Given a set of examples (x_nat, y), returns a set of adversarial
  #   examples within epsilon of x_nat in l_infinity norm. An optional
  #   spatial perturbations can be given as (trans_x, trans_y, rot).
  #   """
  #   # This is Tsipras code
  #   if self.rand:
  #     # For random restart
  #     n = len(x_nat)
  #     t = np.stack((np.random.uniform(-l, l, n) for l in self.limits), axis=1)
  #   else:
  #     t = np.zeros([len(x_nat), 3])

  #   lim_arr = np.array(self.limits)

  #   no_op = np.zeros([len(x_nat), 3])
  #   f_x_dict = {self.model.x_input: x_nat,
  #               self.model.y_input: y,
  #               self.model.is_training: False,
  #               self.model.transform: no_op}

  #   f_x = sess.run(self.model.predictions, feed_dict=f_x_dict)
  #   for i in range(self.num_steps):
  #       curr_dict = {self.model.x_input: x_nat,
  #                    self.model.y_input: f_x,
  #                    self.model.is_training: False,
  #                    self.model.transform: t}
  #       grad = sess.run(self.grad, feed_dict=curr_dict)

  #       t = np.add(t, [self.step_size] * np.sign(grad), out=t, casting='unsafe')

  #       t = np.clip(t, -lim_arr, lim_arr)

  #   x = x_nat

  #   return x, t