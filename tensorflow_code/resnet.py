# based on https://github.com/tensorflow/models/tree/master/resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from spatial_transformer import transformer


class Model(object):
  """ResNet model."""

  def __init__(self, config, num_ids, differentiable, adversarial_ce=False, nat_ce=False,
               reg_type="kl", cce_adv_exp_wrt="cce", reg_adv_exp_wrt="kl"):
    """ResNet constructor.
    """
    self._build_model(config, config.filters, num_ids, differentiable,
                      adversarial_ce, nat_ce, reg_type, cce_adv_exp_wrt, reg_adv_exp_wrt,
                      pad_mode=config.pad_mode,
                      pad_size=config.pad_size,
                      )

  def add_internal_summaries(self):
    pass

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self, config, filters, num_ids, differentiable=False,
                   adversarial_ce=False, nat_ce=False, reg_type="l2",
                   cce_adv_exp_wrt="cce", reg_adv_exp_wrt="kl",
                   pad_mode='CONSTANT',
                   pad_size=32):
    """Build the core model within the graph."""
    with tf.variable_scope('input'):

      self.group = tf.placeholder(tf.int32, [None], name="group")
      self.num_ids = num_ids

      self.x_input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
      self.y_input = tf.placeholder(tf.int64, shape=None)

      self.transform = tf.placeholder(tf.float32, shape=[None, 3])
      trans_x, trans_y, rot = tf.unstack(self.transform, axis=1)
      rot *= np.pi / 180  # convert degrees to radians

      self.is_training = tf.placeholder(tf.bool)

      x = self.x_input
      x = tf.pad(x, [[0,0], [16,16], [16,16], [0,0]], pad_mode)

      if not differentiable:
        # For spatial non-PGD attacks: rotate and translate image
        ones = tf.ones(shape=tf.shape(trans_x))
        zeros = tf.zeros(shape=tf.shape(trans_x))
        trans = tf.stack([ones,  zeros, -trans_x,
                          zeros, ones,  -trans_y,
                          zeros, zeros], axis=1)
        x = tf.contrib.image.rotate(x, rot, interpolation='BILINEAR')
        x = tf.contrib.image.transform(x, trans, interpolation='BILINEAR')
      else:
        # for spatial PGD attacks need to use diffble transformer
        theta = tf.stack([tf.cos(rot), -tf.sin(rot), trans_x/64,
                          tf.sin(rot),  tf.cos(rot), trans_y/64], axis=1)
        x = transformer(x, theta, (64, 64))
      x = tf.image.resize_image_with_crop_or_pad(x, pad_size, pad_size)

      # everything below this point is generic (independent of spatial attacks)
      self.x_image = x
      x = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x)

      x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))

    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    res_func = self._residual

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in range(1, config.resnet_depth_n):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in range(1, config.resnet_depth_n):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in range(1, config.resnet_depth_n):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, 0.1)
      x = self._global_avg_pool(x)

    # uncomment to add and extra fc layer
    #with tf.variable_scope('unit_fc'):
    #  self.pre_softmax = self._fully_connected(x, 1024)
    #  x = self._relu(x, 0.1)

    with tf.variable_scope('logit'):
      self.pre_softmax = self._fully_connected(x, config.n_classes)

    self.predictions = tf.argmax(self.pre_softmax, 1)
    self.correct_prediction = tf.equal(self.predictions, self.y_input)
    self.num_correct = tf.reduce_sum(
        tf.cast(self.correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(
        tf.cast(self.correct_prediction, tf.float32))

    # assuming first num_ids are natural examples; second num_ids
    # are adversarially transformed ones
    with tf.variable_scope('costs'):
      if nat_ce:
          self.nat_ex_presm, self.adv_ex_reg_presm = tf.split(self.pre_softmax, 2, axis=0)
          self.nat_ex_y, self.adv_ex_reg_y = tf.split(self.y_input, 2, axis=0)
          self.y_xent_for_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=self.nat_ex_presm, labels=self.nat_ex_y)
          self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.pre_softmax, labels=self.y_input)

      elif adversarial_ce:
        if config.use_reg & (cce_adv_exp_wrt != reg_adv_exp_wrt):  # the batch is 3 folds
          self.nat_ex_presm, self.adv_ex_reg_presm, self.adv_ex_cce_presm = tf.split(
                    self.pre_softmax, 3, axis=0)
          self.nat_ex_y, self.adv_ex_reg_y, self.adv_ex_cce_y = tf.split(
                    self.y_input, 3, axis=0)
          self.y_xent_for_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.adv_ex_cce_presm, labels=self.adv_ex_cce_y)

        else:   #cce uses the same adversarial examples as regularizer : batchsize 2b
          self.nat_ex_presm, self.adv_ex_reg_presm = tf.split(self.pre_softmax, 2, axis=0)
          self.nat_ex_y, self.adv_ex_reg_y = tf.split(self.y_input, 2, axis=0)
          self.y_xent_for_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.adv_ex_reg_presm, labels=self.adv_ex_reg_y)

        self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
           logits=self.pre_softmax, labels=self.y_input)

      else:  # mixed batch
        if config.use_reg & (cce_adv_exp_wrt != reg_adv_exp_wrt):  # the batch is 3 folds
          self.nat_ex_presm, self.adv_ex_reg_presm, self.adv_ex_cce_presm = tf.split(
                    self.pre_softmax, 3, axis=0)
          self.nat_ex_y, self.adv_ex_reg_y, self.adv_ex_cce_y = tf.split(
                    self.y_input, 3, axis=0)

          self.mix_ex_presm = tf.concat((self.nat_ex_presm,self.adv_ex_cce_presm), axis=0)
          self.mix_ex_y = tf.concat((self.nat_ex_y, self.adv_ex_cce_y), axis=0)

          self.y_xent_for_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.mix_ex_presm, labels=self.mix_ex_y)

        else:
          self.nat_ex_presm, self.adv_ex_reg_presm = tf.split(self.pre_softmax, 2, axis=0)
          self.nat_ex_y, self.adv_ex_reg_y = tf.split(self.y_input, 2, axis=0)
          self.y_xent_for_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.pre_softmax, labels=self.y_input)

        self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.pre_softmax, labels=self.y_input)

      self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
      # the cross entropy part of the loss
      self.mean_xent = tf.reduce_mean(self.y_xent_for_loss)
      self.weight_decay_loss = self._decay()

      # the loss (regularizer part) for gradient update
      if config.use_reg:
          if reg_type == "l2":
            self.reg_loss = self._l2_2tensors()
          elif reg_type == "kl":
            self.reg_loss = self._kl_2tensors()
          else:
            raise NotImplementedError

  def _l2_2tensors(self):
    group_vars = tf.reduce_sum(
      tf.square(self.nat_ex_presm-self.adv_ex_reg_presm), axis=1)
    # we divide by 4 to match the variance computation from above
    # a factor of 1/2 is required anyways; the second factor of 1/2
    # corresponds to the biased estimate of the variance
    # which is also used in tf.nn.moments*()
    countfact_loss = tf.reduce_mean(group_vars)/4.
    return countfact_loss

  def _klDivLoss(self, x, y):
      X = tf.distributions.Categorical(probs=x)
      Y = tf.distributions.Categorical(probs=y)
      return tf.distributions.kl_divergence(X, Y)

  def _kl_2tensors(self):
      epsilon = tf.fill(tf.shape(self.adv_ex_reg_presm), 1e-08)
      prob_a = tf.nn.softmax(self.adv_ex_reg_presm) + epsilon
      prob_b = tf.nn.softmax(self.nat_ex_presm) + epsilon
      loss_robust = tf.reduce_mean(self._klDivLoss(prob_b, prob_a))
      return loss_robust

  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.name_scope(name):
      return tf.contrib.layers.batch_norm(
          inputs=x,
          decay=.9,
          center=True,
          scale=True,
          activation_fn=None,
          updates_collections=None,
          is_training=self.is_training)

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, 0.1)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find('DW') >= 0:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable(
        'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])
