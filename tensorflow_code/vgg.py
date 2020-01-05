# based on https://github.com/tensorflow/models/tree/master/resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from spatial_transformer import transformer

# from hyper_parameters import *

# This is bad design, because it's not in the model class
# copy and pasted code from https://github.com/liuyang079/vgg-tensorflow-cifar10/blob/master/vgg.py
def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''

    ## TODO: to allow different weight decay to fully connected layer and conv layer
    # Some sort of weight decay built in
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)
    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables

# def fc_layer(input_layer, num_output, is_relu=True):
#     '''
#     full connection layer
#     :param input_layer: 2D tensor
#     :param num_output: number of output layer
#     :param is_relu: judge use activation function: relu
#     :return: output layer, 2D tensor
#     '''
#     input_dim = input_layer.get_shape().as_list()[-1]
#     fc_w = create_variables(name='fc_weights', shape=[input_dim, num_output], is_fc_layer=True,
#                             initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
#     fc_b = create_variables(name='fc_bias', shape=[num_output], initializer=tf.zeros_initializer())

#     fc_result = tf.matmul(input_layer, fc_w) + fc_b
#     if is_relu is True:
#         return tf.nn.relu(fc_result)
#     else:
#         return fc_result


def fc_bn_layer(input_layer, num_output, is_relu=True, input_dim=None):
    '''
    full connection layer
    :param input_layer: 2D tensor
    :param num_output: number of output layer
    :param is_relu: judge use activation function: relu
    :return: output layer, 2D tensor
    '''
    if input_dim == None:
      input_dim = input_layer.get_shape().as_list()[-1]

    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_output], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_output], initializer=tf.zeros_initializer())

    fc_result = tf.matmul(input_layer, fc_w) + fc_b
    fc_bn_layer = batch_fc_normalization_layer(fc_result, num_output)
    if is_relu is True:
        return tf.nn.relu(fc_bn_layer)
    else:
        return fc_bn_layer

def batch_fc_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation of full connection layer
    :param input_layer: 2D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 2D tensor
    :return: the 2D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0])
    beta = tf.get_variable('beta', dimension, tf.float32,
                           initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                            initializer=tf.constant_initializer(1.0, tf.float32))
    fc_bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return fc_bn_layer


def batch_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer

# def conv_relu_layer(input_layer, filter_shape, stride):
#     '''
#     A helper function to conv and relu the input tensor sequentially
#     :param input_layer: 4D tensor
#     :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
#     :param stride: stride size for conv
#     :return: 4D tensor. Y = Relu(conv(X))
#     '''
#     filter = create_variables(name='conv_relu', shape=filter_shape)
#     conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
#     output = tf.nn.relu(conv_layer)

#     return output


def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv_bn_relu', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output


# def bn_relu_conv_layer(input_layer, filter_shape, stride):
#     '''
#     A helper function to batch normalize, relu and conv the input layer sequentially
#     :param input_layer: 4D tensor
#     :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
#     :param stride: stride size for conv
#     :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
#     '''

#     in_channel = input_layer.get_shape().as_list()[-1]

#     bn_layer = batch_normalization_layer(input_layer, in_channel)
#     relu_layer = tf.nn.relu(bn_layer)

#     filter = create_variables(name='bn_relu_conv', shape=filter_shape)
#     conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
#     return conv_layer

BN_EPSILON = 0.001

class Model(object):
  """ResNet model."""

  def __init__(self, config, num_ids, differentiable, adversarial_ce=False, nat_ce=False,
               reg_type="kl", cce_adv_exp_wrt="cce", reg_adv_exp_wrt="kl"):
    """ResNet constructor.
    """
    self._build_model(config, config.filters, num_ids, differentiable,
                      adversarial_ce, nat_ce, reg_type,
                      cce_adv_exp_wrt, reg_adv_exp_wrt,
                      pad_mode=config.pad_mode,
                      pad_size=config.pad_size)

  def add_internal_summaries(self):
    pass

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self, config, filters, num_ids, differentiable=False,
                   adversarial_ce=False, nat_ce=False, reg_type="l2",
                   cce_adv_exp_wrt="cce", reg_adv_exp_wrt="kl",
                   pad_mode='CONSTANT',
                   pad_size=32, reuse=False):
    """Build the core model within the graph."""
    with tf.variable_scope('input'):

      self.group = tf.placeholder(tf.int32, [None], name="group")
      self.num_ids = num_ids

      self.x_input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
      self.y_input = tf.placeholder(tf.int64, shape=None)

      self.transform = tf.placeholder(tf.float32, shape=[None, 3])
      trans_x, trans_y, rot = tf.unstack(self.transform, axis=1)
      rot *= np.pi / 180 # convert degrees to radians

      self.is_training = tf.placeholder(tf.bool)

      x = self.x_input
      x = tf.pad(x, [[0,0], [16,16], [16,16], [0,0]], pad_mode)

      if not differentiable:
          # For non PGD attacks: rotate and translate image
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
          x = transformer(x, theta, (64,64))
      x = tf.image.resize_image_with_crop_or_pad(x, pad_size, pad_size)

      # everything below this point is generic (independent of spatial attacks)
      self.x_image = x
      x = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x)

    # strides = [1, 2, 2]

    # Don't need activation_summary and layers.append....
    layers = []
    # block1
    with tf.variable_scope('conv1_1', reuse=reuse):
        conv1_1 = conv_bn_relu_layer(x, [3, 3, 3, 64], 1)
        activation_summary(conv1_1)
        layers.append(conv1_1)
    with tf.variable_scope('conv1_2', reuse=reuse):
        conv1_2 = conv_bn_relu_layer(conv1_1, [3, 3, 64, 64], 1)
        activation_summary(conv1_2)
        layers.append(conv1_2)
    with tf.name_scope('conv1_max_pool'):
        conv2 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        activation_summary(conv2)
        layers.append(conv2)
    # block2
    with tf.variable_scope('conv2_1', reuse=reuse):
        conv2_1 = conv_bn_relu_layer(conv2, [3, 3, 64, 128], 1)
        activation_summary(conv2_1)
        layers.append(conv2_1)
    with tf.variable_scope('conv2_2', reuse=reuse):
        conv2_2 = conv_bn_relu_layer(conv2_1, [3, 3, 128, 128], 1)
        activation_summary(conv2_2)
        layers.append(conv2_2)
    with tf.name_scope('conv2_max_pool'):
        conv3 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        activation_summary(conv3)
        layers.append(conv3)
    # block3
    with tf.variable_scope('conv3_1', reuse=reuse):
        conv3_1 = conv_bn_relu_layer(conv3, [3, 3, 128, 256], 1)
        activation_summary(conv3_1)
        layers.append(conv3_1)
    with tf.variable_scope('conv3_2', reuse=reuse):
        conv3_2 = conv_bn_relu_layer(conv3_1, [3, 3, 256, 256], 1)
        activation_summary(conv3_2)
        layers.append(conv3_2)
    with tf.variable_scope('conv3_3', reuse=reuse):
        conv3_3 = conv_bn_relu_layer(conv3_2, [3, 3, 256, 256], 1)
        activation_summary(conv3_3)
        layers.append(conv3_3)
    # with tf.variable_scope('conv3_4', reuse=reuse):
    #     conv3_4 = conv_bn_relu_layer(conv3_3, [3, 3, 256, 256], 1)
    #     activation_summary(conv3_4)
    #     layers.append(conv3_4)
    with tf.name_scope('conv3_max_pool'):
        conv4 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        activation_summary(conv4)
        layers.append(conv4)
    # block4
    with tf.variable_scope('conv4_1', reuse=reuse):
        conv4_1 = conv_bn_relu_layer(conv4, [3, 3, 256, 512], 1)
        activation_summary(conv4_1)
        layers.append(conv4_1)
    with tf.variable_scope('conv4_2', reuse=reuse):
        conv4_2 = conv_bn_relu_layer(conv4_1, [3, 3, 512, 512], 1)
        activation_summary(conv4_2)
        layers.append(conv4_2)
    with tf.variable_scope('conv4_3', reuse=reuse):
        conv4_3 = conv_bn_relu_layer(conv4_2, [3, 3, 512, 512], 1)
        activation_summary(conv4_3)
        layers.append(conv4_3)
    # with tf.variable_scope('conv4_4', reuse=reuse):
    #     conv4_4 = conv_bn_relu_layer(conv4_3, [3, 3, 512, 512], 1)
    #     activation_summary(conv4_4)
    #     layers.append(conv4_4)
    with tf.name_scope('conv4_max_pool'):
        conv5 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        activation_summary(conv5)
        layers.append(conv5)
    # block5
    with tf.variable_scope('conv5_1', reuse=reuse):
        conv5_1 = conv_bn_relu_layer(conv5, [3, 3, 512, 512], 1)
        activation_summary(conv5_1)
        layers.append(conv5_1)
    with tf.variable_scope('conv5_2', reuse=reuse):
        conv5_2 = conv_bn_relu_layer(conv5_1, [3, 3, 512, 512], 1)
        activation_summary(conv5_2)
        layers.append(conv5_2)
    with tf.variable_scope('conv5_3', reuse=reuse):
        conv5_3 = conv_bn_relu_layer(conv5_2, [3, 3, 512, 512], 1)
        activation_summary(conv5_3)
        layers.append(conv5_3)
    # with tf.variable_scope('conv5_4', reuse=reuse):
    #     conv5_4 = conv_bn_relu_layer(conv5_3, [3, 3, 512, 512], 1)
    #     activation_summary(conv5_4)
    #     layers.append(conv5_4)
    with tf.name_scope('conv5_max_pool'):
        conv6 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        activation_summary(conv6)
        layers.append(conv6)
    # full connection layer
    fc_shape = conv6.get_shape().as_list()
    nodes = fc_shape[1]*fc_shape[2]*fc_shape[3]
    # fc_reshape = tf.reshape(conv6, (fc_shape[0], nodes), name='fc_reshape')
    # fc_reshape = tf.reshape(conv6, [-1, nodes])
    fc_reshape = tf.reshape(conv6, [tf.shape(conv6)[0], -1])
    print(nodes)
    # fc6
    with tf.variable_scope('fc6', reuse=reuse):
        fc6 = fc_bn_layer(fc_reshape, 4096, input_dim=nodes)
        activation_summary(fc6)
        layers.append(fc6)
    with tf.name_scope('dropout1'):
        fc6_drop = tf.nn.dropout(fc6, 0.5)
        activation_summary(fc6_drop)
        layers.append(fc6_drop)
    # fc7
    with tf.variable_scope('fc7', reuse=reuse):
        fc7 = fc_bn_layer(fc6_drop, 4096)
        activation_summary(fc7)
        layers.append(fc7)
    with tf.name_scope('dropout2'):
        fc7_drop = tf.nn.dropout(fc7, 0.5)
        activation_summary(fc7_drop)
        layers.append(fc7_drop)
    # fc8, logit layer
    with tf.variable_scope('fc8', reuse=reuse):
        self.pre_softmax = fc_bn_layer(fc7_drop, config.n_classes, is_relu=False)
        # activation_summary(fc8)
        # layers.append(fc8)

    # End of VGG code

    # with tf.variable_scope('logit'):
    #   self.pre_softmax = self._fully_connected(x, config.n_classes)

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
            if cce_adv_exp_wrt != reg_adv_exp_wrt:  # the batch is 3b
                self.nat_ex_presm, self.adv_ex_reg_presm, self.adv_ex_cce_presm = tf.split(
                    self.pre_softmax, 3, axis=0)
                self.nat_ex_y, self.adv_ex_reg_y, self.adv_ex_cce_y = tf.split(
                    self.y_input, 3, axis=0)
                self.y_xent_for_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.adv_ex_cce_presm, labels=self.adv_ex_cce_y)

            else:  # cce uses the same adversarial examples as regularizer : batchsize 2b
                self.nat_ex_presm, self.adv_ex_reg_presm = tf.split(self.pre_softmax, 2, axis=0)
                self.nat_ex_y, self.adv_ex_reg_y = tf.split(self.y_input, 2, axis=0)
                self.y_xent_for_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.adv_ex_reg_presm, labels=self.adv_ex_reg_y)

            self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.pre_softmax, labels=self.y_input)

        else:  # mixed batch
            self.nat_ex_presm, self.adv_ex_reg_presm = tf.split(self.pre_softmax, 2, axis=0)
            self.nat_ex_y, self.adv_ex_reg_y = tf.split(self.y_input, 2, axis=0)
            self.y_xent_for_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.pre_softmax, labels=self.y_input)
            self.y_xent = self.y_xent_for_loss

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

  def _l2_defense(self):
      # assuming first num_ids are natural examples; second num_ids
      # are adversarially transformed ones
      natural_examples = tf.gather(self.pre_softmax,
                                   tf.cast(tf.range(tf.shape(self.pre_softmax)[0] - self.num_ids), tf.int32))
      adversarial_examples = tf.gather(self.pre_softmax,
                                       tf.cast(tf.range(self.num_ids, tf.shape(self.pre_softmax)[0]), tf.int32))
      group_vars = tf.reduce_sum(
          tf.square(natural_examples - adversarial_examples), axis=1)
      return group_vars

  def _l2_2tensors(self):
      group_vars = tf.reduce_sum(
          tf.square(self.nat_ex_presm - self.adv_ex_reg_presm), axis=1)
      # we divide by 4 to match the variance computation from above
      # a factor of 1/2 is required anyways; the second factor of 1/2
      # corresponds to the biased estimate of the variance
      # which is also used in tf.nn.moments*()
      countfact_loss = tf.reduce_mean(group_vars) / 4.
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

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find('fc_weights') + var.op.name.find('conv_bn_relu') >= 0:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)
