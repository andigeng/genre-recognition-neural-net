import numpy as np
import tensorflow as tf


def weight_variable(shape):
  """ Weight variables for connections. """
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """ Bias variable for nodes. """
  initial = tf.random_uniform(shape=shape, minval=0.05, maxval=0.15)
  return tf.Variable(initial)


def conv2d(input, filter_shape):
  """ """
  conv_weights = weight_variable(filter_shape)
  conv = tf.nn.conv2d(input, conv_weights, strides=[1,1,1,1], padding='SAME')
  return conv


def fc_layer(input, shape):
  """ """
  weights = weight_variable(shape)
  bias = bias_variable([shape[1]])

  result = tf.matmul(input, weights) + bias
  return result


def conv_layer(input, filter_shape):
  """ """
  conv = conv2d(input, filter_shape)
  conv_bias = bias_variable([filter_shape[3]])
  conv_with_bias = tf.nn.bias_add(conv, conv_bias)
  conv_with_activation = tf.nn.relu(conv_with_bias)

  return conv_with_activation


def pool_layer(conv_layer):
  """ """
  return tf.nn.max_pool(conv_layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')