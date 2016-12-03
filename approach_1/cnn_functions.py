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
  conv_weights = weight_variable(shape)
  conv = tf.nn.conv2d(input, conv_weights, strides=[1,1,1,1], padding='SAME')
  
  return conv


def max_pool_2x2(input):
  return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def fully_connected_layer(input, shape):
  weights = weight_variable(shape)
  bias = bias_variable(shape)

  result = tf.matmul(input, weights) + bias


def conv_layer(input, filter_shape):
  conv = conv2d(input, filter_shape)
  conv_bias = bias_variable([shape[3]])
  conv_with_bias = tf.nn.bias_add(conv, conv_bias)
  conv_with_activation = tf.nn.relu(conv_w_bias)

  return conv_with_activation


def pool_layer(conv_layer):
  return max_pool_2x2(conv_layer, padding='SAME')