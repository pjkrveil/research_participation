import tensorflow as tf
import math
from tensorflow.contrib.layers.python.layers import batch_norm

def conv_layer(input, n_in, n_out, k=3, s=1, scope='conv', trainable=True):
    with tf.variable_scope(scope) as scope:
        # Input Batch Normalization
        batch_norm(input)

        init_std = math.sqrt(2.0/(k*k*n_in))
        kernel = tf.get_variable('weights', shape=[k, k, n_in, n_out], dtype=tf.float32, trainable=trainable,
                                 initializer=tf.truncated_normal_initializer(0.0, init_std))
        conv = tf.nn.conv2d(input, kernel, strides=[1, 1, s, s], padding='SAME')
        biases = tf.get_variable('biases', shape=[n_out], dtype=tf.float32, trainable=trainable,
                                 initializer=tf.constant_initializer(0.0))
        tf.get_collection('parameters')
        tf.add_to_collection('parameters', kernel)
        tf.add_to_collection('parameters', biases)
        add_bias = tf.nn.bias_add(conv, biases)
        conv_out = tf.nn.relu(add_bias)

    return conv_out


def pool(input, k=2, s=2, scope='pool'):
    with tf.variable_scope(scope) as scope:
        pool = tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME')
    return pool


def fc_layer(input, n_in, n_out, keep_prob, scope='fc', trainable=True, dropout=False):
    with tf.variable_scope(scope) as scope:
        init_std = math.sqrt(1.0/n_out)
        weights = tf.get_variable('weights', shape=[n_in, n_out], dtype=tf.float32, trainable=trainable,
                        initializer=tf.truncated_normal_initializer(0.0, init_std))
        biases = tf.get_variable('biases', shape=[n_out], dtype=tf.float32, trainable=trainable,
                                 initializer=tf.constant_initializer(0.0))
        tf.get_collection('new_parameters')
        tf.add_to_collection('new_parameters', weights)
        tf.add_to_collection('new_parameters', biases)
        fc_out = tf.nn.bias_add(tf.matmul(input, weights), biases)
        if dropout:
            print("Dropout ", scope)
            fc_out = tf.nn.dropout(fc_out, keep_prob=keep_prob)
    return fc_out


def train(loss, optimizer, conv_lr, fc_lr):
    conv_params = tf.get_collection('parameters')
    fc_params = tf.get_collection('new_parameters')
    opt_conv = optimizer(conv_lr).minimize(loss, var_list=conv_params)
    opt_fc = optimizer(fc_lr).minimize(loss, var_list=fc_params)
    train = tf.group(opt_conv, opt_fc)
    return train


def loss(logits, labels, one_hot=False, scope='loss'):
    '''

    :param logits: [batch_size, n_classes]
    :param labels: one-hot labels or 1-dim labels from 0 to n_classes-1 depending on param one_hot

    :return: reduce mean of loss
    '''
    with tf.name_scope(scope) as sc:
        if one_hot:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        else:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', loss)
        return loss


def accuracy(logits, labels, one_hot=False, scope='accuracy'):
    '''

    :param logits: [batch_size, n_classes]
    :param labels: one-hot labels or 1-dim labels from 0 to n_classes-1 depending on param one_hot

    :return: average accuracy (tf.float32)
    '''
    with tf.name_scope(scope) as sc:
        if one_hot:
            correct = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
        else:
            correct = tf.equal(tf.argmax(logits, axis=1), tf.cast(labels, tf.int64))
        accu = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', accu)
        return accu