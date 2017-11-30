import tensorflow as tf
import numpy as np
import ops

BATCH_SIZE = 32

class Vggmodel:
    def __init__(self, lr=0.0001, optimizer=tf.train.Optimizer, fine_tuning=True, dropout=False, adaptive_ratio=1.0):
        '''

        ----------Hyperparameters -------------
        :param fine_tuning: If True, the parameters of CNN layers will also be fine-tuned.
                             Otherwise, only the parameters of FC layers will be trained.
        :param dropout: If True, dropout is applied to all fully connected layers except for the last one.
                        Also, dropout_keep_prob should be fed. (default value is 1.0)
        :param adaptive_ratio: If True, the learning rate of convolutional layer will be learning rate * adaptive_ratio
        :return:
        '''
        self.desc = "Learning rate : {}, optimizer : {}, fine_tuning : {}, dropout : {}, adaptive ratio : {}"\
            .format(lr, optimizer.__name__, fine_tuning, dropout, adaptive_ratio)
        print(self.desc)
        self.params = {'lr': lr, 'optimizer': optimizer, 'fine_tuning': fine_tuning,
                       'dropout': dropout, 'adaptive_ratio': adaptive_ratio}
        self.xs = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.ys = tf.placeholder(tf.int32, [None])
        self.dropout_keep_prob = tf.placeholder_with_default(1.0, None)

        pool5 = self.build_convnet(fine_tuning)
        fc3 = self.build_fcnet(pool5, dropout)
        self.probs = tf.nn.softmax(fc3, name='softmax')

        self.loss = ops.loss(logits=self.probs, labels=self.ys, one_hot=False)
        self.accuracy = ops.accuracy(logits=self.probs, labels=self.ys, one_hot=False)
        if adaptive_ratio < 1.0:
            self.train = ops.train(self.loss, optimizer=optimizer, conv_lr=lr*adaptive_ratio, fc_lr=lr)
        else:
            self.train = optimizer(learning_rate=lr).minimize(self.loss)


    def build_convnet(self, trainable=True):
        conv1_1 = ops.conv_layer(self.xs, 3, 64, scope='conv1_1', trainable=trainable)
        conv1_2 = ops.conv_layer(conv1_1, 64, 64, scope='conv1_2', trainable=trainable)
        pool1 = ops.pool(conv1_2, scope='pool1')

        conv2_1 = ops.conv_layer(pool1, 64, 128, scope='conv2_1', trainable=trainable)
        conv2_2 = ops.conv_layer(conv2_1, 128, 128, scope='conv2_2', trainable=trainable)
        pool2 = ops.pool(conv2_2, scope='pool2')

        conv3_1 = ops.conv_layer(pool2, 128, 256, scope='conv3_1', trainable=trainable)
        conv3_2 = ops.conv_layer(conv3_1, 256, 256, scope='conv3_2', trainable=trainable)
        conv3_3 = ops.conv_layer(conv3_2, 256, 256, scope='conv3_3', trainable=trainable)
        pool3 = ops.pool(conv3_3, scope='pool3')

        conv4_1 = ops.conv_layer(pool3, 256, 512, scope='conv4_1', trainable=trainable)
        conv4_2 = ops.conv_layer(conv4_1, 512, 512, scope='conv4_2', trainable=trainable)
        conv4_3 = ops.conv_layer(conv4_2, 512, 512, scope='conv4_3', trainable=trainable)
        pool4 = ops.pool(conv4_3, scope='pool4')

        conv5_1 = ops.conv_layer(pool4, 512, 512, scope='conv5_1', trainable=trainable)
        conv5_2 = ops.conv_layer(conv5_1, 512, 512, scope='conv5_2', trainable=trainable)
        conv5_3 = ops.conv_layer(conv5_2, 512, 512, scope='conv5_3', trainable=trainable)
        pool5 = ops.pool(conv5_3, scope='pool5') # N*7*7*512
        return pool5

    def build_fcnet(self, pool5, dropout):
        '''

        :param pool5: The tensor to pass to the fully connected layer.
        :param dropout: If True, dropout is applied to fully connected layer
        :return:
        '''
        pool_dim = int(np.prod(pool5.get_shape()[1:]))
        pool5_flatten = tf.reshape(pool5, [-1, pool_dim])

        fc1 = ops.fc_layer(pool5_flatten, pool_dim, 4096, scope='fc6', dropout=dropout, keep_prob=self.dropout_keep_prob)
        fc2 = ops.fc_layer(fc1, 4096, 4096, scope='fc7', dropout=dropout, keep_prob=self.dropout_keep_prob)
        fc3 = ops.fc_layer(fc2, 4096, 10, scope='fc8', dropout=False, keep_prob=1.0)
        return fc3

    def predict(self, sess, xs):
        return sess.run(self.probs, feed_dict={self.xs: xs})

    def validate(self, sess, xs, ys, summary_op):
        return sess.run([self.loss, self.accuracy, summary_op], feed_dict={self.xs: xs, self.ys: ys})

    def update(self, sess, xs, ys, summary_op):
        return sess.run([self.loss, self.accuracy, summary_op, self.train],
                        feed_dict={self.xs: xs, self.ys: ys, self.dropout_keep_prob: 0.5})

    def load_weight_with_skip(self, sess, weight_file, skip_layers=[]):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        parameters = tf.get_collection('parameters')+tf.get_collection('new_parameters')
        print(parameters)
        for i, key in enumerate(keys):
            if key[:-2] in skip_layers:
                print("Skip parameters: ", key, parameters[i])
                continue
            else:
                with tf.variable_scope(key[:-2], reuse=True):       # For re-assigning variables already initialized
                    if key[-1]=='W':
                        sess.run(tf.assign(tf.get_variable('weights'), weights[key]))
                    elif key[-1]=='b':
                        sess.run(tf.assign(tf.get_variable('biases'), weights[key]))
