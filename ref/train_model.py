import tensorflow as tf
import numpy as np
import os
import cifar10_input
import pickle
from model_vgg16 import Vggmodel

MAX_STEP = 40000
BATCH_SIZE = 32


def restore(sess, timestamp):
    ckpt = tf.train.get_checkpoint_state('./tmp/'+str(timestamp))
    if ckpt and ckpt.model_checkpoint_path:
        with open(os.path.join('./log', str(timestamp), 'params.pkl'), 'rb') as f:
            params = pickle.load(f)
            print("Restore params : {}".format(params))
            vgg = Vggmodel(**params)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Checkpoint Path :", ckpt.model_checkpoint_path)
            return vgg, ckpt.model_checkpoint_path
    return None, None


def train(vgg, timestamp=None):
    '''

    :param timestamp: Timestamp is used as a label to differentiate each experiment.

                      If None or there isn't any folder named the timestamp,
                        it will make new folder named current timestamp to save logs and models.
                        (e.g. logs/1506496800/train)
                      Otherwise(folder exists), it restore the model from the folder.

    '''

    log_dirs = './log/'+str(timestamp)
    tmp_dirs = './tmp/'+str(timestamp)
    if not os.path.exists(log_dirs):
        os.makedirs(log_dirs+'/train')
        os.makedirs(log_dirs+'/test')
        os.makedirs(tmp_dirs)

        for dir in [log_dirs, tmp_dirs]:
            with open(os.path.join(dir, 'params.pkl'), 'wb') as f:  # Save hyperparameters of vgg models to retrieve
                pickle.dump(vgg.params, f)
            with open(os.path.join(dir, "spec.txt"), "w") as f:  # For logging
                f.write(vgg.desc)


    summary_op = tf.summary.merge_all()
    train_summary_writer = tf.summary.FileWriter(log_dirs+'/train')
    test_summary_writer = tf.summary.FileWriter(log_dirs+'/test')

    with tf.Session() as sess:
        train_batch_xs, train_batch_ys = cifar10_input.read_cifar10(is_train=True, batch_size=BATCH_SIZE, shuffle=True)
        test_batch_xs, test_batch_ys = cifar10_input.read_cifar10(is_train=False, batch_size=BATCH_SIZE*4, shuffle=False)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        vgg.load_weight_with_skip(sess, 'vgg16_weights.npz', ['fc6', 'fc7', 'fc8'])

        for step in range(0, MAX_STEP+1):
            if coord.should_stop():
                break

            xs, ys, = sess.run([train_batch_xs, train_batch_ys])  # Multi-thread
            loss, acc, summaries, _ = vgg.update(sess, xs, ys, summary_op)
            if step % 50 == 0:
                print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, loss, acc*100))
                train_summary_writer.add_summary(summaries, step)

            if step % 100 == 0:
                xs, ys, = sess.run([test_batch_xs, test_batch_ys])
                test_loss, test_acc, test_summaries = vgg.validate(sess, xs, ys, summary_op)
                test_result = 'Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, test_loss, test_acc*100)
                print ('[Test]'+test_result)
                if step % 5000 == 0:
                    with open(os.path.join(log_dirs, "result_{}.txt".format(int(test_acc*100))), "w") as f:
                        f.write(test_result)
                test_summary_writer.add_summary(test_summaries, step)

                if step != 0:
                    if not os.path.exists('./tmp'):
                        os.makedirs('./tmp')
                    saver_path = saver.save(sess, os.path.join('./tmp', str(timestamp), 'vgg.skpt'), global_step=step)
                    print("Model saved in file: %s" % saver_path)


        coord.request_stop()
        coord.join(threads)


def evaluate(timestamp, batch_size=BATCH_SIZE*64, n_batch=10):
    '''

    :param timestamp: It will retrieve model from the folder named the timestamp.
    :param n_batch: The number of batches to evaluate model.
    :return:
    '''

    with tf.Session() as sess:
        vgg, ckpt_path = restore(sess, timestamp)
        if ckpt_path is None:
            print("Checkpoint not found")
            return

        test_batch_xs, test_batch_ys = cifar10_input.read_cifar10(is_train=False, batch_size=batch_size, shuffle=False)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        accs = np.zeros(n_batch)
        for step in range(n_batch):
            test_xs, test_ys = sess.run([test_batch_xs, test_batch_ys])
            acc = sess.run(vgg.accuracy, feed_dict={vgg.xs: test_xs, vgg.ys: test_ys})
            print("Accuracy : %.4f%%" % (acc*100))
            accs[step] = acc
        coord.request_stop()
        coord.join(threads)
        return np.mean(accs)
