
import tensorflow as tf
import numpy as np
import os


def read_cifar10(is_train, batch_size, shuffle, data_dir='./cifar-10-batches-bin/'):
    """Read CIFAR10

    Args:
        data_dir: the directory of CIFAR10
        is_train: boolen
        batch_size:
        shuffle:
    Returns:
        label: 1D tensor, tf.int32
        image: 4D tensor, [batch_size, height, width, 3], tf.float32

    """
    img_width = 32
    img_height = 32
    img_depth = 3
    label_bytes = 1
    image_bytes = img_width*img_height*img_depth

    # Create queues for execution of queue runners
    if is_train:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' %ii) for ii in np.arange(1, 6)]
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]

    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.FixedLengthRecordReader(label_bytes + image_bytes)
    key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)
    label = tf.slice(record_bytes, [0], [label_bytes])
    label = tf.cast(label, tf.int32)

    image_raw = tf.slice(record_bytes, [label_bytes], [image_bytes])
    image_raw = tf.reshape(image_raw, [img_depth, img_height, img_width])
    image = tf.transpose(image_raw, (1,2,0)) # C/H/W to H/W/C
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image) # whitening the image

    if shuffle:
        images, label_batch = tf.train.shuffle_batch([image, label],
                                                     batch_size=batch_size, num_threads=64,
                                                     capacity=20000, min_after_dequeue=3000)
    else:
        images, label_batch = tf.train.batch([image, label],
                                             batch_size=batch_size, num_threads=64, capacity=2000)

    label_batch = tf.reshape(tf.cast(label_batch, dtype=tf.int32), [-1])
    return images, label_batch


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_label_names():
    meta = unpickle('cifar-10-batches-py/batches.meta')
    return meta[b'label_names']






