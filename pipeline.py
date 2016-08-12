import tensorflow as tf
import numpy as np

PATCH_WIDTH=139

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, record = reader.read(filename_queue)

    features = tf.parse_single_example(
        record,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string)
    })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    mask = tf.decode_raw(features['mask_raw'], tf.uint8)

    image = tf.cast(tf.reshape(image, (420, 580, 1)), tf.float32)
    mask = tf.cast(tf.reshape(mask, (420, 580, 1)), tf.int64)

    return image, mask

def inputs(filenames, batch_size, num_epochs, train=True):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=train)

        example, label = read_and_decode(filename_queue)

        min_after_dequeue = 15
        capacity = min_after_dequeue + 3 * batch_size

        if train:
            example_batch, label_batch = tf.train.shuffle_batch(
                [example, label], batch_size=batch_size, capacity=capacity,
                min_after_dequeue=min_after_dequeue)
        else:
            example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size,
                    capacity = capacity)

        return example_batch, label_batch
