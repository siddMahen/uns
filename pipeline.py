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
            'label': tf.FixedLenFeature([], tf.int64),
    })

    label = tf.cast(features['label'], tf.int32)
    image = tf.decode_raw(features['image_raw'], tf.uint8)

    image.set_shape([PATCH_WIDTH*PATCH_WIDTH*1])
    image = tf.cast(tf.reshape(image, (PATCH_WIDTH, PATCH_WIDTH, 1)), tf.float32)

    #image = tf.image.per_image_whitening(image)

    return image, label

def inputs(filenames, batch_size, num_epochs, train=True):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=train)

        example, label = read_and_decode(filename_queue)

        min_after_dequeue = 100
        capacity = min_after_dequeue + 3 * batch_size

        if train:
            example_batch, label_batch = tf.train.shuffle_batch(
                [example, label], batch_size=batch_size, capacity=capacity,
                min_after_dequeue=min_after_dequeue)
        else:
            example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size,
                    capacity = capacity)

        return example_batch, label_batch
