import tensorflow as tf
import numpy as np
import cv2

import random
import os

import argparse

PATCH_WIDTH = 139

def open_image(filename):
    return cv2.imread(filename, 0)

def pad_zeros(vec, w, iaxis, kwargs):
    vec[:w[0]] = 0
    vec[-w[1]:] = 0
    return vec

def get_patches_with_width(image, mask, width):
    k = (width - 1)/2
    padded_image = np.pad(image, k, pad_zeros)
    padded_mask = np.pad(mask, k, pad_zeros)

    patches = []
    pos = []

    for x in range(k, 420 + k):
        for y in range(k, 580 + k):
            if padded_mask[x,y] != 0:
                patch = np.zeros((width, width))
                for i in range(width):
                    for j in range(width):
                        patch[i,j] = padded_image[x + (i - k), y + (j - k)]
                pos.append((1, patch))

    #collect 100 (or as many as there are) positive samples
    ns = min(100, len(pos))
    patches = random.sample(pos, ns)

    # collect 100 random samples
    # the vast majority of these will be negative
    for l in range(100):
        x = random.randint(k, 420 + k - 1)
        y = random.randint(k, 580 + k - 1)
        patch = np.zeros((width, width))
        label = ~~bool(padded_mask[x,y])
        for i in range(width):
            for j in range(width):
                patch[i,j] = padded_image[x + (i - k), y + (j - k)]
        patches.append((label, patch))

    return patches

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def generate_deconv_dataset(train_im_dir, train_mask_dir, name, filenames):
    data_dir = 'data'

    n = 1
    for filename in filenames:
        path_name = os.path.join(data_dir, str(n) + '-' + name + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(path_name)

        im_path = os.path.join(train_im_dir, filename + '.tif')
        mask_path = os.path.join(train_mask_dir, filename + '_mask.tif')

        image = open_image(im_path)
        mask = open_image(mask_path)
        mask = mask/255

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image.astype(np.uint8).tostring()),
            'mask_raw' : _bytes_feature(mask.astype(np.uint8).tostring())
        }))

        writer.write(example.SerializeToString())
        writer.close()
        n += 1

def generate_dataset(name, filenames):
    train_dir = 'train'
    data_dir = 'data'

    n = 1
    for filename in filenames:
        path_name = os.path.join(data_dir, str(n) + '-' + name + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(path_name)

        im_path = os.path.join(train_dir, filename + '.tif')
        mask_path = os.path.join(train_dir, filename + '_mask.tif')
        image = open_image(im_path)
        mask = open_image(mask_path)
        patches = get_patches_with_width(image, mask, PATCH_WIDTH)

        for (label, patch) in patches:
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(int(label)),
                'image_raw': _bytes_feature(patch.astype(np.uint8).tostring())
            }))
            writer.write(example.SerializeToString())

        writer.close()
        n += 1

def generate_training_list():
    l = []
    for i in range(1,32):
        if (i == 3) or (i == 7) or (i == 17):
            continue
        l += ["%i_%d" % (i, d) for d in range(1,121)]

    # some training sets are missing the 120th example
    l += ["3_%d" % d for d in range(1,120)]
    l += ["7_%d" % d for d in range(1,120)]
    l += ["17_%d" % d for d in range(1,120)]

    return l

def generate_testing_list():
    l = []
    for i in range(32, 42):
        if (i == 34):
            continue
        l += ["%i_%d" % (i, d) for d in range(1,121)]

    l += ["34_%d" % d for d in range(1,120)]
    return l

def generate_validation_list():
    l = []
    for i in range(42, 48):
        if (i == 44):
            continue
        l += ["%i_%d" % (i, d) for d in range(1,121)]

    l += ["44_%d" % d for d in range(1,120)]
    return l

#l = ["1_1"]
#generate_dataset("TEST2", l)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-im', '--image_dir', required=True)
    parser.add_argument('-msk', '--mask_dir', required=True)
    parser.add_argument('-s', '--suffix', default="")
    parser.add_argument('-n', '--name', required=True)

    args = parser.parse_args()

    t_list = generate_training_list()
    l = map(lambda x: x + "-" + args.suffix, t_list)

    generate_deconv_dataset(args.image_dir, args.mask_dir, args.name, l)

#generate_dataset("p139-training", training_list)

#testing_list = generate_testing_list()
#generate_dataset("p139-testing", testing_list)

#validation_list = generate_validation_list()
#generate_dataset("p139-validation", validation_list)


