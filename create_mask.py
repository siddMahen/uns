import tensorflow as tf
import numpy as np
import cv2

from model import inference

import os
import sys
import glob

PATCH_WIDTH = 139
WRITE_DIR = "out"

def pad_zeros(vec, w, iaxis, kwargs):
    vec[:w[0]] = 0
    vec[-w[1]:] = 0
    return vec

def cutoff(im, thres):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            v = im[i,j]
            if v < thres:
                im[i,j] = 0
            else:
                im[i,j] = 255

def create_mask(run_name, direc, filenames):
    directory = os.path.join(WRITE_DIR, direc)
    if not os.path.exists(directory):
        os.mkdir(directory)
        print("Created directory %s" % directory)

    with tf.Graph().as_default():
        images = tf.placeholder(tf.float32, shape=(1, PATCH_WIDTH, PATCH_WIDTH, 1))
        keep_prob = tf.Variable(1.0, name='keep_prob', trainable=False)

        logits = inference(images, keep_prob)
        prob = tf.nn.softmax(logits)

        sess = tf.Session()
        saver = tf.train.Saver()

        globs = glob.glob('checkpoint/model-'+run_name+'-*')
        stripped = filter(lambda x: 'meta' not in x, globs)
        ckpt_paths = sorted(stripped, key=lambda x: int(x.split('-')[-1]), reverse=True)
        ckpt_path = ckpt_paths[0]

        print(ckpt_path)

        saver.restore(sess, ckpt_path)
        keep_prob.assign(1.0)

        for fname in filenames:
            im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            k = (PATCH_WIDTH - 1)/2

            padded_im = np.pad(im, k, pad_zeros)

            heatmap = np.zeros((21, 29))
            # try only looking at every fourth pixel?
            for x in range(0, 420, 20):
                for y in range(0, 580, 20):
                    patch = padded_im[x:x+PATCH_WIDTH, y:y+PATCH_WIDTH].reshape((PATCH_WIDTH,
                        PATCH_WIDTH, 1))
                    p = sess.run(prob, feed_dict={ images: [patch] })
                    heatmap[x/20, y/20] = p[0][1]

            heatmap = 255*heatmap

            #upsample image twice
            u1 = cv2.pyrUp(heatmap)
            u2 = cv2.pyrUp(u1)
            cutoff(u2, 128)

            blur = cv2.GaussianBlur(u2, (5,5), 0)
            resize = cv2.resize(blur, (580, 420))
            cutoff(resize, 128)

            final = resize.astype("uint8")

            path = os.path.splitext(fname)[0]
            name = os.path.basename(path)
            write_path = os.path.join(directory, "%s_mask.tif" % name)
            print("Mask saved to %s" % write_path)

            cv2.imwrite(write_path, final)

        sess.close()

if __name__ == '__main__':
    run_name = sys.argv[1]
    direc = sys.argv[2]
    filenames = sys.argv[3:]
    path = create_mask(run_name, direc, filenames)
    # example usage: python create_mask.py alpha alpha-masks 1_1.tiff
