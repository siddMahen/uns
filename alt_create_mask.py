import tensorflow as tf
import numpy as np
import cv2

from model import inference
from eval import prediction

import os
import sys
import glob

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
        images = tf.placeholder(tf.float32, shape=(1, 420, 580, 1))
        keep_prob = tf.Variable(1.0, name='keep_prob', trainable=False)

        logits = inference(images, keep_prob, 1)
        flat_logits = tf.reshape(logits, [-1,2])
        prob = tf.nn.softmax(flat_logits)
        pred = prediction(flat_logits)

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
            im = im.reshape((420, 580, 1))
            hm, res = sess.run([prob, pred], feed_dict={ images: [im] })

            hm = np.array(map(lambda x: x[1],
                hm.tolist())).reshape((420,580))

            res = res.reshape((420, 580)).astype("uint8")

            mask = 255*(res)
            heatmap = 255*hm

            print(heatmap)

            cv2.imwrite("deconv_mask.tif", mask)
            cv2.imwrite("deconv_hm.tif", heatmap)
            sys.exit()

            res = 255*(res.reshape(27, 37).astype("uint8"))
            heatmap = 255*(np.array(map(lambda x: x[1],
                p.tolist())).reshape((27,37)))


            res = cv2.medianBlur(res, 3)
            res = cv2.resize(res, (580 + 2*k, 420 + 2*k), cv2.INTER_NEAREST)
            res = res[k:k + 420, k:k + 580]
            #res = cv2.medianBlur(res, (3,3), 0)
            cutoff(res, 240)
            cv2.imwrite("res.tif", res.astype("uint8"))

            #cutoff(heatmap, thresh)
            # changing the kernel size and the interpolation method has enourmous
            # consequences: NEAREST_NEIGHBOR is much more dependable, and has
            # a much "smoother" averaging effect.
            blur = cv2.GaussianBlur(heatmap, (7,7), 0)
            re = cv2.resize(blur, (580 + 2*k, 420 + 2*k), cv2.INTER_NEAREST)
            cutoff(re, 128)

            #re = cv2.resize(heatmap, (580 + 2*k, 420 + 2*k), cv2.INTER_CUBIC)
            #cutoff(re, 128)
            #blur = cv2.GaussianBlur(re, (3,3), 0)
            #cutoff(blur, 128)

            crop = re[k:k + 420, k:k + 580]
            final = crop.astype("uint8")

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