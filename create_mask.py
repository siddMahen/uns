import tensorflow as tf
import numpy as np
from PIL import Image

from model import inference

import sys
import glob

PATCH_WIDTH = 139

def pad_zeros(vec, w, iaxis, kwargs):
    vec[:w[0]] = 0
    vec[-w[1]:] = 0
    return vec

def create_mask(run_name, filename):
    im = np.array(Image.open(filename))
    k = (PATCH_WIDTH - 1)/2
    padded_im = np.pad(im, k, pad_zeros)

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

        #heatmap = np.zeros((84, 116))
        heatmap = np.zeros((21, 29))

        # try only looking at every fourth pixel?
        for x in range(0, 420, 20):
            for y in range(0, 580, 20):
                patch = padded_im[x:x+PATCH_WIDTH, y:y+PATCH_WIDTH].reshape((PATCH_WIDTH,
                    PATCH_WIDTH, 1))
                p = sess.run(prob, feed_dict={ images: [patch] })
                heatmap[x/20, y/20] = p[0][1]

        # could be (12,11) below
        #heatmap = np.pad(255*heatmap, ((8,8), (11,12)), pad_zeros)
        heatmap = 255*heatmap
        print(heatmap.shape)

        im = Image.fromarray(heatmap.astype("uint8"))
        im.save("image_mask.tif")

        sess.close()

if __name__ == '__main__':
    run_name = sys.argv[1]
    filename = sys.argv[2]
    path = create_mask(run_name, filename)
    print("Mask saved")
    # example usage: python create_mask.py alpha 1_1.tiff
