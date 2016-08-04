import tensorflow as tf
import numpy as np
from PIL import Image

from model import inference
from eval import prediction

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
        pred = prediction(logits)

        sess = tf.Session()
        saver = tf.train.Saver()

        globs = glob.glob('checkpoint/model-'+run_name+'-*')
        stripped = filter(lambda x: 'meta' not in x, globs)
        ckpt_paths = sorted(stripped, key=lambda x: int(x.split('-')[-1]), reverse=True)
        ckpt_path = ckpt_paths[0]

        print(ckpt_path)

        saver.restore(sess, ckpt_path)
        keep_prob.assign(1.0)

        #result = np.zeros((420, 580))
        result = np.zeros((105, 145))

        # try only looking at every fourth pixel?
        patch = np.zeros((PATCH_WIDTH, PATCH_WIDTH))
        for x in range(k, 420 + k, 4):
            for y in range(k, 580 + k, 4):
                for i in range(PATCH_WIDTH):
                    for j in range(PATCH_WIDTH):
                        patch[i,j] = padded_im[x + (i - k), y + (j - k)]
                res = sess.run(prob, feed_dict={
                    images: [patch.reshape((PATCH_WIDTH, PATCH_WIDTH, 1))] })
                result[(x - k)/4, (y - k)/4] = res[0][1]

        result = 255*result
        im = Image.fromarray(result.astype('uint8'))
        im.save('mask-test.tif')
        #ar = np.reshape(255*ar, (420, 580))
        #im = Image.fromarray(ar, mode='L').convert('1')

        sess.close()

if __name__ == '__main__':
    run_name = sys.argv[1]
    filename = sys.argv[2]
    path = create_mask(run_name, filename)
    print("Mask saved")
    # example usage: python create_mask.py alpha 1_1.tiff
