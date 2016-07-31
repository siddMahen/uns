import tensorflow as tf
import numpy as np

from pipeline import inputs
from model import inference

from PIL import Image

import glob
import sys
import os

SUMMARY_DIR = 'log'
PATCH_WIDTH = 139

def prediction(logits):
    with tf.name_scope('prediction'):
        y = tf.nn.softmax(logits)
        pred = tf.argmax(y, 1)
        tf.scalar_summary('density', tf.nn.zero_fraction(pred))
        return pred

def batch_training_error(logits, labels):
    with tf.name_scope('training_error'):
        correct_predictions = tf.equal(prediction(logits), tf.cast(labels, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        tf.scalar_summary('accuracy', accuracy)
        return 1.0 - accuracy

def pad_zeros(vec, w, iaxis, kwargs):
    vec[:w[0]] = 0
    vec[-w[1]:] = 0
    return vec

def gen_mask(run_name):
    image = np.array(Image.open('expr_train/1_1.tif'))
    k = (PATCH_WIDTH - 1)/2

    padded_image = np.pad(image, k, pad_zeros)
    padded_image = np.reshape(padded_image, (420 + 2*k,580 + 2*k,1))

    with tf.Graph().as_default():
        images = tf.placeholder(tf.float32, shape=(1, 420 + 2*k, 580 + 2*k, 1))

        keep_prob = tf.Variable(1.0, name='keep_prob', trainable=False)

        logits = inference(images, keep_prob)
        pred = prediction(logits)

        summary_op = tf.merge_all_summaries()

        sess = tf.Session()

        log_dir = os.path.join(SUMMARY_DIR, "gen2")
        writer = tf.train.SummaryWriter(log_dir, sess.graph)

        saver = tf.train.Saver()

        globs = glob.glob('checkpoint/model-'+run_name+'-*')
        print(globs)
        stripped = filter(lambda x: 'meta' not in x, globs)
        ckpt_paths = sorted(stripped, key=lambda x: int(x.split('-')[-1]), reverse=True)
        ckpt_path = ckpt_paths[0]

        print(ckpt_path)

        saver.restore(sess, ckpt_path)

        keep_prob.assign(1.0)

        tf.train.start_queue_runners(sess=sess)

        try:
            p = sess.run(pred, feed_dict={ images: [padded_image] })
            ar = np.array(p)
            print(ar)
            print(np.shape(ar))
            ar = np.reshape(255*ar, (420, 580))
            im = Image.fromarray(ar, mode='L').convert('1')
            im.save('1_1_mask_gen.jpg')
            summary = sess.run(summary_op)
            writer.add_summary(summary, global_step=step)
            print('Summary written.')
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (NUM_EPOCHS, step))

        sess.close()

# so num_epochs needs t be None for this to work....
def simple_eval():
    with tf.Graph().as_default():
        var = tf.Variable(1.0, tf.float32)
        images, labels = inputs(train=False, batch_size=1024, num_epochs=None)

        sess = tf.Session()
        saver = tf.train.Saver()

        saver.restore(sess, 'checkpoint/test')
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        l = sess.run(labels)
        print(l)


def evaluate_model(run_name, filenames):
    with tf.Graph().as_default():
        images, labels = inputs(filenames, batch_size=128,
                num_epochs=1, train=False)

        keep_prob = tf.Variable(1.0, name='keep_prob', trainable=False)

        logits = inference(images, keep_prob)
        training_error = batch_training_error(logits, labels)
        summary_op = tf.merge_all_summaries()

        local_init_op = tf.initialize_local_variables()

        sess = tf.Session()

        log_dir = os.path.join(SUMMARY_DIR, run_name)
        writer = tf.train.SummaryWriter(log_dir, sess.graph)

        saver = tf.train.Saver()

        # find path names (w/o .meta endings) to load
        # TODO: this is a cheap hack; fix using regex
        paths = glob.glob('checkpoint/model-'+run_name+'-*')
        no_meta_paths = filter(lambda x: x.find(".meta") == -1, paths)

        ckpt_paths = sorted(no_meta_paths,
                key=lambda x: int(x.split('-')[-1]), reverse=True)

        ckpt_path = ckpt_paths[0]
        print(ckpt_path)

        saver.restore(sess, ckpt_path)
        keep_prob.assign(1.0)

        sess.run(local_init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        results = []

        try:
            step = 0
            while not coord.should_stop():
                err = sess.run(training_error)
                print("Step %d, batch training error: %.3f" % (step, err))
                results.append(err)

                if step % 10 == 0:
                    summary = sess.run(summary_op)
                    writer.add_summary(summary, global_step=step)
                    print('Summary written.')

                step += 1
        except tf.errors.OutOfRangeError:
            print('Done evaluation for %d steps.' % step)
            print('Total training error: %.3f' % np.mean(results))
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    run_name = sys.argv[1]
    filenames = sys.argv[2:]
    evaluate_model(run_name, filenames)
    #gen_mask(run_name)

