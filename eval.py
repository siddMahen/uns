import tensorflow as tf
import numpy as np

from pipeline import inputs
from model import inference

from PIL import Image

import glob
import sys
import os

SUMMARY_DIR = 'log'

def prediction(logits):
    with tf.name_scope('prediction'):
        y = tf.nn.softmax(logits)
        pred = tf.argmax(y, 1)
        tf.scalar_summary('density', tf.nn.zero_fraction(pred))
        return pred

def batch_training_error(logits, labels):
    with tf.name_scope('training_error'):
        p = prediction(logits)
        l = labels
        inter = tf.cast(p * l, tf.float32)

        inter_size = tf.reduce_sum(inter)
        p_size = tf.cast(tf.reduce_sum(p), tf.float32)
        l_size = tf.cast(tf.reduce_sum(l), tf.float32)

        is_empty_label = tf.equal(l_size, 0)
        is_empty_pred = tf.equal(p_size, 0)

        dice_ratio = (2*inter_size)/(p_size + l_size + 0.001)

        return dice_ratio, is_empty_label, is_empty_pred

def evaluate_model(run_name, filenames):
    with tf.Graph().as_default():
        images, labels = inputs(filenames, batch_size=1,
                num_epochs=1, train=False)

        logits = inference(images, 1)
        flat_logits = tf.reshape(logits, [-1,2])
        flat_labels = tf.reshape(labels, [-1])

        training_error = batch_training_error(flat_logits, flat_labels)
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

        sess.run(local_init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        results = []
        false_neg = []
        false_pos = []
        empty = []

        try:
            step = 0
            while not coord.should_stop():
                dc, empty_l, empty_p = sess.run(training_error)

                if (empty_l == False) and (empty_p == False):
                    print("Step %d, dice ratio: %.3f - NT present and correctly identified" % (step, dc))
                    results.append(dc)
                    false_neg.append(0)
                    false_pos.append(0)
                    empty.append(0)

                if (empty_l == False) and (empty_p == True):
                    print("Step %d, dice ratio: %.3f - NT present but not identified" % (step, dc))
                    results.append(dc)
                    false_neg.append(1)
                    false_pos.append(0)
                    empty.append(0)

                if (empty_l == True) and (empty_p == False):
                    print("Step %d, dice ratio: %.3f - NT not present but identified" % (step, dc))
                    results.append(dc)
                    false_neg.append(0)
                    false_pos.append(1)
                    empty.append(0)

                if (empty_l == True) and (empty_p == True):
                    print("Step %d, dice ratio: 1.00 - NT not present and correctly identified" % step)
                    results.append(1)
                    false_neg.append(0)
                    false_pos.append(0)
                    empty.append(1)

                if step % 10 == 0:
                    summary = sess.run(summary_op)
                    writer.add_summary(summary, global_step=step)
                    print('Summary written.')

                step += 1
        except tf.errors.OutOfRangeError:
            print('Done evaluation for %d steps.' % step)
            print('No. false positives: %d' % len(false_pos))
            print('No. false negatives: %d' % len(false_neg))
            print('No. empty label/prediction pairs: %d' % len(empty))
            print('False positive rate: %.3f' % np.mean(false_pos))
            print('False negative rate: %.3f' % np.mean(false_neg))
            print('Empty label/prediction pair rate: %.3f' % np.mean(empty))
            print('Mean dice ratio: %.3f' % np.mean(results))
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    run_name = sys.argv[1]
    filenames = sys.argv[2:]
    evaluate_model(run_name, filenames)

