import tensorflow as tf
import numpy as np
from PIL import Image

from pipeline import inputs
from model import inference

import sys
import os

CKPT_DIR = 'checkpoint'
SUMMARY_DIR = 'log'

BATCH_SIZE = 128
NUM_EPOCHS = 1

def loss(logits, labels):
    with tf.name_scope('sparse_cross_entropy'):
        sft_max = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
        with tf.name_scope('total'):
            total = tf.reduce_sum(sft_max)
        with tf.name_scope('normalized'):
            normalized = tf.reduce_mean(sft_max)
        tf.scalar_summary('cross_entropy', normalized)
    return normalized

def train(loss):
    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
        return train_op

def run_training(run_name, filenames):
    with tf.Graph().as_default():
        images, labels = inputs(filenames, batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS, train=True)

        keep_prob = tf.Variable(0.5, name='keep_prob', trainable=False)

        logits = inference(images, keep_prob)
        loss_op = loss(logits, labels)
        train_op = train(loss_op)

        init_op = tf.initialize_all_variables()
        local_init_op = tf.initialize_local_variables()
        summary_op = tf.merge_all_summaries()

        sess = tf.Session()
        saver = tf.train.Saver()

        # check to see if there's anything saved under this run_name; if so
        # load it up

        prev_step = 0
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)

        if ckpt and ckpt.model_checkpoint_path:
            # Check if the run name matches ours
            ending = ckpt.model_checkpoint_path.split('/')[-1].split('-')
            alt_name = ending[1]

            if alt_name == run_name:
                prev_step = int(ending[2])
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(init_op)
        else:
            sess.run(init_op)

        sess.run(local_init_op)

        coord = tf.train.Coordinator()

        log_dir = os.path.join(SUMMARY_DIR, run_name)
        ckpt_path = os.path.join(CKPT_DIR, "model-" + run_name)

        writer = tf.train.SummaryWriter(log_dir, sess.graph)

        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        for v in tf.all_variables():
            if 'epoch' in v.name:
                print(v.name)
                print(v.eval(sess))
                v.assign(1)

        try:
            step = prev_step
            while not coord.should_stop():
                _, loss_val = sess.run([train_op, loss_op])
                print("Step %d, loss: %.3f" % (step, loss_val))

                if step % 10 == 0:
                    summary = sess.run(summary_op)
                    writer.add_summary(summary, global_step=step)
                    print('Summary written.')
                    save_path = saver.save(sess, ckpt_path, global_step=step)
                    print('Model saved to %s' % save_path)

                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (NUM_EPOCHS, step))
            # save the model before we exit
            save_path = saver.save(sess, ckpt_path, global_step=step)
            print('Model saved to %s' % save_path)
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    run_name = sys.argv[1]
    filenames = sys.argv[2:]
    run_training(run_name, filenames)
    # example usage: python training.py test5 data/TRAIN.tfrecords

