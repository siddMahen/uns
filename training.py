import tensorflow as tf
import numpy as np
from PIL import Image

from pipeline import inputs
from model import inference

import argparse
import sys
import os

CKPT_DIR = 'checkpoint'
SUMMARY_DIR = 'log'

BATCH_SIZE = 10
NUM_EPOCHS = 1

def loss(logits, labels):
    with tf.name_scope('sparse_cross_entropy'):
        flat_logits = tf.reshape(logits, [-1, 2])
        flat_labels = tf.reshape(labels, [-1])
        # use dice ratio as loss? can be implemented using simple tf fns

        sft_max = tf.nn.sparse_softmax_cross_entropy_with_logits(flat_logits, flat_labels)
        with tf.name_scope('total'):
            total = tf.reduce_sum(sft_max)
        with tf.name_scope('normalized'):
            normalized = tf.reduce_mean(sft_max)
        tf.scalar_summary('cross_entropy', normalized)

    for l in tf.get_collection("losses"):
        normalized += l

    return normalized

def train(loss):
    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(1e-5).minimize(loss)
        return train_op

def run_training_from_ckpt(run_name, filenames, is_gpu=False):
    with tf.Graph().as_default():
        images, labels = inputs(filenames, batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS, train=True)

        keep_prob = tf.Variable(0.5, name='keep_prob', trainable=False)

        logits = inference(images, keep_prob, BATCH_SIZE)
        loss_op = loss(logits, labels)
        train_op = train(loss_op)

        local_init_op = tf.initialize_local_variables()
        summary_op = tf.merge_all_summaries()

        sess = tf.Session()

        old_vars = []
        for v in tf.all_variables():
            if "deconv" not in v.name:
                old_vars.append(v)

        saver = tf.train.Saver(old_vars)
        everything_saver = tf.train.Saver()

        if is_gpu:
            saver.restore(sess, "checkpoint/model-C-4932")
        else:
            saver.restore(sess, "checkpoint/model-F-4932")

        sess.run(local_init_op)

        need_start = sess.run(tf.report_uninitialized_variables())

        need_start_vars = []
        for v in tf.all_variables():
            for n in need_start:
                if n in v.name:
                    need_start_vars.append(v)
                    continue

        init_new = tf.initialize_variables(need_start_vars, "init_new")
        sess.run(init_new)

        coord = tf.train.Coordinator()

        log_dir = os.path.join(SUMMARY_DIR, run_name)
        ckpt_path = os.path.join(CKPT_DIR, "model-" + run_name)

        writer = tf.train.SummaryWriter(log_dir, sess.graph)

        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        prev_step = 0

        try:
            step = prev_step
            while not coord.should_stop():
                _, loss_val = sess.run([train_op, loss_op])
                print("Step %d, loss: %.3f" % (step, loss_val))

                if step % 10 == 0:
                    summary = sess.run(summary_op)
                    writer.add_summary(summary, global_step=step)
                    print('Summary written.')
                    save_path = everything_saver.save(sess, ckpt_path, global_step=step)
                    print('Model saved to %s' % save_path)

                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (NUM_EPOCHS, step))
            # save the model before we exit
            save_path = everything_saver.save(sess, ckpt_path, global_step=step)
            print('Model saved to %s' % save_path)
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()

def run_training(run_name, filenames):
    with tf.Graph().as_default():
        images, labels = inputs(filenames, batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS, train=True)

        keep_prob = tf.Variable(0.5, name='keep_prob', trainable=False)

        logits = inference(images, keep_prob, BATCH_SIZE)
        loss_op = loss(logits, labels)
        train_op = train(loss_op)

        init_op = tf.initialize_all_variables()
        local_init_op = tf.initialize_local_variables()
        summary_op = tf.merge_all_summaries()

        sess = tf.Session()
        saver = tf.train.Saver()

        # check to see if there's anything saved under this run_name; if so
        # load it up

        # find path names (w/o .meta endings) to load
        # TODO: this is a cheap hack; fix using regex
        paths = glob.glob('checkpoint/model-'+run_name+'-*')
        no_meta_paths = filter(lambda x: x.find(".meta") == -1, paths)
        ckpt_paths = sorted(no_meta_paths,
                key=lambda x: int(x.split('-')[-1]), reverse=True)

        prev_step = 0

        if len(ckpt_paths) >= 1:
            ckpt_path = ckpt_paths[0]
            ending = ckpt_path.split('/')[-1].split('-')

            prev_step = int(ending[2])
            saver.restore(sess, ckpt_path)
        else:
            sess.run(init_op)

        sess.run(local_init_op)

        coord = tf.train.Coordinator()

        log_dir = os.path.join(SUMMARY_DIR, run_name)
        ckpt_path = os.path.join(CKPT_DIR, "model-" + run_name)

        writer = tf.train.SummaryWriter(log_dir, sess.graph)
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-rn', '--run_name', required=True)
    parser.add_argument('--training_data', nargs='+', required=True)
    parser.add_argument('--from_ckpt', action='store_true')
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    if args.from_ckpt:
        run_training_from_ckpt(args.run_name, args.training_data, is_gpu=args.gpu)
    else:
        run_training(args.run_name, args.training_data)

    #run_name = sys.argv[1]
    #filenames = sys.argv[2:]
    #run_training_with_old_model(run_name, filenames)
    # example usage: python training.py test5 data/TRAIN.tfrecords

