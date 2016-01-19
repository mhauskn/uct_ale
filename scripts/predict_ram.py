#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import tensorflow.python.platform
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import h5py
from IPython.core.debugger import Tracer

NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 1024
VALID_FRACTION = .1
TRAIN_FRACTION = .7

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('save', '/tmp/', 'path to save assets')
flags.DEFINE_string('data', '', 'input hdf5 dataset')
flags.DEFINE_integer('epochs', 500, 'number of epochs to run')
flags.DEFINE_integer('batch', 64, 'batch size')
flags.DEFINE_integer('seed', None, 'random seed')

def eval_correct(predictions, labels):
  return np.sum(np.all(np.rint(predictions) == labels, axis=1))

def avg_correct_bits(predictions, labels):
  return np.mean(np.sum(np.rint(predictions) == labels, axis=1))

def error_rate(predictions, labels):
  return 100.0 - (100.0 * eval_correct(predictions, labels) / predictions.shape[0])

def get_batch(h5f, offset):
  batch_data = h5f['screens'][offset:(offset + FLAGS.batch)].astype('float32')
  batch_labels = h5f['ram'][offset:(offset + FLAGS.batch)]
  batch_data /= 255
  batch_labels = np.unpackbits(batch_labels, axis=1).astype('float32')
  return batch_data, batch_labels

def evaluate(sess, images_pl, labels_pl, accuracy_pl, correct_pl, start_offset,
             end_offset, h5f):
  offset = start_offset
  total_acc, total_correct, batch = 0, 0, 0
  while offset + FLAGS.batch <= end_offset:
    batch_data, batch_labels = get_batch(h5f, offset)
    accuracy, correct = sess.run(
      [accuracy_pl, correct_pl],
      feed_dict={images_pl:batch_data, labels_pl:batch_labels})
    offset += FLAGS.batch
    total_acc += accuracy
    total_correct += correct
    batch += 1
  return total_acc / batch, total_correct / batch

def main(argv=None):
  h5f = h5py.File(FLAGS.data, 'r')
  print('Training Data Shape:', h5f['screens'].shape)
  n_examples, img_rows, img_cols, n_channels = h5f['screens'].shape
  train_size = int(TRAIN_FRACTION * n_examples)
  valid_size = int(VALID_FRACTION * n_examples)

  data_node = tf.placeholder(
    tf.float32, shape=(FLAGS.batch, img_rows, img_cols, n_channels), name='input_data')
  labels_node = tf.placeholder(
    tf.float32, shape=(FLAGS.batch, NUM_LABELS), name='input_labels')
  # _ = tf.image_summary('input_images', data_node)
  with tf.name_scope('conv1'):
    conv1_weights = tf.Variable(
      tf.truncated_normal([8, 8, n_channels, 32], stddev=0.1, seed=FLAGS.seed),
      name='conv1_weights')
    conv1_biases = tf.Variable(tf.zeros([32]), name='conv1_biases')
    conv = tf.nn.conv2d(data_node, conv1_weights, strides=[1, 4, 4, 1], padding='VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    _ = tf.histogram_summary('conv1_weights', conv1_weights)
    _ = tf.histogram_summary('conv1_biases', conv1_biases)
    # _ = tf.image_summary('conv1_images', conv[:,:,:,:1])
  with tf.name_scope('conv2'):
    conv2_weights = tf.Variable(
      tf.truncated_normal([4, 4, 32, 64], stddev=0.1, seed=FLAGS.seed),
      name='conv2_weights')
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]), name='conv2_biases')
    conv = tf.nn.conv2d(relu, conv2_weights, strides=[1, 2, 2, 1], padding='VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    _ = tf.histogram_summary('conv2_weights', conv2_weights)
    _ = tf.histogram_summary('conv2_biases', conv2_biases)
    # _ = tf.image_summary('conv2_images', conv[:,:,:,:1])
  with tf.name_scope('conv3'):
    conv3_weights = tf.Variable(
      tf.truncated_normal([3, 3, 64, 64], stddev=0.1, seed=FLAGS.seed),
      name='conv3_weights')
    conv3_biases = tf.Variable(tf.constant(0.1, shape=[64]), name='conv3_biases')
    conv = tf.nn.conv2d(relu, conv3_weights, strides=[1, 1, 1, 1], padding='VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
  with tf.name_scope('fc1'):
    fc1_weights = tf.Variable(tf.truncated_normal(
      [np.prod(relu.get_shape().as_list()[1:]), 512],
      stddev=0.1, seed=FLAGS.seed), name='fc1_weights')
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]), name='fc1_biases')
    pool_shape = relu.get_shape().as_list()
    reshape = tf.reshape(
      relu, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # _ = tf.histogram_summary('fc1_weights', fc1_weights)
    # _ = tf.histogram_summary('fc1_biases', fc1_biases)
  with tf.name_scope('fc2'):
    fc2_weights = tf.Variable(
      tf.truncated_normal([512, NUM_LABELS], stddev=0.1, seed=FLAGS.seed), name='fc2_weights')
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]), name='fc2_biases')
    logits = tf.matmul(hidden, fc2_weights) + fc2_biases
    # _ = tf.histogram_summary('fc2_weights', fc2_weights)
    # _ = tf.histogram_summary('fc2_biases', fc2_biases)
  with tf.name_scope('regularization'):
    regularizers = 5e-4 * (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                           tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
  with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, labels_node))
    loss += regularizers
    _ = tf.scalar_summary('loss', loss)
  with tf.name_scope('eval'):
    prediction = tf.sigmoid(logits, name='prediction')
    equality = tf.equal(tf.round(prediction), labels_node)
    accuracy = tf.reduce_mean(tf.cast(equality, 'float'))
    correct = tf.reduce_sum(tf.cast(tf.reduce_all(equality, 1), 'float'))
    _ = tf.scalar_summary('accuracy', accuracy)
    _ = tf.scalar_summary('correct', correct)
  optimizer = tf.train.AdamOptimizer().minimize(loss)
  with tf.Session() as s:
    summary_op = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(FLAGS.save, s.graph_def)
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    best_valid_acc = 0
    print('Model Initialized!')
    for step in xrange(FLAGS.epochs * train_size // FLAGS.batch):
      offset = (step * FLAGS.batch) % (train_size - FLAGS.batch)
      batch_data, batch_labels = get_batch(h5f, offset)
      summaries,_,l,acc,c = s.run(
        [summary_op, optimizer, loss, accuracy, correct],
        feed_dict={data_node:batch_data, labels_node:batch_labels})
      writer.add_summary(summaries, step)
      if step % 100 == 0:
        print('Epoch %.2f' % (float(step) * FLAGS.batch / train_size))
        print('  Minibatch: Loss=%.3f Accuracy=%.3f Correct=%.0f' % (l, acc, c))
        valid_acc, valid_correct = evaluate(s, data_node, labels_node, accuracy,
                                            correct, train_size,
                                            train_size + valid_size, h5f)
        print('  Validation: Accuracy=%.3f Correct=%.0f' % (valid_acc, valid_correct))
        if valid_acc > best_valid_acc:
          fname = os.path.join(FLAGS.save, 'best.ckpt')
          saver.save(s, fname)
          print('Saved model at: %s' % fname)
          best_valid_acc = valid_acc
        sys.stdout.flush()
    # Eval final model on test
    test_acc, test_correct = evaluate(s, data_node, labels_node, accuracy, correct,
                                      train_size + valid_size, n_examples, h5f)
    print('[Final] TestEval: Accuracy=%.1f Correct=%.0f' % (test_acc, test_correct))
    fname = os.path.join(FLAGS.save, 'final.ckpt')
    saver.save(s, fname)
    print('Model saved in file: %s' % fname)
    # Eval on best valid model
    saver.restore(s, os.path.join(FLAGS.save, 'best.ckpt'))
    print('Best model loaded from file: %s' % os.path.join(FLAGS.save, 'best.ckpt'))
    test_acc, test_correct = evaluate(s, data_node, labels_node, accuracy, correct,
                                      train_size + valid_size, n_examples, h5f)
    print('[Best] TestEval: Accuracy=%.1f Correct=%.0f' % (test_acc, test_correct))

if __name__ == '__main__':
  tf.app.run()
