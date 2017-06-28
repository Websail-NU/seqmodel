import unittest

import numpy as np
import tensorflow as tf

from seqmodel import graph


class TestGraph(tf.test.TestCase):

    sess_config = tf.ConfigProto(device_count={'GPU': 0})

    def setUp(self):
        super().setUp()
        self.batch_size = 3
        self.max_seq_len = 4
        self.dim = 5
        self.seq_len = [4, 2, 0]
        self.num_layers = 2

    def tearDown(self):
        super().tearDown()
        graph.empty_tfph_collection('*')

    def test_rnn(self):
        with self.test_session(config=self.sess_config) as sess:
            seq_len = tf.constant(self.seq_len, dtype=tf.int32)
            cell = graph.create_cells(self.dim, self.num_layers, in_keep_prob=0.5)
            self.assertTrue(isinstance(cell._cells[0], tf.nn.rnn_cell.DropoutWrapper),
                            'input dropout at first layer')
            self.assertTrue(isinstance(cell._cells[-1], tf.nn.rnn_cell.BasicLSTMCell),
                            'no input dropout after first layer')
            o, i, f = graph.create_rnn(cell, tf.constant(
                np.random.randn(self.max_seq_len, self.batch_size, self.dim),
                dtype=tf.float32), seq_len)
            x = graph.select_rnn(o, tf.nn.relu(seq_len - 1))
            sess.run(tf.global_variables_initializer())
            o_, x_ = sess.run([o, x])
            np.testing.assert_array_equal(o_[:, -1, :], 0,
                'output is zero if seq len is zero')  # noqa
            for i, last in enumerate(self.seq_len):
                if last == 0:
                    continue
                np.testing.assert_array_equal(o_[last - 1, i, :], x_[i])

    def test_tdnn(self):
        with self.test_session(config=self.sess_config) as sess:
            tdnn = graph.create_tdnn(tf.constant(
                np.random.randn(self.max_seq_len, self.batch_size, self.dim),
                dtype=tf.float32))
            self.assertEqual(tdnn.get_shape()[-1], 160,
                             'default fileter setting results in feature size of 160')

    def test_input_output(self):
        with self.test_session(config=self.sess_config) as sess:
            input_, seq_len_ = graph.get_seq_input_placeholders()
            input2_, seq2_len_ = graph.get_seq_input_placeholders()
            self.assertEqual(input_, input2_, 'reuse placeholders of the same name')
            label_, tk_w, seq_w = graph.get_seq_label_placeholders()
            emb_init = np.random.randn(10, 5)
            lookup, emb_vars = graph.create_lookup(input_, vocab_size=10, dim=5,
                                                   init=emb_init)
            onehot, __ = graph.create_lookup(input_, onehot=True, vocab_size=10,
                                             prefix='onehot')
            sess.run(tf.global_variables_initializer())
            emb_tf = sess.run(emb_vars)
            self.assertAllClose(emb_tf, emb_init)  # 'init embedding works'
            output = sess.run(onehot, {input_: [[1]]})
            self.assertEqual(output.shape, (1, 1, 10), 'onehot works')
            self.assertEqual(output[0, 0, 1], 1, 'onehot works')
            self.assertEqual(np.sum(output), 1, 'onehot works')
