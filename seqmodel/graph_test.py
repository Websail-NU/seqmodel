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

    def test_neural_cache(self):
        for i in range(5):
            with tf.variable_scope(f'{i}'):
                self._test_neural_cache()

    def _test_neural_cache(self):
        batch_size = 3
        cache_size = 4
        key_dims = 2
        inputs = [[8, 4, 3],
                  [1, 3, 9],
                  [17, 16, 14],
                  [18, 7, 4],
                  [3, 4, 2]]
        h = np.random.randn(5, batch_size, key_dims)
        X = tf.constant(inputs, dtype=tf.int32)
        H = tf.constant(h, dtype=tf.float32)
        ctime, cvalues, cscores, reset_op = graph.create_neural_cache(
            H, X, batch_size, key_dims, cache_size, default_v=-1)
        with self.test_session(config=self.sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            t1, v1, s1 = sess.run([ctime, cvalues, cscores])
            t2, v2, s2 = sess.run([ctime, cvalues, cscores])
            np.testing.assert_array_equal(
                t1[0], np.full((cache_size, batch_size), -1, dtype=np.int32),
                err_msg='initial cache times are -1', verbose=True)
            np.testing.assert_array_equal(
                t2[-1], np.array([[8]*3, [5]*3, [6]*3, [7]*3]),
                err_msg='cache time is not reset between run', verbose=True)
            np.testing.assert_array_equal(
                v1[0], np.full((cache_size, batch_size), -1, dtype=np.int32),
                err_msg='initial cache values are -1', verbose=True)
            np.testing.assert_array_equal(
                v1[-1], inputs[0:4], err_msg='cache values are correct', verbose=True)
            np.testing.assert_array_equal(
                v2[0][0], inputs[4], err_msg='cache values persist', verbose=True)
            np.testing.assert_array_equal(
                s1[0], np.zeros((cache_size, batch_size)),
                err_msg='initial cache scores are zero', verbose=True)
            np.testing.assert_array_almost_equal(
                s1[4][0], np.sum(h[4] * h[0], axis=-1),
                err_msg='some cache values are correct', verbose=True)
            np.testing.assert_array_almost_equal(
                s1[3][1], np.sum(h[3] * h[1], axis=-1),
                err_msg='some cache values are correct', verbose=True)
            np.testing.assert_array_almost_equal(
                s2[0][0], np.sum(h[0] * h[4], axis=-1),
                err_msg='some cache values are correct', verbose=True)
            np.testing.assert_array_almost_equal(
                s2[3][1], np.sum(h[3] * h[0], axis=-1),
                err_msg='some cache values are correct', verbose=True)
            sess.run(reset_op)
            t3, v3, s3 = sess.run([ctime, cvalues, cscores])
            np.testing.assert_array_equal(t3, t1, err_msg='', verbose=True)
            np.testing.assert_array_equal(v3, v1, err_msg='', verbose=True)
            np.testing.assert_array_equal(s3, s1, err_msg='', verbose=True)
