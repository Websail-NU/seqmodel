import unittest

import numpy as np
import tensorflow as tf

from seqmodel import graph


class TestGraph(unittest.TestCase):

    def setUp(self):
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(config=sess_config)
        self.batch_size = 3
        self.max_seq_len = 4
        self.dim = 5
        self.seq_len = [4, 2, 0]
        self.num_layers = 2

    def tearDown(self):
        tf.reset_default_graph()
        self.sess.close()

    def test_rnn(self):
        seq_len = tf.constant(self.seq_len, dtype=tf.int32)
        cell = graph.create_cells(self.dim, self.num_layers, in_keep_prob=0.5)
        self.assertTrue(isinstance(cell._cells[0], tf.contrib.rnn.DropoutWrapper),
                        'input dropout at first layer')
        self.assertTrue(isinstance(cell._cells[-1], tf.contrib.rnn.BasicLSTMCell),
                        'no input dropout after first layer')
        o, i, f = graph.create_rnn(cell, tf.constant(
            np.random.randn(self.max_seq_len, self.batch_size, self.dim),
            dtype=tf.float32), seq_len)
        x = graph.select_rnn_output(o, tf.nn.relu(seq_len - 1))
        self.sess.run(tf.global_variables_initializer())
        o_, x_ = self.sess.run([o, x])
        self.assertTrue(np.all(o_[:, -1, :] == 0), 'output is zero if seq len is zero')
        for i, last in enumerate(self.seq_len):
            if last == 0:
                continue
            self.assertTrue(np.all(o_[last - 1, i, :] == x_[i]),
                            'last relevant is correct')

    def test_tdnn(self):
        tdnn = graph.create_tdnn(tf.constant(
            np.random.randn(self.max_seq_len, self.batch_size, self.dim),
            dtype=tf.float32))
        self.assertEqual(tdnn.get_shape()[-1], 160,
                         'default fileter setting results in feature size of 160')

    def test_placeholders(self):
        input_, seq_len_ = graph.get_seq_input_placeholders()
        input2_, seq2_len_ = graph.get_seq_input_placeholders()
        self.assertEqual(input_, input2_, 'reuse placeholders of the same name')
        label_, tk_w, seq_w = graph.get_seq_label_placeholders()
        lookup, emb_vars = graph.get_lookup(input_, vocab_size=10, dim=5)
        onehot, __ = graph.get_lookup(input_, onehot=True, vocab_size=10,
                                      lookup_name='onehotlookup')
