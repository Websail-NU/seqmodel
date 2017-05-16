import unittest

import numpy as np
import tensorflow as tf

from seqmodel import model


class TestModel(unittest.TestCase):

    def setUp(self):
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(config=sess_config)
        self.f_feed = tf.placeholder(tf.int32, shape=None)
        self.l_feed = tf.placeholder(tf.int32, shape=None)
        self.f_fetch = self.f_feed * 2
        self.t_fetch = self.f_feed * self.l_feed * 3
        self.e_fetch = self.f_feed * self.l_feed * 4

    def tearDown(self):
        tf.reset_default_graph()
        self.sess.close()

    def test_check_feed_dict(self):
        test_input, ignore_input = 1, 2
        m = model.Model(check_feed_dict=False)
        self.assertEqual(m._get_feed, m._get_feed_lite, 'no check feed method')
        m.set_graph([self.f_feed, None], self.f_fetch)
        self.assertRaises(TypeError, m.predict, self.sess, (test_input, ignore_input))
        m.check_feed_dict = True
        self.assertEqual(m._get_feed, m._get_feed_safe, 'check feed method')
        output, __ = m.predict(self.sess, (test_input, ignore_input))
        self.assertEqual(output, test_input * 2, 'ignore none placeholder')
        self.assertRaises(ValueError, m.predict, self.sess, (None, None))

        def ten_fn(*args):
            return 10

        output, __ = m.predict(self.sess, [ten_fn])
        self.assertEqual(output, ten_fn() * 2, 'callable works')

    def test_run(self):
        m = model.Model(check_feed_dict=False)
        m.set_graph([self.f_feed], {'x': self.f_fetch}, label_feed=[self.l_feed],
                    train_fetch=self.t_fetch, eval_fetch=self.e_fetch,
                    node_dict={'a': self.f_feed})
        test_input, test_label = 1, 10
        output, extra = m.predict(self.sess, [test_input], extra_fetch=['a'])
        self.assertEqual(output, {'x': test_input * 2}, 'prediction')
        self.assertEqual(extra, [test_input], 'extra')
        output, __ = m.predict(self.sess, [test_input], predict_key='x')
        self.assertEqual(output, test_input * 2, 'prediction key')
        output, __ = m.train(self.sess, [test_input], [test_label], tf.no_op())
        self.assertEqual(output, test_input * test_label * 3, 'train')
        output, __ = m.evaluate(self.sess, [test_input], [test_label])
        self.assertEqual(output, test_input * test_label * 4, 'eval')


class TestSeqModel(unittest.TestCase):

    def setUp(self):
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(config=sess_config)

    def tearDown(self):
        tf.reset_default_graph()
        self.sess.close()

    def test_build(self):
        m = model.SeqModel(check_feed_dict=False)
        opt = {'emb:vocab_size': 20, 'emb:dim': 5, 'cell:num_units': 10,
               'cell:cell_class': 'tensorflow.contrib.rnn.BasicLSTMCell',
               'logit:output_size': 2}
        expected_vars = {'t/embedding:0': (20, 5),
                         't/rnn/basic_lstm_cell/weights:0': (10 + 5, 10 * 4),
                         't/rnn/basic_lstm_cell/biases:0': (10 * 4,),
                         't/logit_w:0': (2, 10),
                         't/logit_b:0': (2,)}
        n = m.build_graph(opt, name='t')
        for v in tf.global_variables():
            self.assertTrue(v.name in expected_vars, 'expected variable scope/name')
            self.assertEqual(v.shape, expected_vars[v.name], 'shape is correct')

    def test_build_no_output(self):
        m = model.SeqModel(check_feed_dict=False)
        n = m.build_graph(**{'out:logit': False})
        self.assertTrue('logit' not in n, 'logit is not in nodes')

    def test_build_overwrite_opt(self):
        m = model.SeqModel(check_feed_dict=False)
        opt = {'emb:vocab_size': 20, 'cell:in_keep_prob': 0.5, 'logit:output_size': 2}
        n = m.build_graph(opt)
        self.assertEqual(n['emb_vars'].get_shape()[0], 20,
                         'overwrite default options')
        self.assertEqual(type(n['cell']), tf.contrib.rnn.DropoutWrapper,
                         'overwrite default options')
        n = m.build_graph(opt, reuse=True, **{'cell:in_keep_prob': 1.0})
        self.assertEqual(type(n['cell']), tf.contrib.rnn.BasicLSTMCell,
                         'overwrite default options with kwargs')

    def test_build_reuse(self):
        m = model.SeqModel(check_feed_dict=False)
        n = m.build_graph()
        num_vars = len(tf.global_variables())
        n = m.build_graph(reuse=True, **{'cell:in_keep_prob': 1.0})
        self.assertEqual(type(n['cell']), tf.contrib.rnn.BasicLSTMCell,
                         'overwrite default options with kwargs')
        num_vars_ = len(tf.global_variables())
        self.assertEqual(num_vars, num_vars, 'no new variables when reuse is True')
