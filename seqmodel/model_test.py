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
        m = model.Model(check_feed_dict=False)
        self.assertEqual(m._get_feed, m._get_feed_lite, 'no check feed method')
        m.check_feed_dict = True
        self.assertEqual(m._get_feed, m._get_feed_safe, 'check feed method')
        m.build_graph([self.f_feed, None], self.f_fetch)
        test_input, ignore_input = 1, 2
        output, __ = m.predict(self.sess, (test_input, ignore_input))
        self.assertEqual(output, test_input * 2, 'ignore none placeholder')
        self.assertRaises(ValueError, m.predict, self.sess, (None, None))

        def ten_fn(*args):
            return 10

        output, __ = m.predict(self.sess, [ten_fn])
        self.assertEqual(output, ten_fn() * 2, 'callable works')

    def test_run(self):
        m = model.Model(check_feed_dict=False)
        m.build_graph([self.f_feed], {'x': self.f_fetch}, label_feed=[self.l_feed],
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
