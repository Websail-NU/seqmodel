from functools import partial
import unittest

import numpy as np
import tensorflow as tf

from seqmodel import generator
from seqmodel.dstruct import Vocabulary
from seqmodel import model as tfm
from seqmodel import run
from seqmodel import util


class TestRun(tf.test.TestCase):

    sess_config = tf.ConfigProto(device_count={'GPU': 0})

    @classmethod
    def setUpClass(cls):
        super(TestRun, cls).setUpClass()
        data_dir = 'test_data/tiny_single'
        cls.vocab = Vocabulary.from_vocab_file(f'{data_dir}/vocab.txt')
        cls.gen = partial(generator.read_lines, f'{data_dir}/valid.txt',
                          token_split=' ')
        cls.num_lines = 1000
        cls.num_tokens = 5606

    def test_run_epoch(self):
        data = generator.read_seq_data(
            self.gen(), self.vocab, self.vocab, keep_sentence=False, seq_len=20)
        batch_iter = partial(generator.seq_batch_iter, *data,
                             batch_size=13, shuffle=True, keep_sentence=False)
        with self.test_session(config=self.sess_config) as sess:
            m = tfm.SeqModel()
            n = m.build_graph()
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(m.training_loss)
            sess.run(tf.global_variables_initializer())
            run_fn = partial(run.run_epoch, sess, m, batch_iter)
            eval_info = run_fn()
            self.assertEqual(eval_info.num_tokens, self.num_lines + self.num_tokens,
                             'run uses all tokens')
            self.assertAlmostEqual(eval_info.eval_loss, np.log(self.vocab.vocab_size),
                                   places=1, msg='eval loss is close to uniform.')
            for __ in range(3):
                train_info = run_fn(train_op=train_op)
            self.assertLess(train_info.eval_loss, eval_info.eval_loss,
                            'after training, eval loss is lower.')

    def test_train(self):
        logger = util.get_logger(level='warning')
        data = generator.read_seq_data(
            self.gen(), self.vocab, self.vocab, keep_sentence=False, seq_len=20)
        batch_iter = partial(generator.seq_batch_iter, *data,
                             batch_size=13, shuffle=True, keep_sentence=False)
        with self.test_session(config=self.sess_config) as sess:
            m = tfm.SeqModel()
            n = m.build_graph()
            m.set_default_feed('train_loss_denom', 13)
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(m.training_loss)
            sess.run(tf.global_variables_initializer())
            train_run_fn = partial(run.run_epoch, sess, m, batch_iter, train_op=train_op)
            eval_run_fn = partial(run.run_epoch, sess, m, batch_iter)

            def stop_early(*args, **kwargs):
                pass

            eval_info = eval_run_fn()
            train_state = run.train(train_run_fn, logger, max_epoch=3,
                                    valid_run_epoch_fn=eval_run_fn,
                                    end_epoch_fn=stop_early)
            self.assertLess(train_state.best_eval, eval_info.eval_loss,
                            'after training, eval loss is lower.')
            self.assertEqual(train_state.cur_epoch, 3, 'train for max epoch')
