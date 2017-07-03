from functools import partial
import unittest

import numpy as np
import tensorflow as tf

from seqmodel import generator
from seqmodel.dstruct import Vocabulary
from seqmodel import graph
from seqmodel import model as tfm
from seqmodel import run
from seqmodel import util
from seqmodel import dstruct


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

    def tearDown(self):
        super().tearDown()
        graph.empty_tfph_collection('*')

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
            n = m.build_graph({'rnn:fn': 'seqmodel.graph.scan_rnn',
                               'cell:num_layers': 2})
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

    def test_lr_update(self):
        d = {'lr': 1}

        def set_lr(lr):
            d['lr'] = lr

        train_state = dstruct.TrainingState(learning_rate=1.0, cur_epoch=1)
        run.update_learning_rate(set_lr, train_state, start_decay_at=0, decay_every=1,
                                 decay_factor=0.5)
        self.assertEqual(train_state.learning_rate, 0.5, 'decay learning rate at 0')
        self.assertEqual(d['lr'], 0.5, 'learning rate is set')
        run.update_learning_rate(set_lr, train_state, start_decay_at=2, decay_every=1,
                                 decay_factor=0.5)
        self.assertEqual(train_state.learning_rate, 0.5, 'decay learning rate at 2')
        self.assertEqual(d['lr'], 0.5, 'learning rate is not change')
        run.update_learning_rate(set_lr, train_state, start_decay_at=0, decay_every=2,
                                 decay_factor=0.5)
        self.assertEqual(train_state.learning_rate, 0.5, 'decay lr every 2, not at 0')
        self.assertEqual(d['lr'], 0.5, 'learning rate is not change')
        # TODO: need more test for adaptive decay


class TestSamplingRun(tf.test.TestCase):

    sess_config = tf.ConfigProto(device_count={'GPU': 0})

    @classmethod
    def setUpClass(cls):
        super(TestSamplingRun, cls).setUpClass()
        data_dir = 'test_data/tiny_copy'
        cls.vocab = Vocabulary.from_vocab_file(f'{data_dir}/vocab.txt')
        cls.gen = partial(generator.read_lines, f'{data_dir}/valid.txt',
                          token_split=' ', part_split='\t')

    def tearDown(self):
        super().tearDown()
        graph.empty_tfph_collection('*')

    def test_run_epoch(self):
        data = generator.read_seq2seq_data(self.gen(), self.vocab, self.vocab)
        batch_iter = partial(generator.seq2seq_batch_iter, *data,
                             batch_size=20, shuffle=True)
        with self.test_session(config=self.sess_config) as sess:
            m = tfm.Seq2SeqModel()
            n = m.build_graph(**{'dec:attn_enc_output': True})
            m.set_default_feed('dec.train_loss_denom', 20)
            return_ph = tf.placeholder(tf.float32, shape=(None, None), name='return')
            train_op = graph.create_train_op(m.training_loss, learning_rate=0.01)
            train_pg_op = graph.create_pg_train_op(m.nll, return_ph)
            return_feed_fn = partial(m.set_default_feed, return_ph)
            sess.run(tf.global_variables_initializer())
            run_fn = partial(run.run_epoch, sess, m, batch_iter)
            for __ in range(5):  # do some pre-train
                train_info = run_fn(train_op=train_op)
                print(train_info.summary())
            reward_fn = generator.reward_match_label
            eval_info = run.run_sampling_epoch(
                sess, m, batch_iter, greedy=True, reward_fn=reward_fn)
            print(eval_info.summary('eval'))
            run_fn = partial(run.run_sampling_epoch, sess, m, batch_iter,
                             reward_fn=reward_fn)
            for __ in range(10):
                train_info = run_fn(train_op=train_pg_op, return_feed_fn=return_feed_fn)
                print(train_info.summary())
            eval_info2 = run.run_sampling_epoch(
                sess, m, batch_iter, greedy=True, reward_fn=reward_fn)
            print(eval_info2.summary('eval'))
            self.assertLess(eval_info2.eval_loss, eval_info.eval_loss,
                            'after training, eval loss is lower.')
