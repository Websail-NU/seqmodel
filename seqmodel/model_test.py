import unittest

import numpy as np
import tensorflow as tf

from seqmodel import dstruct
from seqmodel import model
from seqmodel import graph


def _run(obj, model_class, rnn_fn, mode='seq', build_opt={}):
    with obj.test_session(config=obj.sess_config) as sess:
        seq, seq_len = np.ones((4, 3)), np.array([2, 3, 0], dtype=np.int32)
        pk, pkn = 'dec:', 'dec.'
        if mode == 'seq2seq':
            features = (seq, seq_len, seq, seq_len)
        elif mode == 'word2def':
            features = (
                seq, seq_len, np.ones((3,)), seq.T, seq_len,
                np.full((3,), -1), seq, seq_len)
        else:
            features = (seq, seq_len)
            pk, pkn = '', ''
        m = model_class(check_feed_dict=False)
        n = m.build_graph({'rnn:fn': rnn_fn, f'{pk}logit:output_size': 2}, **build_opt)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(m.training_loss)
        sess.run(tf.global_variables_initializer())
        # prediction
        output, __ = m.predict(sess, features, fetch_state=False)
        co = output['cell_output']
        for iseq in range(co.shape[1]):
            np.testing.assert_array_equal(co[seq_len[iseq]:, iseq, :], 0,
                                          'cell output is zero after seq_len')
        for i in range(2):
            obj.assertEqual(output['dec_sample'][i].shape, (4, 3),
                            'sample shape is the same as input\'s')
            obj.assertEqual(output['dec_max'][i].shape, (4, 3),
                            'sample shape is the same as input\'s')
        obj.assertEqual(output['logit'].shape, (4, 3, 2), 'logit shape is correct')
        obj.assertEqual(output['dist'].shape, (4, 3, 2), 'dist shape is correct')
        output, __ = m.predict(sess, features, fetch_state=True,
                               predict_key='cell_output')
        output2, __ = m.predict(sess, features, fetch_state=True,
                                predict_key='cell_output')
        output3, __ = m.predict(sess, features, fetch_state=False,
                                state=output.state, predict_key='cell_output')
        obj.assertIsInstance(output, dstruct.OutputStateTuple, 'output with state')
        np.testing.assert_allclose(output.output, output2.output,
                                   err_msg='same input, same state, same output')
        np.testing.assert_allclose(output.state.h, output2.state.h,
                                   err_msg='same input, same state h')
        np.testing.assert_allclose(output.state.c, output2.state.c,
                                   err_msg='same input, same state c')
        obj.assertRaises(AssertionError, np.testing.assert_array_equal,
                         output.output, output3)
        # evaluation
        output, __ = m.evaluate(sess, features, (seq, seq, np.ones((3))))
        obj.assertNotEqual(output['avg.tokens::eval_loss'], 0.0,
                           'eval_loss is not zero')
        output, __ = m.evaluate(sess, features, (seq, seq, np.zeros((3))))
        obj.assertEqual(output['avg.tokens::eval_loss'], 0.0,
                        'eval_loss is zero if seq_weight is zero')
        output, __ = m.evaluate(sess, features, (seq, np.zeros((4, 3)), np.ones((3))))
        obj.assertEqual(output['avg.tokens::eval_loss'], 0.0,
                        'eval_loss is zero if token_weight is zero')
        # training
        output, __ = m.train(sess, features, (seq, seq, np.ones((3))), m._no_op)
        obj.assertGreaterEqual(
            output['train_loss'], output['avg.tokens::eval_loss'],
            'sum loss is at least larger than mean loss.')
        for i in range(20):
            output2, __ = m.train(sess, features, (seq, seq, np.ones((3))), train_op)
        obj.assertLess(output2['train_loss'], output['train_loss'],
                       'training loss is lower after training')
        m.set_default_feed(f'{pkn}train_loss_denom', 1)
        output3, __ = m.train(sess, features, (seq, seq, np.ones((3))), m._no_op)
        m.set_default_feed(f'{pkn}train_loss_denom', 10)
        output4, __ = m.train(sess, features, (seq, seq, np.ones((3))), m._no_op)
        obj.assertAlmostEqual(output4['train_loss'], output3['train_loss'] / 10,
                              places=1, msg='training loss denom is used')
        # decode
        if mode == 'seq2seq' or mode == 'word2def':
            # just smoke test for decode result
            output4, __ = m.predict(sess, features, predict_key='decode_greedy')
            np.testing.assert_array_equal(output4[:, :2], 1, err_msg='output is all 1')


class TestModel(tf.test.TestCase):

    sess_config = tf.ConfigProto(device_count={'GPU': 0})

    def setUp(self):
        super().setUp()
        self.f_feed = tf.placeholder(tf.int32, shape=None)
        self.l_feed = tf.placeholder(tf.int32, shape=None)
        self.f_fetch = self.f_feed * 2
        self.t_fetch = self.f_feed * self.l_feed * 3
        self.e_fetch = self.f_feed * self.l_feed * 4

    def tearDown(self):
        super().tearDown()
        graph.empty_tfph_collection('*')

    def test_check_feed_dict(self):
        with self.test_session(config=self.sess_config) as sess:
            test_input, ignore_input = 1, 2
            m = model.Model(check_feed_dict=False)
            self.assertEqual(m._get_feed, m._get_feed_lite, 'no check feed method')
            m.set_graph([self.f_feed, None], self.f_fetch)
            self.assertRaises(TypeError, m.predict, sess, (test_input, ignore_input))
            m.check_feed_dict = True
            self.assertEqual(m._get_feed, m._get_feed_safe, 'check feed method')
            output, __ = m.predict(sess, (test_input, ignore_input))
            self.assertEqual(output, test_input * 2, 'ignore none placeholder')
            self.assertRaises(ValueError, m.predict, sess, (None, None))

            def ten_fn(*args):
                return 10

            output, __ = m.predict(sess, [ten_fn])
            self.assertEqual(output, ten_fn() * 2, 'callable works')

    def test_run(self):
        with self.test_session(config=self.sess_config) as sess:
            m = model.Model(check_feed_dict=False)
            m.set_graph([self.f_feed], {'x': self.f_fetch},
                        label_feed=[self.l_feed], train_fetch=self.t_fetch,
                        eval_fetch=self.e_fetch,
                        node_dict={'a': self.f_feed, 'b': self.e_fetch})
            test_input, test_label = 1, 10
            output, extra = m.predict(sess, [test_input], extra_fetch=['a'])
            self.assertEqual(output, {'x': test_input * 2}, 'prediction')
            self.assertEqual(extra, [test_input], 'extra')
            output, __ = m.predict(sess, [test_input], predict_key='x')
            self.assertEqual(output, test_input * 2, 'prediction key')
            output, __ = m.train(sess, [test_input], [test_label], tf.no_op())
            self.assertEqual(output, test_input * test_label * 3, 'train')
            output, __ = m.evaluate(sess, [test_input], [test_label])
            self.assertEqual(output, test_input * test_label * 4, 'eval')

    def test_default_feed(self):
        with self.test_session(config=self.sess_config) as sess:
            m = model.Model(check_feed_dict=False)
            m.set_graph([self.f_feed], {'x': self.f_fetch, 'y': self.e_fetch},
                        label_feed=[self.l_feed], train_fetch=self.t_fetch,
                        eval_fetch=self.e_fetch,
                        node_dict={'a': self.f_feed, 'b': self.e_fetch},
                        default_feed={self.e_fetch: 10})
            test_input, test_label = 1, 10
            output, __ = m.predict(sess, [test_input])
            self.assertEqual(output, {'x': test_input * 2, 'y': 10},
                             'prediction with default feed')
            m.set_default_feed('b', 200)
            output, __ = m.predict(sess, [test_input])
            self.assertEqual(output, {'x': test_input * 2, 'y': 200},
                             'change value of default feed')
            m.set_default_feed('a', 100)
            self.assertEqual(output, {'x': test_input * 2, 'y': 200},
                             'default feed is overwritten')


class TestSeqModel(tf.test.TestCase):

    sess_config = tf.ConfigProto(device_count={'GPU': 0})

    def tearDown(self):
        super().tearDown()
        graph.empty_tfph_collection('*')

    def test_build(self):
        with self.test_session(config=self.sess_config) as sess:
            m = model.SeqModel(check_feed_dict=False)
            opt = {'emb:vocab_size': 20, 'emb:dim': 5, 'cell:num_units': 10,
                   'cell:cell_class': 'tensorflow.nn.rnn_cell.BasicLSTMCell',
                   'logit:output_size': 2}
            expected_vars = {'t/embedding:0': (20, 5),
                             't/rnn/basic_lstm_cell/kernel:0': (10 + 5, 10 * 4),
                             't/rnn/basic_lstm_cell/bias:0': (10 * 4,),
                             't/logit_w:0': (2, 10),
                             't/logit_b:0': (2,)}
            n = m.build_graph(opt, name='t')
            for v in tf.global_variables():
                self.assertTrue(v.name in expected_vars, 'expected variable scope/name')
                self.assertEqual(v.shape, expected_vars[v.name], 'shape is correct')
            for k, v in m._fetches.items():
                if k is not None:
                    self.assertNotEqual(v[0], v[1], 'fetch array is set')

    def test_build_no_output(self):
        with self.test_session(config=self.sess_config) as sess:
            m = model.SeqModel(check_feed_dict=False)
            n = m.build_graph(**{'out:logit': False, 'out:loss': False})
            self.assertTrue('logit' not in n, 'logit is not in nodes')
            for k, v in m._fetches.items():
                if k is None or k == m._TRAIN_ or k == m._EVAL_:
                    self.assertEqual(v[0], v[1], 'fetch array is not set')
                if k == m._PREDICT_:
                    self.assertNotEqual(v[0], v[1], 'predict fetch array is set')
            self.assertRaises(ValueError, m.build_graph, **{'out:logit': False})

    def test_build_decode(self):
        with self.test_session(config=self.sess_config) as sess:
            m = model.SeqModel(check_feed_dict=False)
            n = m.build_graph(**{'out:logit': True, 'out:decode': True,
                                 'decode:add_greedy': True,
                                 'decode:add_sampling': True})
            self.assertTrue('decode_greedy' in n, 'decode_greedy in nodes')
            self.assertTrue('decode_greedy' in m._predict_fetch,
                            'decode_greedy in predict dict')
            self.assertTrue('decode_sampling' in n, 'decode_sampling in nodes')
            self.assertTrue('decode_sampling' in m._predict_fetch,
                            'decode_sampling in predict dict')

    def test_build_overwrite_opt(self):
        with self.test_session(config=self.sess_config) as sess:
            m = model.SeqModel(check_feed_dict=False)
            opt = {'emb:vocab_size': 20, 'cell:in_keep_prob': 0.5,
                   'logit:output_size': 2}
            n = m.build_graph(opt)
            self.assertEqual(n['emb_vars'].get_shape()[0], 20,
                             'overwrite default options')
            self.assertEqual(type(n['cell']), tf.nn.rnn_cell.DropoutWrapper,
                             'overwrite default options')
            n = m.build_graph(opt, reuse=True, **{'cell:in_keep_prob': 1.0})
            self.assertEqual(type(n['cell']), tf.nn.rnn_cell.BasicLSTMCell,
                             'overwrite default options with kwargs')

    def test_build_reuse(self):
        with self.test_session(config=self.sess_config) as sess:
            m = model.SeqModel(check_feed_dict=False)
            n = m.build_graph()
            num_vars = len(tf.global_variables())
            n = m.build_graph(reuse=True, no_dropout=True)
            self.assertEqual(type(n['cell']), tf.nn.rnn_cell.BasicLSTMCell,
                             'overwrite default options with kwargs')
            num_vars_ = len(tf.global_variables())
            self.assertEqual(num_vars, num_vars, 'no new variables when reuse is True')

    def test_build_gns(self):
        with self.test_session(config=self.sess_config) as sess:
            m = model.SeqModel(check_feed_dict=False)
            opt = {'emb:vocab_size': 20, 'emb:dim': 5, 'cell:num_units': 10,
                   'cell:cell_class': 'tensorflow.nn.rnn_cell.BasicLSTMCell',
                   'logit:output_size': 2, 'loss:add_gns': True}
            gns_node_names = [
                'log_ckld', 'rep_cond_assign', 'p0_repk', 'p_repk', 'unigram_assign',
                'p0_unigram', 'p_unigram', 'gns_decay']
            n = m.build_graph(opt, name='t')
            for k in gns_node_names:
                self.assertTrue(k in n, msg=f'{k} must be built.')

    def test_build_entropy(self):
        with self.test_session(config=self.sess_config) as sess:
            m = model.SeqModel(check_feed_dict=False)
            opt = {'emb:vocab_size': 20, 'emb:dim': 5, 'cell:num_units': 10,
                   'cell:cell_class': 'tensorflow.nn.rnn_cell.BasicLSTMCell',
                   'logit:output_size': 2, 'loss:add_entropy': True}
            n = m.build_graph(opt, name='t')
            self.assertTrue('minus_avg_ent' in n, msg='minus_avg_ent must be built.')

    def test_dynamic_rnn_run(self):
        _run(self, model.SeqModel, tf.nn.dynamic_rnn)

    def test_scan_rnn_run(self):
        _run(self, model.SeqModel, graph.scan_rnn)


class TestSeq2SeqModel(tf.test.TestCase):

    sess_config = tf.ConfigProto(device_count={'GPU': 0})

    def tearDown(self):
        super().tearDown()
        graph.empty_tfph_collection('*')

    def test_build(self):
        with self.test_session(config=self.sess_config) as sess:
            m = model.Seq2SeqModel(check_feed_dict=False)
            opt = {'emb:vocab_size': 20, 'emb:dim': 5, 'cell:num_units': 10,
                   'cell:cell_class': 'tensorflow.nn.rnn_cell.BasicLSTMCell'}
            opt = {f'{n}:{k}': v for k, v in opt.items() for n in ('enc', 'dec')}
            opt['dec:logit:output_size'] = 2
            expected_vars = {'embedding:0': (20, 5),
                             'rnn/basic_lstm_cell/kernel:0': (10 + 5, 10 * 4),
                             'rnn/basic_lstm_cell/bias:0': (10 * 4,)}
            expected_vars = {f't/{n}/{k}': v for k, v in expected_vars.items()
                             for n in ('enc', 'dec')}
            expected_vars.update({'t/dec/logit_w:0': (2, 10), 't/dec/logit_b:0': (2,)})
            n = m.build_graph(opt, name='t')
            for v in tf.global_variables():
                # print(f'{v.name}, {v.get_shape()}')
                self.assertTrue(v.name in expected_vars, 'expected variable scope/name')
                self.assertEqual(v.shape, expected_vars[v.name], 'shape is correct')
            for k, v in m._fetches.items():
                if k is not None:
                    self.assertNotEqual(v[0], v[1], 'fetch array is set')

    def test_build_share_emb(self):
        with self.test_session(config=self.sess_config) as sess:
            m = model.Seq2SeqModel(check_feed_dict=False)
            opt = {'emb:vocab_size': 20, 'emb:dim': 5, 'cell:num_units': 10,
                   'cell:cell_class': 'tensorflow.nn.rnn_cell.BasicLSTMCell'}
            opt = {f'{n}:{k}': v for k, v in opt.items() for n in ('enc', 'dec')}
            opt['dec:logit:output_size'] = 2
            n = m.build_graph(opt, name='t', **{'share:enc_dec_emb': True})
            c = 0
            for v in tf.global_variables():
                self.assertNotEqual(v.name, 't/dec/embedding:0', 'no decoder emb')
                if '/embedding:0' in v.name:
                    c += 1
            self.assertEqual(c, 1, 'only one embedding')

    def test_build_share_rnn(self):
        with self.test_session(config=self.sess_config) as sess:
            m = model.Seq2SeqModel(check_feed_dict=False)
            opt = {'emb:vocab_size': 20, 'emb:dim': 5, 'cell:num_units': 10,
                   'cell:cell_class': 'tensorflow.nn.rnn_cell.BasicLSTMCell'}
            opt = {f'{n}:{k}': v for k, v in opt.items() for n in ('enc', 'dec')}
            opt['dec:logit:output_size'] = 2
            c = 0
            n = m.build_graph(opt=opt, name='q', **{'share:enc_dec_rnn': True})
            for v in tf.global_variables():
                self.assertNotEqual(v.name, 'q/dec/rnn/basic_lstm_cell/kernel:0',
                                    'no decoder rnn')
                self.assertNotEqual(v.name, 'q/dec/rnn/basic_lstm_cell/bias:0',
                                    'no decoder rnn')
                if 'rnn/basic_lstm_cell/kernel:0' in v.name:
                    c += 1
            self.assertEqual(c, 1, 'only one cell')

    def test_build_share_emb_rnn_reuse(self):
        with self.test_session(config=self.sess_config) as sess:
            m = model.Seq2SeqModel(check_feed_dict=False)
            opt = {'emb:vocab_size': 20, 'emb:dim': 5, 'cell:num_units': 10,
                   'cell:cell_class': 'tensorflow.nn.rnn_cell.BasicLSTMCell'}
            opt = {f'{n}:{k}': v for k, v in opt.items() for n in ('enc', 'dec')}
            opt['dec:logit:output_size'] = 2
            c_rnn = 0
            c_emb = 0
            n = m.build_graph(
                opt=opt, name='q',
                **{'share:enc_dec_emb': True, 'share:enc_dec_rnn': True})
            for v in tf.global_variables():
                self.assertNotEqual(v.name, 'q/dec/rnn/basic_lstm_cell/kernel:0',
                                    'no decoder rnn')
                self.assertNotEqual(v.name, 'q/dec/rnn/basic_lstm_cell/bias:0',
                                    'no decoder rnn')
                if 'rnn/basic_lstm_cell/kernel:0' in v.name:
                    c_rnn += 1
                self.assertNotEqual(v.name, 't/dec/embedding:0', 'no decoder emb')
                if '/embedding:0' in v.name:
                    c_emb += 1
            self.assertEqual(c_emb, 1, 'only one embedding')
            self.assertEqual(c_rnn, 1, 'only one cell')
            num_vars = len(tf.global_variables())
            n = m.build_graph(
                opt=opt, name='q', reuse=True,
                **{'share:enc_dec_emb': True, 'share:enc_dec_rnn': True})
            num_vars_ = len(tf.global_variables())
            self.assertEqual(num_vars, num_vars, 'no new variables when reuse is True')

    def test_build_attn(self):
        with self.test_session(config=self.sess_config) as sess:
            m = model.Seq2SeqModel(check_feed_dict=False)
            opt = {'emb:vocab_size': 20, 'emb:dim': 5, 'cell:num_units': 10,
                   'cell:cell_class': 'tensorflow.nn.rnn_cell.BasicLSTMCell'}
            opt = {f'{n}:{k}': v for k, v in opt.items() for n in ('enc', 'dec')}
            opt['dec:logit:output_size'] = 2
            expected_vars = {'embedding:0': (20, 5),
                             'rnn/basic_lstm_cell/kernel:0': (10 + 5, 10 * 4),
                             'rnn/basic_lstm_cell/bias:0': (10 * 4,)}
            expected_vars = {f't/{n}/{k}': v for k, v in expected_vars.items()
                             for n in ('enc', 'dec')}
            expected_vars.update({
                't/dec/logit_w:0': (2, 10), 't/dec/logit_b:0': (2,),
                't/dec/attention/dense/kernel:0': (20, 10),
                't/dec/attention/dense/bias:0': (10,)})
            n = m.build_graph(opt, name='t', **{'dec:attn_enc_output': True})
            for v in tf.global_variables():
                # print(f'{v.name}, {v.get_shape()}')
                self.assertTrue(v.name in expected_vars, 'expected variable scope/name')
                self.assertEqual(v.shape, expected_vars[v.name], 'shape is correct')
            for k, v in m._fetches.items():
                if k is not None:
                    self.assertNotEqual(v[0], v[1], 'fetch array is set')

    def test_dynamic_rnn_run(self):
        _run(self, model.Seq2SeqModel, tf.nn.dynamic_rnn, 'seq2seq')

    def test_scan_rnn_run(self):
        _run(self, model.Seq2SeqModel, graph.scan_rnn, 'seq2seq',
             build_opt={'dec:attn_enc_output': True})


class TestWord2DefModel(tf.test.TestCase):

    sess_config = tf.ConfigProto(device_count={'GPU': 0})

    def tearDown(self):
        super().tearDown()
        graph.empty_tfph_collection('*')

    def test_build(self):
        with self.test_session(config=self.sess_config) as sess:
            m = model.Word2DefModel(check_feed_dict=False)
            opt = {'emb:vocab_size': 20, 'emb:dim': 5, 'cell:num_units': 10,
                   'cell:cell_class': 'tensorflow.nn.rnn_cell.BasicLSTMCell',
                   'cell:out_keep_prob': 0.5}
            opt = {f'{n}:{k}': v for k, v in opt.items() for n in ('enc', 'dec')}
            opt['dec:logit:output_size'] = 2
            opt['wbdef:keep_prob'] = 0.5
            expected_vars = {'embedding:0': (20, 5),
                             'rnn/basic_lstm_cell/kernel:0': (10 + 5, 10 * 4),
                             'rnn/basic_lstm_cell/bias:0': (10 * 4,)}
            expected_vars = {f't/{n}/{k}': v for k, v in expected_vars.items()
                             for n in ('enc', 'dec')}
            expected_vars.update({'t/dec/logit_w:0': (2, 10), 't/dec/logit_b:0': (2,)})
            expected_vars.update({'t/wbdef/filter_2:0': (1, 2, 55, 20),
                                  't/wbdef/filter_3:0': (1, 3, 55, 30),
                                  't/wbdef/filter_4:0': (1, 4, 55, 40),
                                  't/wbdef/filter_5:0': (1, 5, 55, 40),
                                  't/wbdef/gate_zr/kernel:0': (145, 145),
                                  't/wbdef/gate_zr/bias:0': (145,),
                                  't/wbdef/transform/kernel:0': (145, 10),
                                  't/wbdef/transform/bias:0': (10,)})
            n = m.build_graph(opt, name='t')
            self.assertIsInstance(n['dec']['cell'], tf.nn.rnn_cell.BasicLSTMCell,
                                  'no dropout at the final cell (1-layer)')

            self.assertTrue('dropout' in n['bridge']['updated_output'].name, 'dropout')
            for v in tf.global_variables():
                self.assertTrue(v.name in expected_vars, 'expected variable scope/name')
                self.assertEqual(v.shape, expected_vars[v.name], 'shape is correct')
            for k, v in m._fetches.items():
                if k is not None:
                    self.assertNotEqual(v[0], v[1], 'fetch array is set')

    def test_dropout(self):
        with self.test_session(config=self.sess_config) as sess:
            m = model.Word2DefModel(check_feed_dict=False)
            opt = {'cell:num_layers': 2, 'cell:out_keep_prob': 0.5,
                   'cell:in_keep_prob': 0.5}
            opt = {f'{n}:{k}': v for k, v in opt.items() for n in ('enc', 'dec')}
            opt['wbdef:keep_prob'] = 0.5
            n = m.build_graph(opt, name='t')
            self.assertNotEqual(type(n['dec']['cell']._cells[-1]),
                                tf.nn.rnn_cell.DropoutWrapper)
            self.assertTrue('dropout' in n['bridge']['updated_output'].name)

    def test_dynamic_rnn_run(self):
        _run(self, model.Word2DefModel, tf.nn.dynamic_rnn, 'word2def')

    def test_scan_rnn_run(self):
        _run(self, model.Word2DefModel, graph.scan_rnn, 'word2def')
