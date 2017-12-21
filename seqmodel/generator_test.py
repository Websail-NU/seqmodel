from collections import Counter
from functools import partial
import unittest

import kenlm
import numpy as np

from seqmodel.dstruct import Vocabulary
from seqmodel import dstruct
from seqmodel import generator


class TestBatch(unittest.TestCase):

    def test_position_batch_iter(self):
        pos = np.arange(10).reshape(2, 5)
        for i, b in enumerate(generator.position_batch_iter(
                10, batch_size=2, shuffle=False, keep_state=False)):
            np.testing.assert_array_equal(list(b.features)[0], pos[:, i],
                                          'generate position')
            self.assertFalse(b.keep_state, 'keep state is False')
            self.assertEqual(b.num_tokens, 2, 'num tokens is batch size')
        num_tokens = np.arange(10)
        for i, b in enumerate(generator.position_batch_iter(
                10, shuffle=False, num_tokens=num_tokens, keep_state=True)):
            self.assertTrue(b.keep_state, 'keep state is True')
            self.assertEqual(b.num_tokens, num_tokens[i], 'num tokens is correct')

    def test_get_batch_data(self):
        data = np.array([[12, 10, 10, 0], [12, 8, 7, 0], [13, 4, 11, 0], [8, 13, 9, 0],
                         [7, 5, 11, 0], [0, 12, 12, 0], [0, 10, 0, 0], [0, 0, 0, 0]])
        seq_len = np.array([6, 8, 7, 0])
        features = dstruct.Seq2SeqFeatureTuple(*(data, seq_len, None, None))
        labels = dstruct.SeqLabelTuple(*(None, None, None))
        batch = dstruct.BatchTuple(features, labels, None, False)
        new_batch = generator.get_batch_data(
            batch, data, start_id=1, seq_len_idx=1, input_key='dec_inputs',
            seq_len_key='dec_seq_len', unmasked_token_weight=np.ones_like(data) * 2)
        _f, _l, _n, _k = new_batch
        self.assertIs(data, _f.enc_inputs, 'enc data is the same object.')
        self.assertIs(seq_len, _f.enc_seq_len, 'enc seq len is the same object.')
        np.testing.assert_array_equal(data[:-1, :], _f.dec_inputs[1:, :],
                                      err_msg='dec input is shifted data.')
        np.testing.assert_array_equal(data, _l.label,
                                      err_msg='dec output is data.')
        np.testing.assert_array_equal(seq_len, _f.dec_seq_len,
                                      err_msg='dec seq len is correct.')
        self.assertEqual(sum(seq_len), _n, 'num tokens is correct.')
        w = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0],
                      [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0]])
        np.testing.assert_array_equal(w * 2, new_batch.labels.label_weight,
                                      err_msg='token weight')


class TestSeq(unittest.TestCase):

    def setUp(self):
        data_dir = 'test_data/tiny_single'
        self.gen = partial(generator.read_lines, f'{data_dir}/valid.txt',
                           token_split=' ')
        self.vocab = Vocabulary.from_vocab_file(f'{data_dir}/vocab.txt')
        self.num_lines = 1000
        self.num_tokens = 5606

    def test_read_seq_data_sen(self):
        x, y = generator.read_seq_data(
            self.gen(), self.vocab, self.vocab, keep_sentence=True)
        self.assertEqual(len(x), self.num_lines, 'number of sequences')
        self.assertEqual(len(y), self.num_lines, 'number of sequences')
        for x_, y_ in zip(x, y):
            self.assertEqual(x_[1:], y_[:-1], 'output is shifted input')

    def test_read_seq_data(self):
        x, y = generator.read_seq_data(
            self.gen(), self.vocab, self.vocab, keep_sentence=False, seq_len=20)
        num_seq = (self.num_lines + self.num_tokens) // 20
        if (self.num_lines + self.num_tokens) % 20 > 1:
            num_seq += 1
        self.assertEqual(len(x), num_seq, 'number of sequences')
        self.assertEqual(len(y), num_seq, 'number of sequences')
        for x_, y_ in zip(x, y):
            self.assertEqual(x_[1:], y_[:-1], 'output is shifted input')

    def test_read_seq_data_random(self):
        x, y = generator.read_seq_data(
            self.gen(), self.vocab, self.vocab, keep_sentence=False, random_seq_len=True)
        tokens = 0
        for x_, y_ in zip(x, y):
            tokens += len(y_)
            self.assertEqual(x_[1:], y_[:-1], 'output is shifted input')
        self.assertEqual(tokens, self.num_tokens + self.num_lines, 'number of tokens')

    def test_seq_batch_iter_callable(self):
        data_fn = partial(
            generator.read_seq_data,
            self.gen(), self.vocab, self.vocab, keep_sentence=False, seq_len=20)
        data = [data_fn, None]
        count = 0
        for batch in generator.seq_batch_iter(
                *data, batch_size=13, shuffle=False, keep_sentence=False):
            count += batch.num_tokens
            self.assertTrue(batch.keep_state, 'keep_state is True')
            self.assertEqual(batch.num_tokens, sum(batch.features.seq_len),
                             'num_tokens is sum of seq_len')
        self.assertEqual(count, self.num_lines + self.num_tokens,
                         'number of tokens (including eos symbol)')

    def test_seq_batch_iter(self):
        data = generator.read_seq_data(
            self.gen(), self.vocab, self.vocab, keep_sentence=False, seq_len=20)
        count = 0
        for batch in generator.seq_batch_iter(
                *data, batch_size=13, shuffle=False, keep_sentence=False):
            count += batch.num_tokens
            self.assertTrue(batch.keep_state, 'keep_state is True')
            self.assertEqual(batch.num_tokens, sum(batch.features.seq_len),
                             'num_tokens is sum of seq_len')
        self.assertEqual(count, self.num_lines + self.num_tokens,
                         'number of tokens (including eos symbol)')

    def test_seq_batch_iter_random(self):
        data = generator.read_seq_data(
            self.gen(), self.vocab, self.vocab, keep_sentence=False, random_seq_len=True)
        count = 0
        for batch in generator.seq_batch_iter(
                *data, batch_size=13, shuffle=False, keep_sentence=False):
            count += batch.num_tokens
            self.assertTrue(batch.keep_state, 'keep_state is True')
            self.assertEqual(batch.num_tokens, sum(batch.features.seq_len),
                             'num_tokens is sum of seq_len')
        self.assertEqual(count, self.num_lines + self.num_tokens,
                         'number of tokens (including eos symbol)')


class TestSeq2Seq(unittest.TestCase):

    def setUp(self):
        data_dir = 'test_data/tiny_copy'
        self.gen = partial(generator.read_lines, f'{data_dir}/valid.txt',
                           token_split=' ', part_split='\t', part_indices=[0, -1])
        self.vocab = Vocabulary.from_vocab_file(f'{data_dir}/vocab.txt')
        self.num_lines = 1000
        self.num_tokens = 5463

    def test_read_seq2seq_data(self):
        x, y = generator.read_seq2seq_data(self.gen(), self.vocab, self.vocab,)
        self.assertEqual(len(x), self.num_lines, 'number of sequences')
        self.assertEqual(len(y), self.num_lines, 'number of sequences')
        for x_, y_ in zip(x, y):
            self.assertEqual(x_[:-1], y_[1:-1], 'output is the same as input')

    def test_seq2seq_batch_iter(self):
        data = generator.read_seq2seq_data(self.gen(), self.vocab, self.vocab,)
        count = 0
        for batch in generator.seq2seq_batch_iter(*data, batch_size=5):
            count += batch.num_tokens
            self.assertFalse(batch.keep_state, 'keep_state is False')
            self.assertEqual(batch.num_tokens, sum(batch.features.dec_seq_len),
                             'num_tokens is sum of seq_len')
        self.assertEqual(count, self.num_lines + self.num_tokens,
                         'number of tokens (including eos symbol)')


class TestWord2Def(unittest.TestCase):

    def setUp(self):
        data_dir = 'test_data/tiny_def'
        self.gen = partial(generator.read_lines, f'{data_dir}/valid.txt',
                           token_split=' ', part_split='\t', part_indices=[0, -1])
        self.enc_vocab = Vocabulary.from_vocab_file(f'{data_dir}/enc_vocab.txt')
        self.dec_vocab = Vocabulary.from_vocab_file(f'{data_dir}/dec_vocab.txt')
        self.char_vocab = Vocabulary.from_vocab_file(f'{data_dir}/char_vocab.txt')
        self.num_lines = 1000
        self.num_tokens = 5465

    def test_read_word2def_data(self):
        x, w, c, m, y, sw = generator.read_word2def_data(
            self.gen(), self.enc_vocab, self.dec_vocab, self.char_vocab)
        self.assertEqual(len(x), self.num_lines, 'number of sequences')
        self.assertEqual(len(y), self.num_lines, 'number of sequences')
        self.assertEqual(len(c), self.num_lines, 'number of sequences')
        for x_, w_, c_, sw_ in zip(x, w, c, sw):
            self.assertEqual(''.join(self.char_vocab.i2w(c_[1:-1])),
                             self.enc_vocab.i2w(x_[0]),
                             'characters same as first word in enc data')
            self.assertEqual(''.join(self.char_vocab.i2w(c_[1:-1])),
                             self.enc_vocab.i2w(w_),
                             'characters same as word in word data')
            self.assertEqual(x_[0], w_, 'first enc data same as word data')
            self.assertEqual(sw_, 1, 'seq weight is set to 1')

    def test_read_word2def_data_freq_weight(self):
        x, w, c, m, y, sw = generator.read_word2def_data(
            self.gen(), self.enc_vocab, self.dec_vocab, self.char_vocab,
            freq_down_weight=True)
        for w_, sw_ in zip(w, sw):
            if self.enc_vocab.i2w(w_) == 'eijfceib':
                self.assertEqual(sw_, 1/3, 'seq weight is set to 1/freq')

    def test_word2def_batch_iter(self):
        data = generator.read_word2def_data(
            self.gen(), self.enc_vocab, self.dec_vocab, self.char_vocab)
        count = 0
        for batch in generator.word2def_batch_iter(*data, batch_size=3):
            count += batch.num_tokens
            self.assertFalse(batch.keep_state, 'keep_state is False')
            self.assertEqual(batch.num_tokens, sum(batch.features.dec_seq_len),
                             'num_tokens is sum of seq_len')
            self.assertEqual(batch.features.chars.shape[0], 3,
                             'char data is in batch-major')
            self.assertEqual(batch.features.words.shape, (3, ),
                             'word data is 1D with batch size elements')
        self.assertEqual(count, self.num_lines + self.num_tokens,
                         'number of tokens (including eos symbol)')


class TestReward(unittest.TestCase):

    def test_reward_match_label(self):
        data = np.array([[12, 10, 10, 0], [12, 8, 7, 0], [13, 4, 11, 0], [8, 13, 9, 0],
                         [7, 5, 11, 0], [0, 12, 12, 0], [0, 10, 0, 0], [0, 0, 0, 0]])
        features = dstruct.Seq2SeqFeatureTuple(*(None, None, None, None))
        labels = dstruct.SeqLabelTuple(*(data, None, None))
        batch = dstruct.BatchTuple(features, labels, None, False)
        sample = np.array([[12, 10, 10, 0], [12, 8, 7, 0], [11, 4, 11, 0], [8, 13, 9, 0],
                           [7, 5, 11, 0], [0, 12, 12, 0], [0, 1, 0, 0], [0, 1, 0, 0],
                           [0, 0, 0, 0]])
        exact_match = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0]])
        parti_match = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 1, 0], [1, 1, 1, 0],
                                [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0]])
        sample_len = np.array([6., 9., 7., 0.])
        _sample_len = np.array([6., 9., 7., 1.])  # for division
        m, avg = generator.reward_match_label(sample, batch, partial_match=False)
        np.testing.assert_array_equal(m, exact_match, 'label exact match reward')
        self.assertEqual(avg, 1 / 3,
                         'average correct')
        m, avg = generator.reward_match_label(sample, batch, partial_match=True)
        np.testing.assert_array_equal(m, parti_match / _sample_len,
                                      'label match reward')
        self.assertEqual(avg, np.sum(parti_match / _sample_len) / np.sum(sample_len > 0),
                         'average correct')

    def test_reward_bleu(self):
        def ref_fn(__):
            return [[[1, 2, 3, 4, 0], [3, 4, 5, 6, 7, 0]],
                    [[1, 1, 1, 1, 0]],
                    None,
                    [[1, 3, 4, 7, 0]]]
        sample = np.array([[1, 2, 3, 4, 0], [0, 0, 0, 0, 0],
                           [1, 2, 4, 7, 9], [1, 2, 4, 7, 9]]).T
        scores = np.zeros_like(sample, dtype=np.float32)
        scores[-1, 0] = 1.0
        scores[0, 1] = 0.010890544041151608
        bleu, avg_bleu = generator.reward_bleu(sample, None, ref_fn)
        np.testing.assert_array_equal(bleu, scores, 'BLEU score is correct.')
        self.assertEqual(avg_bleu, (scores[-1, 0] + scores[0, 1]) / 3,
                         'Average score is correct.')

    def test_reward_ngram_lm(self):
        lm = kenlm.Model('test_data/tiny_single/train.arpa')
        vocab = dstruct.Vocabulary.from_vocab_file('test_data/tiny_single/vocab.txt')
        sample = np.arange(1, 13).reshape((4, 3))
        sample[-1, :] = 0
        scores, avg_score = generator.reward_ngram_lm(sample, None, lm, vocab)
        # np.random.randn(*sample.shape))
        np.testing.assert_array_less(scores, 1, err_msg='scores less than 1')
        np.testing.assert_array_less(-1 * scores, 0.0001,
                                     err_msg='scores more than 0')

    def test_count_ngrams(self):
        data = generator.read_lines('test_data/tiny_copy/train.txt', token_split=' ',
                                    part_split='\t', part_indices=(-1,))
        vocab = dstruct.Vocabulary.from_vocab_file('test_data/tiny_copy/vocab.txt')
        count = generator.count_ngrams(data, 3, token_vocab=vocab)
        self.assertEqual(count[(1, 1, 10)], 1018,
                         'check grep "^g " test_data/tiny_copy/train.txt  | wc -l')
        self.assertEqual(count[(6, 5, 5)], 43,
                         'check grep "c b b" test_data/tiny_copy/train.txt  | wc -l')

    def test_reward_global_ngram_stat(self):
        data = generator.read_lines('test_data/tiny_copy/train.txt', token_split=' ',
                                    part_split='\t', part_indices=(-1,))
        vocab = dstruct.Vocabulary.from_vocab_file('test_data/tiny_copy/vocab.txt')
        global_count = generator.count_ngrams(data, 3, token_vocab=vocab)
        sample = np.array([[12, 10, 10, 0], [12, 8, 7, 0], [11, 4, 11, 0], [8, 13, 9, 0],
                           [7, 5, 11, 0], [0, 12, 12, 0], [0, 5, 0, 0], [0, 9, 0, 0],
                           [0, 0, 0, 0]])
        ngram_fn = partial(generator.make_ngrams, n=3, left_pad=(1, 1), right_pad=(0, ))
        current_count = Counter()
        ngram_cache = []

        def update_fn(batch, batch_ngrams):
            if len(ngram_cache) > 0:
                for i in range(len(batch_ngrams)):
                    ngrams = ngram_cache.pop(0)
                    for ngram in ngrams:
                        current_count[ngram] -= 1
            for ngrams in batch_ngrams:
                ngram_cache.append(ngrams)
                for ngram in ngrams:
                    current_count[ngram] += 1

        score, avg_score = generator.reward_global_ngram_stat(
            sample, None, global_count, current_count, update_fn, ngram_fn)
        np.testing.assert_array_less(-1e-6, score,
                                     'score more than 0 (no initial count)')
        score2, avg_score = generator.reward_global_ngram_stat(
            sample, None, global_count, current_count, update_fn, ngram_fn)
        np.testing.assert_array_less(score2 - 1e-6, score,
                                     'score lower once we have stat')
        score3, avg_score = generator.reward_global_ngram_stat(
            sample, None, global_count, current_count, update_fn, ngram_fn)
        np.testing.assert_array_equal(score2, score3,
                                      'score is the same if we update with the same')

if __name__ == '__main__':
    unittest.main()
