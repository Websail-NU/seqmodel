from functools import partial
import unittest

import numpy as np

from seqmodel.dstruct import Vocabulary
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


class TestSeq(unittest.TestCase):

    def setUp(self):
        data_dir = 'test_data/tiny_single'
        self.gen = partial(generator.read_lines, f'{data_dir}/valid.txt',
                           token_split=' ')
        self.vocab = Vocabulary.from_vocab_file(f'{data_dir}/vocab.txt')
        self.num_lines = 1000
        self.num_tokens = 5606

    def test_read_seq_data(self):
        x, y = generator.read_seq_data(self.gen(), self.vocab, self.vocab,
                                       keep_sentence=True)
        self.assertEqual(len(x), self.num_lines, 'number of sequences')
        self.assertEqual(len(y), self.num_lines, 'number of sequences')
        for x_, y_ in zip(x, y):
            self.assertEqual(x_[1:], y_[:-1], 'output is shifted input')

    def test_read_seq_data_sen(self):
        x, y = generator.read_seq_data(self.gen(), self.vocab, self.vocab,
                                       keep_sentence=False, seq_len=20)
        num_seq = (self.num_lines + self.num_tokens) // 20
        if (self.num_lines + self.num_tokens) % 20 > 1:
            num_seq += 1
        self.assertEqual(len(x), num_seq, 'number of sequences')
        self.assertEqual(len(y), num_seq, 'number of sequences')
        for x_, y_ in zip(x, y):
            self.assertEqual(x_[1:], y_[:-1], 'output is shifted input')

    def test_seq_batch_iter(self):
        data = generator.read_seq_data(self.gen(), self.vocab, self.vocab,
                                       keep_sentence=False, seq_len=20)
        count = 0
        for batch in generator.seq_batch_iter(*data, batch_size=13, shuffle=False,
                                              keep_sentence=False):
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
            self.assertEqual(batch.num_tokens, sum(batch.features.decoder.seq_len),
                             'num_tokens is sum of seq_len')
        self.assertEqual(count, self.num_lines + self.num_tokens,
                         'number of tokens (including eos symbol)')


if __name__ == '__main__':
    unittest.main()
