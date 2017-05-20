import unittest

from seqmodel import dstruct


class TestVocab(unittest.TestCase):

    def test_create_vocab(self):
        vocab = dstruct.Vocabulary.from_vocab_file('test_data/tiny_copy/vocab.txt')
        self.assertEqual(len(vocab.word_set()), 14, 'num vocab equals to num line.')
        self.assertEqual(vocab.vocab_size, 14, 'num vocab equals to num line.')
        for i in range(97, 97 + 10):
            self.assertEqual(vocab.i2w(i - 93), chr(i), 'index map in order of file')

    def test_map(self):
        vocab = dstruct.Vocabulary.from_vocab_file('test_data/tiny_copy/vocab.txt')
        self.assertEqual(vocab.i2w(4), 'a')
        self.assertEqual(vocab.w2i('</s>'), 0)
        idx = [[0, 3, 4], [5, 6, 7]]
        words = [['</s>', '<unk>', 'a'], ['b', 'c', 'd']]
        idx_ = vocab.w2i(words)
        words_ = vocab.i2w(idx)
        self.assertEqual(idx, idx_, 'i2w from iterable')
        self.assertEqual(words, words_, 'w2i from iterable')
