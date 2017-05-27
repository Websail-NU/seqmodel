from functools import partial
import unittest

import numpy as np

from seqmodel import util


class TestUtil(unittest.TestCase):

    def test_dict_with_key_startswith(self):
        people = {'biggus_': 'dickus', 'incontinentia_': 'buttocks',
                  'jew:jesus': 'christ', 'jew:brian': 'cohen'}
        jew = {'jesus': 'christ', 'brian': 'cohen'}
        jew_ = util.dict_with_key_startswith(people, 'jew:')
        self.assertEqual(jew, jew_, 'filter and remove prefix')

    def test_dict_with_key_endswith(self):
        people = {'biggus_': 'dickus', 'incontinentia_': 'buttocks',
                  'jew:jesus': 'christ', 'jew:brian': 'cohen'}
        roman = {'biggus': 'dickus', 'incontinentia': 'buttocks'}
        roman_ = util.dict_with_key_endswith(people, '_')
        self.assertEqual(roman, roman_, 'filter and remove suffix')

    def test_get_with_dot_key(self):
        x, n, o, b = range(4)
        y = {'n': n, 'o': o}
        a = {'x': x, 'y': y}
        d = {'a': a, 'b': b}
        get = partial(util.get_with_dot_key, d)
        self.assertEqual(a, get('a'))
        self.assertEqual(b, get('b'))
        self.assertEqual(x, get('a.x'))
        self.assertEqual(y, get('a.y'))
        self.assertEqual(n, get('a.y.n'))
        self.assertEqual(o, get('a.y.o'))

    def test_hstack_list(self):
        inputs = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11], []]
        targets = np.array([[1, 2, 3, 4, 0],
                            [5, 6, 0, 0, 0],
                            [7, 8, 9, 10, 11],
                            [0, 0, 0, 0, 0]], dtype=np.int32).T
        outputs, lengths = util.hstack_list(inputs, padding=0, dtype=np.int32)
        self.assertTrue(np.all(outputs == targets), 'data is correct')
        self.assertTrue(np.all(lengths == np.array(list(map(len, inputs)),
                                                   dtype=np.int32)), 'length is correct')

    def test_vstack_list(self):
        inputs = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11], []]
        targets = np.array([[1, 2, 3, 4, 0],
                            [5, 6, 0, 0, 0],
                            [7, 8, 9, 10, 11],
                            [0, 0, 0, 0, 0]], dtype=np.int32)
        outputs, lengths = util.vstack_list(inputs, padding=0, dtype=np.int32)
        self.assertTrue(np.all(outputs == targets), 'data is correct')
        self.assertTrue(np.all(lengths == np.array(list(map(len, inputs)),
                                                   dtype=np.int32)), 'length is correct')

    def test_masked_full_like(self):
        data = np.random.randn(4, 3)
        num_non_padding = np.array([4, 1, 0])
        outputs, total = util.masked_full_like(data, 1, num_non_padding=num_non_padding,
                                               padding=0, dtype=np.float32)
        self.assertEqual(total, np.sum(num_non_padding), 'num padding is correct')
        self.assertTrue(np.all(np.sum(outputs, axis=0) == num_non_padding),
                        'num padding is correct')
        self.assertEqual(data.shape, outputs.shape, 'output has the same shape')
        self.assertTrue(np.all(outputs[:, -1] == 0), 'padding value is correct')
