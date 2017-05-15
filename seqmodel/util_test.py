from functools import partial
import unittest

import numpy as np

from seqmodel import util


class TestUtil(unittest.TestCase):

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
