"""
Abstract class for data iterator

A data iterator takes in input data, usually a file, and format
it into batches. Each batch is a dictionary with keys corresponding to
feed dictionary. The keys can be in "inputs" or "labels"
"""
import abc

import six
import numpy as np

from seqmodel.bunch import Bunch
from seqmodel.common_tuple import *
from seqmodel.data.vocab import Vocabulary


@six.add_metaclass(abc.ABCMeta)
class BatchIterator(object):
    """
    opt:
        shuffle: If true, shuffle the data.
    """
    def __init__(self, opt=None):
        self.opt = opt
        if self.opt is None:
            self.opt = self.default_opt()

    @staticmethod
    def default_opt():
        return Bunch(
            shuffle=True)

    @property
    def input_keys(self):
        return set()

    @property
    def label_keys(self):
        return set()

    @property
    def batch_size(self):
        if hasattr(self, '_batch_size'):
            return self._batch_size
        else:
            return 1

    @abc.abstractmethod
    def initialize(self):
        """
        Call before using.
        """
        raise NotImplementedError('Not implemented.')

    @abc.abstractmethod
    def init_batch(self, batch_size, no_label_seq=False):
        """
        Prepare data into batches of size batch_size and shuffle the data
        if opt.shuffle is True
        """
        raise NotImplementedError('Not implemented.')

    @abc.abstractmethod
    def next_batch(self):
        """
        Step to the next batch and return the data in
        dictionary format (Bunch). If there is no more data, return None
        """
        raise NotImplementedError('Not implemented.')

    def iterate_epoch(self, batch_size, no_label_seq=False):
        self.init_batch(batch_size, no_label_seq)
        while True:
            batch = self.next_batch()
            if batch is None:
                break
            yield batch


@six.add_metaclass(abc.ABCMeta)
class TextIterator(BatchIterator):
    """
    args:
        opt: Bunch of option, see below
        in_vocab: input Vocabulary
        out_vocab: label Vocabulary
    opt:
        shuffle: If true, shuffle the data.
        add_start_seq: Add start symbol id to the start of each sequence
        add_end_seq: Add end symbol id to the end of each sequence
    """
    def __init__(self, in_vocab, out_vocab, opt=None):
        super(TextIterator, self).__init__(opt)
        self.in_vocab = in_vocab
        self.out_vocab = out_vocab

    @staticmethod
    def default_opt():
        default_opt = BatchIterator.default_opt()
        return Bunch(
            default_opt,
            add_start_seq=False,
            add_end_seq=True)


class RawBatchIterator(BatchIterator):
    def __init__(self, batches, opt=None, batch_iter=None):
        super(RawBatchIterator, self).__init__(opt)
        self.data = batches
        self.iter = batch_iter

    def initialize(self):
        return None

    def init_batch(self, batch_size=None, no_label_seq=False):
        self._pos = 0
        if self.opt.shuffle:
            self._perm = np.random.permutation(len(self.data))
        else:
            self._perm = np.arange(len(self.data))

    def next_batch(self):
        if self._pos < len(self.data):
            batch = self.data[self._perm[self._pos]]
            self._pos += 1
            return batch
        return None
    #
    # def is_all_end(self, batch, outputs):
    #     return self.iter.is_all_end(batch, outputs)
    #
    # def update_last_input(self, batch, outputs, **kwargs):
    #     self.iter.update_last_input(batch, outputs, **kwargs)
    #
    # def format_sample_output(self, batch, samples):
    #     self.iter.format_sample_output(batch, samples)
