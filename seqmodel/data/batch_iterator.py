"""
Abstract class for data iterator

A data iterator takes in input data, usually a file, and format
it into batches. Each batch is a dictionary with keys corresponding to
feed dictionary. The keys can be in "inputs" or "labels"
"""
import abc

import six

from seqmodel.bunch import Bunch
from seqmodel.data.vocab import Vocabulary


@six.add_metaclass(abc.ABCMeta)
class BatchIterator(object):
    """
    opt:
        shuffle: If true, shuffle the data.
    """
    def __init__(self, opt):
        self.opt = opt

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
        return 1

    @abc.abstractmethod
    def initialize(self, **kwargs):
        """
        Call before using.
        """
        raise NotImplementedError('Not implemented.')

    @abc.abstractmethod
    def init_batch(self, batch_size):
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

    @abc.abstractmethod
    def update_last_input(self, batch, input, **kwargs):
        raise NotImplementedError

    def iterate_epoch(self, batch_size):
        self.init_batch(batch_size)
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
    def __init__(self, opt, in_vocab, out_vocab):
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

    def is_all_end(self, outputs):
        """ Return True, if all elements in the outputs is "end_seq" symbol """
        return False
