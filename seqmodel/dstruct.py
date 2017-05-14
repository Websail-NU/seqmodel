import json
import copy
import collections
import codecs
import six


# ######## ##     ## ########  ##       ########  ######
#    ##    ##     ## ##     ## ##       ##       ##    ##
#    ##    ##     ## ##     ## ##       ##       ##
#    ##    ##     ## ########  ##       ######    ######
#    ##    ##     ## ##        ##       ##             ##
#    ##    ##     ## ##        ##       ##       ##    ##
#    ##     #######  ##        ######## ########  ######


BatchTuple = collections.namedtuple(
    'BatchTuple', ('features', 'labels', 'num_tokens', 'keep_state'))

SeqFeatureTuple = collections.namedtuple(
    'SeqFeatureTuple', ('inputs', 'seq_len'))
SeqLabelTuple = collections.namedtuple(
    'SeqLabelTuple', ('label', 'label_weight', 'seq_weight'))


Seq2SeqFeatureTuple = collections.namedtuple(
    'Seq2SeqFeatureTuple', ('encoder', 'decoder'))


# ##     ##  #######   ######     ###    ########
# ##     ## ##     ## ##    ##   ## ##   ##     ##
# ##     ## ##     ## ##        ##   ##  ##     ##
# ##     ## ##     ## ##       ##     ## ########
#  ##   ##  ##     ## ##       ######### ##     ##
#   ## ##   ##     ## ##    ## ##     ## ##     ##
#    ###     #######   ######  ##     ## ########


class Vocabulary(object):

    special_symbols = {'end_seq': '</s>', 'start_seq': '<s>',
                       'end_encode': '</enc>', 'unknown': '<unk>'}

    def __init__(self):
        self._w2i = {}
        self._i2w = []
        self._i2freq = {}
        self._vocab_size = 0

    @property
    def vocab_size(self):
        return self._vocab_size

    def add(self, word, count):
        self._w2i[word] = self._vocab_size
        self._i2w.append(word)
        self._i2freq[self._vocab_size] = count
        self._vocab_size += 1

    def w2i(self, word):
        if isinstance(word, six.string_types):
            if self.special_symbols['unknown'] in self._w2i:
                unk_id = self._w2i[self.special_symbols['unknown']]
                return self._w2i.get(word, unk_id)
            else:
                return self._w2i[word]
        if isinstance(word, collections.Iterable):
            return [self.w2i(_w) for _w in word]

    def i2w(self, index):
        if isinstance(index, six.string_types):
            raise ValueError(
                ('index must be an integer, recieved `{}`. '
                 'Call `w2i()` for converting word to id').format(index))
        if isinstance(index, collections.Iterable):
            return [self.i2w(_idx) for _idx in index]
        return self._i2w[index]

    def word_set(self):
        return set(self._w2i.keys())

    @staticmethod
    def from_vocab_file(filepath, special_symbols=None):
        vocab = Vocabulary()
        with codecs.open(filepath, 'r', 'utf-8') as ifp:
            for line in ifp:
                parts = line.strip().split()
                count = 0
                word = parts[0]
                if len(parts) > 1:
                    count = int(parts[1])
                vocab.add(word, count)
        return vocab

    @staticmethod
    def vocab_index_map(vocab_a, vocab_b):
        a2b = {}
        b2a = {}
        for w in vocab_a.word_set():
            a2b[vocab_a.w2i(w)] = vocab_b.w2i(w)
        for w in vocab_b.word_set():
            b2a[vocab_b.w2i(w)] = vocab_a.w2i(w)
        return a2b, b2a

    @staticmethod
    def list_ids_from_file(filepath, vocab):
        output = []
        with codecs.open(filepath, 'r', 'utf-8') as ifp:
            for line in ifp:
                word = line.strip().split()[0]
                output.append(vocab.w2i(word))
        return output

    @staticmethod
    def create_vocab_mask(keep_vocab, full_vocab):
        mask = np.zeros(full_vocab.vocab_size)
        for w in keep_vocab.word_set():
            mask[full_vocab.w2i(w)] = 1
        return mask
