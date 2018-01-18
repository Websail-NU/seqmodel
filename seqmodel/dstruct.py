import json
import copy
import collections
import codecs
import six
import time
import math


__all__ = ['BatchTuple', 'SeqFeatureTuple', 'SeqLabelTuple', 'Seq2SeqFeatureTuple',
           'OutputStateTuple', 'IndexScoreTuple', 'Vocabulary', 'TrainingState',
           'RunningInfo', 'RunSamplingInfo', 'Word2DefFeatureTuple',
           'LSeq2SeqFeatureTuple']

########################################################
#    ######## ##     ## ########  ##       ########    #
#       ##    ##     ## ##     ## ##       ##          #
#       ##    ##     ## ##     ## ##       ##          #
#       ##    ##     ## ########  ##       ######      #
#       ##    ##     ## ##        ##       ##          #
#       ##    ##     ## ##        ##       ##          #
#       ##     #######  ##        ######## ########    #
########################################################


BatchTuple = collections.namedtuple(
    'BatchTuple', ('features', 'labels', 'num_tokens', 'keep_state'))

SeqFeatureTuple = collections.namedtuple('SeqFeatureTuple', ('inputs', 'seq_len'))
SeqLabelTuple = collections.namedtuple(
    'SeqLabelTuple', ('label', 'label_weight', 'seq_weight'))

# always arrange enc before dec (help zipping placeholder during decoding)
Seq2SeqFeatureTuple = collections.namedtuple(
    'Seq2SeqFeatureTuple', ('enc_inputs', 'enc_seq_len', 'dec_inputs', 'dec_seq_len'))

LSeq2SeqFeatureTuple = collections.namedtuple(
    'LSeq2SeqFeatureTuple', ('enc_inputs', 'enc_seq_len',
                             'dec_inputs', 'dec_seq_len', 'label', 'mask'))

Word2DefFeatureTuple = collections.namedtuple(
    'Word2DefFeatureTuple', ('enc_inputs', 'enc_seq_len', 'words', 'chars', 'char_len',
                             'word_masks', 'dec_inputs', 'dec_seq_len'))

OutputStateTuple = collections.namedtuple('OutputStateTuple', ('output', 'state'))

IndexScoreTuple = collections.namedtuple('IndexScoreTuple', ('index', 'score'))

##########################################################
#    ##     ##  #######   ######     ###    ########     #
#    ##     ## ##     ## ##    ##   ## ##   ##     ##    #
#    ##     ## ##     ## ##        ##   ##  ##     ##    #
#    ##     ## ##     ## ##       ##     ## ########     #
#     ##   ##  ##     ## ##       ######### ##     ##    #
#      ## ##   ##     ## ##    ## ##     ## ##     ##    #
#       ###     #######   ######  ##     ## ########     #
##########################################################


class Vocabulary(object):

    special_symbols = {
        'end_seq': '</s>', 'start_seq': '<s>', 'end_encode': '</enc>',
        'unknown': '<unk>'}

    def __init__(self):
        self._w2i = {}
        self._i2w = []
        self._i2freq = {}
        self._vocab_size = 0

    def __getitem__(self, arg):
        if isinstance(arg, six.string_types):
            return self._w2i[arg]
        elif isinstance(arg, int):
            return self._i2w[arg]
        else:
            raise ValueError('Only support either integer or string')

    @property
    def vocab_size(self):
        return self._vocab_size

    def add(self, word, count):
        self._w2i[word] = self._vocab_size
        self._i2w.append(word)
        self._i2freq[self._vocab_size] = count
        self._vocab_size += 1

    def w2i(self, word, unk_id=None):
        if isinstance(word, six.string_types):
            if unk_id is None and self.special_symbols['unknown'] in self._w2i:
                unk_id = self._w2i[self.special_symbols['unknown']]
            if unk_id is not None:
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

    def __len__(self):
        return self.vocab_size

    @staticmethod
    def from_vocab_file(filepath):
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


##########################################################################
#    ########  ##     ## ##    ##    #### ##    ## ########  #######     #
#    ##     ## ##     ## ###   ##     ##  ###   ## ##       ##     ##    #
#    ##     ## ##     ## ####  ##     ##  ####  ## ##       ##     ##    #
#    ########  ##     ## ## ## ##     ##  ## ## ## ######   ##     ##    #
#    ##   ##   ##     ## ##  ####     ##  ##  #### ##       ##     ##    #
#    ##    ##  ##     ## ##   ###     ##  ##   ### ##       ##     ##    #
#    ##     ##  #######  ##    ##    #### ##    ## ##        #######     #
##########################################################################


class TrainingState(object):
    def __init__(
            self, learning_rate=1e-4, cur_epoch=0, cur_eval=float('inf'),
            last_imp_eval=float('inf'), best_eval=float('inf'), best_epoch=-1,
            last_imp_epoch=-1, imp_wait=0, best_checkpoint_epoch=-1):
        self.learning_rate = learning_rate
        self.cur_epoch = cur_epoch
        self.cur_eval = cur_eval
        self.last_imp_eval = last_imp_eval
        self.best_eval = best_eval
        self.best_epoch = best_epoch
        self.last_imp_epoch = last_imp_epoch
        self.imp_wait = imp_wait
        self.best_checkpoint_epoch = best_checkpoint_epoch

    def summary(self, mode='train'):
        return f'ep: {self.cur_epoch}, lr: {self.learning_rate:.6f}'

    def update_epoch(self, info):
        cur_eval = info.eval_loss
        if self.best_eval > cur_eval:
            self.best_eval = cur_eval
            self.best_epoch = self.cur_epoch
        self.cur_epoch += 1
        self.cur_eval = cur_eval

    def __str__(self):
        return f'{self.__class__.__name__}: {vars(self)}'


class RunningInfo(object):
    def __init__(
            self, start_time=None, end_time=None, eval_loss=0.0, train_loss=0.0,
            num_tokens=0, step=0):
        self._start_time = start_time or time.time()
        self._end_time = end_time
        self._eval_loss = eval_loss
        self._train_loss = train_loss
        self._num_tokens = num_tokens
        self._step = step

    @property
    def eval_loss(self):
        return self._eval_loss / self._num_tokens

    @property
    def train_loss(self):
        return self._train_loss / self._step

    @ property
    def num_tokens(self):
        return self._num_tokens

    @ property
    def step(self):
        return self._step

    @property
    def wps(self):
        return self._num_tokens / self.elapse_time

    @property
    def elapse_time(self):
        end_time = self._end_time
        if end_time is None:
            end_time = time.time()
        return end_time - self._start_time

    def update_step(self, result, num_tokens):
        if 'train_loss' in result:
            self._train_loss += result['train_loss']
        if 'eval_loss' in result:
            self._eval_loss += result['eval_loss'] * num_tokens
        self._num_tokens += num_tokens
        self._step += 1

    def end(self):
        self._end_time = time.time()

    def summary(self, mode='train'):
        exp_loss = math.exp(self.eval_loss)
        if mode == 'train':
            return (f'(T) eval_loss: {self.eval_loss:.5f} ({exp_loss:.5f}), '
                    f'tr_loss: {self.train_loss:.5f}, '
                    f'wps: {self.wps:.1f}, {self._step} steps in '
                    f'{self.elapse_time:.2f}s')
        else:
            return (f'(E) eval_loss: {self.eval_loss:.5f} ({exp_loss:.5f}), '
                    f'wps: {self.wps:.1f}, {self._step} steps in '
                    f'{self.elapse_time:.2f}s')

    def __str__(self):
        return f'{self.__class__.__name__}: {vars(self)}'


class RunSamplingInfo(RunningInfo):

    @property
    def eval_loss(self):
        return -1 * self._eval_loss / self._step

    def summary(self, mode='train'):
        exp_loss = math.exp(self.eval_loss)
        if mode == 'train':
            return (f'(T) avg_reward: {-1 * self.eval_loss:.5f}, '
                    f'tr_loss: {self.train_loss:.5f}, '
                    f'wps: {self.wps:.1f}, {self._step} steps in '
                    f'{self.elapse_time:.2f}s')
        else:
            return (f'(E) avg_reward: {-1 * self.eval_loss:.5f}, '
                    f'wps: {self.wps:.1f}, {self._step} steps in '
                    f'{self.elapse_time:.2f}s')

    def update_step(self, avg_reward, num_tokens, train_result=None):
        if train_result is not None and 'train_loss' in train_result:
            self._train_loss += train_result['train_loss']
        self._eval_loss += avg_reward
        self._num_tokens += num_tokens
        self._step += 1
        # if self._step % 100 == 0:
        #     print(self.summary())

    def __str__(self):
        return f'{self.__class__.__name__}: {vars(self)}'
