import os
import pickle
import shutil
import fileinput
from itertools import groupby
from itertools import product
from collections import defaultdict
from collections import deque
from functools import partial

import numpy as np
import kenlm

from seqmodel import util as squ
from seqmodel.ngram_stat import *


def _tuple_C(C):
    for k in tuple(C.keys()):
        e = C[k]
        e = tuple(tuple(v) for v in e)
        C[k] = e
    return C


def format_uword_logprob(
        vocab, p_u, p0_u, min_p_count=-1, min_p0_count=-1, min_ratio=-1,
        replace_sos='</s>'):
    min_ratio = -1
    use_p = p_u is not None
    w2i = partial(default_tokens2ids, vocab=vocab, replace_sos=replace_sos)
    p_udist_arr = np.zeros((vocab.vocab_size, ), dtype=np.float32)
    p0_udist_arr = np.zeros((vocab.vocab_size, ), dtype=np.float32)

    if use_p:
        p_udist = get_unigram_logprob(
            p_u, word_set=vocab.word_set(), tokens2ids=w2i,
            num_vocab=vocab.vocab_size)
    p0_udist = get_unigram_logprob(
        p0_u, word_set=vocab.word_set(), tokens2ids=w2i,
        num_vocab=vocab.vocab_size)

    for wid in p0_udist:
        count0, p0 = p0_udist[wid]
        if use_p:
            count, p = p_udist[wid]
            if ((count < min_p_count and count0 < min_p0_count) or
                    abs(p - p0) < min_ratio):
                continue
        else:
            if count0 < min_p0_count:
                continue
        p0_udist_arr[wid] = p0
        if use_p:
            p_udist_arr[wid] = p
    return p_udist_arr, p0_udist_arr


def _safe_log(number, e=1e-5):
    return np.log(number + e)


def format_urep_logprob(
        vocab, p_repk_count, p0_repk_count, p_total_count, p0_total_count,
        max_order, min_p_count=-1, min_p0_count=-1, min_ratio=-1, replace_sos='</s>'):
    min_ratio = -1
    use_p = p_total_count is not None
    w2i = partial(default_tokens2ids, vocab=vocab, replace_sos=replace_sos)
    p_urep_logp = np.zeros((max_order - 1, ), dtype=np.float32)
    p0_urep_logp = np.zeros((max_order - 1, ), dtype=np.float32)
    if p0_total_count is not None:
        for order in range(max_order - 1):
            p0_urep_logp[order] = \
                _safe_log(p0_repk_count[order+1].N()) - _safe_log(p0_total_count[order+1])
            if use_p:
                p_urep_logp[order] = \
                    _safe_log(p_repk_count[order+1].N()) - _safe_log(p_total_count[order+1])  # noqa
    return p_urep_logp, p0_urep_logp


def format_clogprob(
        clogprob_fn, ngram_set_fn, vocab, p_pr, p0_pr, p_fr, p0_fr,
        min_p_count=-1, min_p0_count=-1, min_ratio=-1, replace_sos='</s>'):
    use_p = p_pr is not None and p_fr is not None

    C = {}
    w2i = partial(default_tokens2ids, vocab=vocab, replace_sos=replace_sos)

    ngram_set = set(ngram_set_fn(p0_fr))
    if use_p:
        ngram_set.update(tuple(ngram_set_fn(p_fr)))
        p_clp = clogprob_fn(p_pr, p_fr, ngram_set, w2i, vocab.vocab_size)
    p0_clp = clogprob_fn(p0_pr, p0_fr, ngram_set, w2i, vocab.vocab_size)
    for ngram in p0_clp:
        w, context = ngram
        if isinstance(context, int):
            context = (w, context)
        if context == BLANK:
            continue
        count0, p0 = p0_clp[ngram]
        if use_p:
            count, p = p_clp[ngram]
            if ((count < min_p_count and count0 < min_p0_count) or
                    abs(p - p0) < min_ratio):
                continue
            data = (w, p, p0)
        else:
            if count0 < min_p0_count:
                continue
            data = (w, 0, p0)
        e = C.setdefault(context, ([], [], []))
        for c, v in zip(e, data):
            c.append(v)
    return _tuple_C(C)


def merge_clogprobs(Cs, min_ratio=0.1, average=True, count=None):

    def _idx(x):
        return x[0]

    C = {}
    mappings = {}
    for i_C in Cs:
        for c, data in i_C.items():
            indices, values, values0 = C.setdefault(c, [[], [], []])
            indices.extend(data[0])
            values.extend(data[1])
            values0.extend(data[2])
    avg_C = {}
    for c, data in C.items():
        indices, values, values0 = [], [], []
        for idx, group in groupby(sorted(zip(*data), key=_idx), _idx):
            __, v, v0 = tuple(zip(*group))

            if average:
                if count is None:
                    _count = len(v)
                    _count0 = len(v0)
                else:
                    _count = count
                    _count0 = count
                v = sum(v) / _count
                v0 = sum(v0) / _count0
            else:
                v = sum(v)
                v0 = sum(v0)
            if abs(v - v0) >= min_ratio or (v == 0 and v0 < 0):
                indices.append(idx)
                values.append(v)
                values0.append(v0)
        if indices:
            avg_C[c] = (tuple(indices), tuple(values), tuple(values0))
    return avg_C


def get_constraint_keys(i, j, inputs, num_prev_words):
    keys = []
    for order in range(num_prev_words):
        context = inputs[(i-order):(i+1), j]
        key = tuple(context)
        if not key:
            break
        keys.append(key)
        keys.append((context[0], -len(context)))
    return tuple(keys)


def _default_e():
    return [0.0, 0.0]


def get_sparse_scalers(inputs, weights, C, max_order=2, max_num=2000):
    num_prev_words = max_order - 1
    cache = {}
    indices, values, values0 = [], [], []  # parallel list
    for i, j in product(range(inputs.shape[0]), range(inputs.shape[1])):
        if weights[i, j] == 0:
            continue
        keys = get_constraint_keys(i, j, inputs, num_prev_words)
        if len(keys) == 0:
            continue
        ij_e = cache.get(keys, None)
        if ij_e is None:
            ij_e = defaultdict(_default_e)
            for key in keys:
                p = C.get(key, None)
                if p is None:
                    continue
                choices = np.arange(len(p[0]))
                if max_num > -1 and len(p[0]) > max_num:
                    vals = np.array(p[1])
                    vals0 = np.array(p[2])
                    scores = np.abs(vals - vals0)
                    scores = scores / np.sum(scores)
                    choices = np.random.choice(
                        choices, size=max_num, replace=False, p=scores)
                for c in choices:
                    val = ij_e[p[0][c]]
                    val[0] += p[1][c]
                    val[1] += p[2][c]
            cache[keys] = ij_e
        for k in sorted(ij_e):
            indices.append((i, j, k))
            val, val0 = ij_e[k]
            values.append(val)
            values0.append(val0)
    if len(indices) == 0:
        indices = np.array([(0, 0, 0)], dtype=np.int32)
        values = np.array([0], dtype=np.float32)
        values0 = np.array([0], dtype=np.float32)
        return indices, values, values0
    indices = np.array(indices, dtype=np.int32)
    values = np.array(values, dtype=np.float32)
    values0 = np.array(values0, dtype=np.float32)
    return indices, values, values0


def precompute_constraint(temp_C_path, max_order, max_num, batches):
    with open(temp_C_path, 'rb') as ifp:
        C = pickle.load(ifp)
    batch_Cs = []
    for b in batches:
        batch_Cs.append(get_sparse_scalers(
            b.features[-2], b.labels.label_weight, C,
            max_order=max_order, max_num=max_num))
    return batch_Cs


class GNS(object):

    def __init__(self, gns_opt, pool, vocab, vocab_filepath, decode_fn):
        self.use_lm, self.use_rep = gns_opt['use_lm'], gns_opt['use_rep']
        self.use_repm = gns_opt['use_repm']
        assert self.use_lm or self.use_rep or self.use_repm,\
            'use_lm or use_rep should be true.'
        self.vocab = vocab
        self.vocab_filepath = vocab_filepath
        self.opt = gns_opt
        self.pool = pool
        self.decode_fn = decode_fn
        self.local_dec_history = deque(maxlen=gns_opt['text_history_size'])
        self.cur_dec_path = None
        self.cur_p_stat = None
        self.C_history = deque(maxlen=gns_opt['avg_C_size'])
        self.U_history = deque(maxlen=gns_opt['avg_unigram_size'])
        self.UR_history = deque(maxlen=gns_opt['average_repk_size'])
        self.cur_C = None
        self.p0_stat = self._build_ngram_stat(gns_opt['ref_text_path'])
        self._precompute_batch_constraint = partial(
            precompute_constraint,
            gns_opt['temp_C_path'], gns_opt['ngram_max_order'],
            gns_opt['num_constraints_per_token'])
        self.train_batches = []
        self.C_batches = []
        if gns_opt['use_model_prob']:
            self.update_estimate_stat = self._fake_estimate_stat
        else:
            self.update_estimate_stat = self._update_estimate_stat

    @classmethod
    def default_gns_exp_opt(cls):
        opt = {'dec_batch_size': 32, 'num_processes': 6, 'num_chunks_per_process': 4,
               'precompute_after_steps': -1, 'percent_new_tokens': -1.0,
               'ngram_max_order': 2, 'ngram_min_order': 2,
               'min_p0_count': 2, 'min_p_count': 2,
               'use_lm': False, 'use_rep': False, 'use_repm': False,
               'remove_unk': False, 'remove_sen': False, 'replace_sos': '</s>',
               'dec_total_tokens': 929589, 'loss_temperature': 1.0,
               'clip_ratio': 2.0, 'min_ratio': 0.1, 'num_constraints_per_token': -1,
               'ref_text_path': 'data/ptb/train.txt',
               'dec_temperature': 1.0, 'avg_C_size': 50, 'text_history_size': 25,
               'avg_unigram_size': 50, 'average_repk_size': 50,
               'use_model_prob': False, 'alpha': 1.0, 'uniq_dec_text': False,
               'add_unigram_kld': False, 'add_repk_kld': False, 'full_average': False}
        return {f'gns:{k}': v for k, v in opt.items()}

    def _build_ngram_stat(self, text_filepath):
        max_order = self.opt['ngram_max_order']
        min_order = self.opt['ngram_min_order']
        out_filepath = os.path.splitext(text_filepath)[0]
        count_filepath = SRILM_ngram_count(
            text_filepath, out_filepath, self.vocab_filepath,
            max_order=max_order)
        # ucount = read_ngram_count_file(
        #     count_filepath, min_order=1, max_order=1)
        # ucount = get_unigram_count(ucount)
        # ucount['</s>'] = file_len(text_filepath)
        count = read_ngram_count_file(
            count_filepath, max_order=max_order,
            remove_unk=self.opt['remove_unk'], remove_sentence=self.opt['remove_sen'])
        ucount = get_unigram_count(count)
        lm, repk_count, repk_dist, total_count = None, None, None, None
        if self.use_lm:
            lm_filepaths = SRILM_ngram_lm(
                text_filepath, out_filepath, self.vocab_filepath, interpolate=True,
                min_order=min_order, max_order=max_order)
            lm = read_ngram_lm_files(lm_filepaths)
        if self.use_rep:
            repk_count, total_count = get_repk_count(count)
            repk_dist = ConditionalProbDist(
                repk_count, WittenBellProbDist, self.vocab.vocab_size)
        if self.use_repm:
            repk_count = get_margin_count(count)
            repk_dist = ConditionalProbDist(
                repk_count, WittenBellProbDist, self.vocab.vocab_size)
        filtered_count = filter_ngram_count(
            count, min_count=self.opt['min_p_count'],
            min_order=min_order,
            max_order=max_order)
        return ucount, filtered_count, lm, repk_count, repk_dist, total_count

    def _fake_estimate_stat(self, epoch=0, step=0):
        self.cur_p_stat = None
        text_filename = f'ep{epoch:02d}.{step:04d}.txt'
        opath = self.decode_fn(text_filename, 1)
        self.cur_dec_path = opath
        return self.cur_p_stat, 1.0

    def _update_estimate_stat(self, epoch=0, step=0):
        pc_rate = self.opt['precompute_after_steps']
        total_dec_tokens = self.opt['dec_total_tokens']
        percent_new_tokens = self.opt['percent_new_tokens']
        if ((pc_rate > -1 and step % pc_rate != 0) or  # not at the step
                (pc_rate == -1 and step != 0)):  # epoch setting
            num_tokens = 0
        elif (self.cur_dec_path is None or  # initial decoding
                pc_rate == -1 or  # epoch based setting
                (percent_new_tokens == -1 and step == 0)):  # precompute > dec
            num_tokens = total_dec_tokens
        else:
            num_tokens = int(total_dec_tokens * percent_new_tokens)
        if num_tokens > 0:
            text_filename = f'ep{epoch:02d}.{step:04d}.txt'
            opath = self.decode_fn(text_filename, num_tokens)
            local_opath = os.path.splitext(opath)[0] + '.local'
            shutil.copy(opath, local_opath, follow_symlinks=True)
            if num_tokens < total_dec_tokens:
                self.local_dec_history.append(local_opath)
            with open(opath, 'a') as ofp:
                num_text_chucks = int(1 / percent_new_tokens)
                if (len(self.local_dec_history) >= num_text_chucks and
                        percent_new_tokens > -1):
                    fchoices = np.random.choice(
                        range(len(self.local_dec_history)), num_text_chucks - 1,
                        replace=False)
                    selected_files = (self.local_dec_history[fc] for fc in fchoices)
                    with fileinput.input(files=selected_files) as lines:
                        for line in lines:
                            ofp.write(line)
                elif self.cur_dec_path is not None and percent_new_tokens > -1:
                    max_copy = int(total_dec_tokens * (1 - percent_new_tokens))
                    num_copied = 0
                    with open(self.cur_dec_path, 'r') as ifp:
                        for line in ifp:
                            ofp.write(line)
                            num_copied += len(line.strip().split()) + 1
                            if num_copied >= max_copy:
                                break
            if self.opt['uniq_dec_text']:
                subprocess.call(['script/sortuniq.sh', opath])
            self.cur_dec_path = opath
            updated = True
            self.cur_p_stat = self._build_ngram_stat(opath)
        return self.cur_p_stat, num_tokens

    def update_C(self, p_stat, step, pickle_path=None):
        p0_ucount, p0_count, p0_lm, p0_repk_count, p0_repk_dist, p0_total_count =\
            self.p0_stat
        pc_rate = self.opt['precompute_after_steps']
        if ((pc_rate == -1 and step != 0) or
                (pc_rate > -1 and step % pc_rate != 0) or
                (self.opt['use_model_prob'] and self.cur_C is not None)):
            return self.cur_C, self.cur_p_u, self.cur_p0_u, self.cur_p_urep, self.cur_p0_urep  # noqa
        if self.opt['use_model_prob']:
            p_ucount, p_count, p_total_count = None, None, None
            p_lm, p_repk_count, p_repk_dist = None, None, None
        else:
            p_ucount, p_count, p_lm, p_repk_count, p_repk_dist, p_total_count = p_stat

        format_kwargs = {'replace_sos': self.opt['replace_sos'],
                         'min_p_count': self.opt['min_p_count'],
                         'min_p0_count': self.opt['min_p0_count']}
        p_u, p0_u = format_uword_logprob(
            self.vocab, p_ucount, p0_ucount, **format_kwargs)
        p_urep, p0_urep = format_urep_logprob(
            self.vocab, p_repk_count, p0_repk_count, p_total_count, p0_total_count,
            max_order=self.opt['ngram_max_order'], **format_kwargs)
        self.cur_p_u = p_u
        self.cur_p0_u = p0_u
        self.cur_p_urep = p_urep
        self.cur_p0_urep = p0_urep
        C_lm, C_rep = None, None
        if self.use_lm:
            C_lm = format_clogprob(
                get_lm_cond_logprob, get_ngrams, self.vocab,
                p_lm, p0_lm, p_count, p0_count, **format_kwargs)
        if self.use_rep:
            C_rep = format_clogprob(
                get_repk_cond_logprob_cpdist, get_repk_conditions, self.vocab,
                p_repk_dist, p0_repk_dist,
                p_repk_count, p0_repk_count, **format_kwargs)
        if self.use_repm:
            C_rep = format_clogprob(
                get_rep_cond_logprob_cpdist, get_ngrams, self.vocab,
                p_repk_dist, p0_repk_dist,
                p_repk_count, p0_repk_count, **format_kwargs)
        C = None
        if C_rep is not None and C_lm is not None:
            C = merge_clogprobs([C_lm, C_rep], average=False)
        elif C_rep is not None:
            C = C_rep
        elif C_lm is not None:
            C = C_lm
        C, p_u, p_urep = self._update_C(C, p_u, p_urep)
        if pickle_path is None:
            pickle_path = os.path.splitext(self.cur_dec_path)[0] + '.pkl'
        with open(pickle_path, 'wb') as ofp:
            pickle.dump(C, ofp)
        temp_C_path = self.opt['temp_C_path']
        if os.path.lexists(temp_C_path):
            os.remove(temp_C_path)
        os.symlink(os.path.abspath(pickle_path), temp_C_path)
        return C, p_u, p0_u, p_urep, p0_urep

    def _update_C(self, C, p_u, p_urep):
        maxlen = self.opt['avg_C_size']
        self.C_history.append(C)
        self.cur_C = merge_clogprobs(
            self.C_history, min_ratio=self.opt['min_ratio'],
            average=True, count=len(self.C_history))
        self.U_history.append(p_u)
        stack_u = np.stack(self.U_history)
        self.UR_history.append(p_urep)
        avg_u = stack_u.mean(axis=0)
        stack_ur = np.stack(self.UR_history)
        avg_ur = stack_ur.mean(axis=0)
        return self.cur_C, avg_u, avg_ur

    def update_C_batches(self, C, epoch=0, step=0):
        pc_rate = self.opt['precompute_after_steps']
        if pc_rate == -1 and step != 0:
            return -1, -1
        if pc_rate > -1 and step % pc_rate != 0:
            return -1, -1
        if pc_rate != -1:
            start = step
            end = min(step + pc_rate, len(self.train_batches))
        else:
            start = 0
            end = len(self.train_batches)
        assert len(self.C_batches) == 0, 'train_batches and C_batches are out of sync.'
        num_chunks = self.opt['num_processes'] * self.opt['num_chunks_per_process']
        par_data = squ.chunks(self.train_batches[start: end], num_chunks)
        for cbat in self.pool.map(self._precompute_batch_constraint, par_data):
            self.C_batches.extend(cbat)
        return start, end

    def update_train_batches(self, batch_iter):
        del self.train_batches[:]
        for batch in batch_iter():
                self.train_batches.append(batch)

    def train_batch_iter(self):
        for b in self.train_batches:
            yield b

    def cur_C_batch(self, *args, **kwargs):
        return self.C_batches.pop(0)
