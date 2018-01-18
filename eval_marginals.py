import os
import sys
import json
import pickle
from collections import defaultdict
from collections import ChainMap
from functools import partial

import numpy as np
import tensorflow as tf

import seqmodel as sq

batch_size = 128
q_mode = sys.argv[1]
if "sample" in q_mode:
    M = int(q_mode[6:])
    q_mode = "sample"

# data_dir = 'data/tinyshakespeare/'
# eval_file = f'{data_dir}/train.char.filter.txt'
# count_file = f'{data_dir}/train.char.filter.count'
# total_tokens = 1016242
# eval_file = 'curexp/ts-v/sample2.filter.txt'
# count_file = 'curexp/ts-v/sample2.filter.count'
# total_tokens = 4000000

# data_dir = 'data/ptb/'
# eval_file = 'curexp/ptb-v/train.count-filter-text'
# count_file = 'curexp/ptb-v/train.count-filter'
# trace_nll_file = 'curexp/ptb-v/train_nll.txt'
# total_tokens = 887521 + 42068

data_dir = 'data/ptb/'
eval_file = 'curexp/ptb-v/sample.count-filter-text'
count_file = 'curexp/ptb-v/sample.count-filter'
trace_nll_file = 'curexp/ptb-v/sample_nll.txt'

gmm_dir = 'curexp/ptb-v'
gmm_file = 'gmm64valid_model.pkl'
model_dir = 'curexp/ptb-cde-direct-only_native'
state_sample_file = 'sample_states.npy'

# data
print('loading data...')
ngram_counts = sq.ngram_stat.read_ngram_count_file(count_file)
dpath = partial(os.path.join, data_dir)
vocab = sq.Vocabulary.from_vocab_file(dpath('vocab.txt'))
ngram_lines = sq.read_lines(eval_file, token_split=' ')
data = sq.read_seq_data(
    ngram_lines, vocab, vocab, keep_sentence=True,
    data_include_eos=True, add_sos=False)
batches = partial(
    sq.seq_batch_iter, *data, batch_size=batch_size, shuffle=True, keep_sentence=True)

# gmm
gmm_path = partial(os.path.join, gmm_dir)
with open(gmm_path(gmm_file), mode='rb') as gmm_pickle:
    gmm = pickle.load(gmm_pickle)
_a = np.load(gmm_path(state_sample_file))
sampled_states = np.transpose(_a, (1, 0, 2)).reshape((-1, _a.shape[-1]))
sampled_states = sampled_states[~np.all(sampled_states == 0, axis=-1)]
get_random_state_ids = partial(np.random.choice, np.arange(len(sampled_states)))

# trace nll
tokens = []
nll_iix = {}
nll_data = []
with open(trace_nll_file) as lines:
    for i, line in enumerate(lines):
        word, nll = line.strip().split('\t')
        tokens.append(word)
        nll = float(nll)
        nll_data.append(nll)
        positions = nll_iix.setdefault(word, [])
        positions.append(i)
for k, v in nll_iix.items():
    nll_iix[k] = np.array(v)
total_tokens = len(tokens)
# model
print('loading model...')
epath = partial(os.path.join, model_dir)
with open(epath('model_opt.json')) as ifp:
    model_opt = json.load(ifp)
model_vocab_opt = sq.AESeqModel.get_vocab_opt(*(v.vocab_size for v in [vocab, vocab]))
model_opt = ChainMap(
    {'out:token_nll': True, 'out:eval_first_token': True}, model_vocab_opt, model_opt)
q_inactive = q_mode in ('zero', 'sample', 'k-means', 'avg_trace')
model = sq.AESeqModel()
nodes = model.build_graph(
    model_opt, no_dropout=True,
    **{'out:q_token_nll': not q_inactive, 'out:no_debug': True})
sess_config = sq.get_tfsession_config(True, num_threads=8)
sess = tf.Session(config=sess_config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(
    sq.filter_tfvars_in_checkpoint(tf.global_variables(), epath('checkpoint/best')))
saver.restore(sess, epath('checkpoint/best'))
print(sq.describe_variables(tf.trainable_variables(), ''))


def compute_nlls(batch, trace_states=None):

    def get_states(vector):
        return tuple((vector[:, 0:200], vector[:, 200:]))

    max_steps = batch.features.inputs.shape[0] + 1
    batch_size = batch.features.inputs.shape[-1]
    run_fn = partial(
        model.evaluate, sess=sess, features=batch.features, labels=batch.labels)
    total_token_probs = np.zeros((max_steps, batch_size), dtype=np.float32)
    q = None
    if q_mode == 'zero':
        result, __ = run_fn()
        nlls = result['token_nll']
    elif q_mode == 'sample':
        for __ in range(M):
            states = sampled_states[get_random_state_ids(batch_size)]
            result, __ = run_fn(state=get_states(states))
            total_token_probs += np.exp(-result['token_nll'])
        nlls = -np.log(total_token_probs / M)
    elif q_mode == 'k-means':
        means = np.reshape(gmm.means_, (gmm.means_.shape[0], 1, -1))
        for prior, mean in zip(gmm.weights_, means):
            states = np.repeat(mean, batch_size, axis=0)
            result, __ = run_fn(state=get_states(states))
            total_token_probs += np.exp(-result['token_nll']) * prior
        nlls = -np.log(total_token_probs)
    elif q_mode == 'avg_trace':
        states = get_states(trace_states)
        result, __ = run_fn(state=states)
        nlls = result['token_nll']
    elif q_mode == 'direct':
        result, extra = run_fn()
        nlls = result['token_nll']
    elif q_mode == 'gaussian':
        # result, extra = run_fn(extra_fetch=['q_out'])
        result, extra = run_fn()
        nlls = result['token_nll']
        # q_out = extra[0]
        # q = q_out[2] + q_out[3]
    elif q_mode == 'gmm':
        result, extra = run_fn()
        nlls = result['token_nll']
    sum_nll = nlls.sum(axis=0)
    if q is not None:
        sum_nll += q
    return sum_nll, nlls, q


def kmpAllMatches(pattern, text):

    def computeShifts(pattern):
        shifts = [None] * (len(pattern) + 1)
        shift = 1
        for pos in range(len(pattern) + 1):
            while shift < pos and pattern[pos-1] != pattern[pos-shift-1]:
                shift += shifts[pos-shift-1]
            shifts[pos] = shift
        return shifts

    shift = computeShifts(pattern)
    startPos = 0
    matchLen = 0
    for c in text:
        while matchLen >= 0 and pattern[matchLen] != c:
            startPos += shift[matchLen]
            matchLen -= shift[matchLen]
        matchLen += 1
        if matchLen == len(pattern):
            yield startPos
            startPos += shift[matchLen]
            matchLen -= shift[matchLen]


def find_ngram(ngram):
    acc = None
    for i, word in enumerate(ngram):
        p = nll_iix[word] - i
        if acc is None:
            acc = p
        else:
            acc = np.intersect1d(acc, p, assume_unique=True)
        if len(acc) == 0:
            raise ValueError(f"{ngram} cannot be found!")
    return acc
    # return tuple(kmpAllMatches(ngram, tokens))


def average_ll_from_trace(ngram, positions):
    probs = np.zeros((len(ngram), ), np.float32)
    for i in range(len(ngram)):
        for pos in positions:
            probs[i] += np.exp(nll_data[pos + i])
    probs = probs / len(positions)
    return -np.sum(np.log(probs))


def average_state_from_trace(positions):
    state = np.zeros((400, ), np.float32)
    for pos in positions:
        if pos > 0:
            state += sampled_states[pos - 1]  # state before n-gram
    state = state / len(positions)
    return state


# eval
results = {}
print('evaluating...')
over_ll = 0.0
under_ll = 0.0
over_count = 0
under_count = 0
close_count = 0
for batch in batches():
    # get batch data
    ngrams = []
    positions = []
    trace_states = []
    batch_lines = np.concatenate((batch.features.inputs[:1, :], batch.labels.label), 0)
    seq_lens = batch.features.seq_len
    for line, seq_len in zip(batch_lines.T, seq_lens):
        ngram = tuple(vocab.i2w(line[0: seq_len + 1]))
        ngrams.append(ngram)
        pos = find_ngram(ngram)
        positions.append(pos)
        trace_states.append(average_state_from_trace(pos))
    trace_states = np.stack(trace_states, axis=0)
    # get prediction
    sum_nll, token_nll, q = compute_nlls(batch, trace_states)
    sum_ll = - sum_nll
    # comparison
    for ngram, pos, seq_len, predict_ll in zip(ngrams, positions, seq_lens, sum_ll):
        if seq_len == 0:
            continue
        count = ngram_counts[ngram]
        target_ll = np.log(count / total_tokens)
        trace_ll = average_ll_from_trace(ngram, pos)
        results[ngram] = (len(ngram), count, target_ll, predict_ll, trace_ll)
        if np.isclose(target_ll, predict_ll, rtol=0, atol=1.e-3):
            close_count += 1
        elif predict_ll > target_ll:
            over_count += 1
            over_ll += predict_ll - target_ll
        elif predict_ll < target_ll:
            under_count += 1
            under_ll += target_ll - predict_ll
    print('.', end='', flush=True)
if q_mode == 'sample':
    q_mode = f'{q_mode}{M}'
with open(epath(f'marginals-{q_mode}.pkl'), 'wb') as ofp:
    pickle.dump(results, ofp)
total = over_count + under_count + close_count
print(f'\nTotal n-grams: {total}')
try:
    print(f'\item[Close]: {close_count / total:.5f}')
except ZeroDivisionError:
    pass
try:
    print(f'\item[Over]: {over_count / total:.5f} ({over_ll / over_count:.5f})')
except ZeroDivisionError:
    pass
try:
    print(f'\item[Under]: {under_count / total:.5f} ({under_ll / under_count:.5f})')
except ZeroDivisionError:
    pass
