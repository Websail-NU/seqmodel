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

batch_size = 512
q_mode = sys.argv[1]
M = 100

data_dir = 'data/tinyshakespeare/'
eval_file = f'{data_dir}/train.char.filter.txt'
count_file = f'{data_dir}/train.char.filter.count'
total_tokens = 1016242
# eval_file = 'curexp/ts-v/sample_filter.txt'
# count_file = 'curexp/ts-v/sample_filter.count'
# total_tokens = 4000063

gmm_dir = 'curexp/ts-v'
gmm_file = 'gmm64valid_model.pkl'
# model_dir = 'curexp/ts-bw-m1'
model_dir = f'curexp/ts-bw-m3-{q_mode}'
state_sample_file = 'states_sample.npy'

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
sampled_states = np.load(gmm_path(state_sample_file))
sampled_states = sampled_states[~np.all(sampled_states == 0, axis=-1)]
get_random_state_ids = partial(np.random.choice, np.arange(len(sampled_states)))


# model
print('loading model...')
epath = partial(os.path.join, model_dir)
with open(epath('model_opt.json')) as ifp:
    model_opt = json.load(ifp)
model_vocab_opt = sq.AESeqModel.get_vocab_opt(*(v.vocab_size for v in [vocab, vocab]))
model_opt = ChainMap(
    {'loss:eval_nll': True}, model_vocab_opt, model_opt)
q_inactive = q_mode in ('zero', 'sample', 'k-means')
model = sq.AESeqModel()
nodes = model.build_graph(
    model_opt, no_dropout=True,
    **{'rnn:use_bw_state': not q_inactive})
sess_config = sq.get_tfsession_config(True, num_threads=8)
sess = tf.Session(config=sess_config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, epath('checkpoint/best'))
print(sq.describe_variables(tf.trainable_variables(), ''))


def compute_nlls(batch):

    def get_states(vector):
        return tuple((vector[:, 0:128], vector[:, 128:]))

    max_steps = batch.features.inputs.shape[0] + 1
    batch_size = batch.features.inputs.shape[-1]
    run_fn = partial(
        model.evaluate, sess=sess, features=batch.features, labels=batch.labels)
    total_token_probs = np.zeros((max_steps, batch_size), dtype=np.float32)
    q = None
    if q_mode == 'zero':
        result, __ = run_fn()
        nlls = result['nll']
    elif q_mode == 'sample':
        for __ in range(M):
            states = sampled_states[get_random_state_ids(batch_size)]
            result, __ = run_fn(state=get_states(states))
            total_token_probs += np.exp(-result['nll'])
        nlls = -np.log(total_token_probs / M)
    elif q_mode == 'k-means':
        means = np.reshape(gmm.means_, (gmm.means_.shape[0], 1, -1))
        for prior, mean in zip(gmm.weights_, means):
            states = np.repeat(mean, batch_size, axis=0)
            result, __ = run_fn(state=get_states(states))
            total_token_probs += np.exp(-result['nll']) * prior
        nlls = -np.log(total_token_probs)
    elif q_mode == 'l2':
        result, extra = run_fn()
        nlls = result['nll']
    elif q_mode == 'gaussian':
        # result, extra = run_fn(extra_fetch=['q_out'])
        result, extra = run_fn()
        nlls = result['nll']
        # q_out = extra[0]
        # q = q_out[2] + q_out[3]
    elif q_mode == 'gmm':
        result, extra = run_fn()
        nlls = result['nll']
    sum_nll = nlls.sum(axis=0)
    if q is not None:
        sum_nll += q
    return sum_nll, nlls, q


# eval
print('evaluating...')
over_ll = 0.0
under_ll = 0.0
over_count = 0
under_count = 0
close_count = 0
for batch in batches():
    # get prediction
    sum_nll, token_nll, q = compute_nlls(batch)
    sum_ll = - sum_nll
    # result, __ = model.evaluate(sess=sess, features=batch.features, labels=batch.labels)
    # token_nll = result['nll']
    # sum_ll = -token_nll.sum(axis=0)
    # comparison
    batch_lines = np.concatenate((batch.features.inputs[:1, :], batch.labels.label), 0)
    for line, seq_len, predict_ll in zip(batch_lines.T, batch.features.seq_len, sum_ll):
        if seq_len == 0:
            continue
        ngrams = tuple(vocab.i2w(line[0: seq_len + 1]))
        count = ngram_counts[ngrams]
        target_ll = np.log(count / total_tokens)
        if np.isclose(target_ll, predict_ll, rtol=0, atol=1.e-3):
            close_count += 1
        elif predict_ll > target_ll:
            over_count += 1
            over_ll += predict_ll - target_ll
        elif predict_ll < target_ll:
            under_count += 1
            under_ll += target_ll - predict_ll
    print('.', end='', flush=True)
total = over_count + under_count + close_count
print(f'\nTotal n-grams: {total}')
print(f'\item[Close]: {close_count / total:.5f}')
print(f'\item[Over]: {over_count / total:.5f} ({over_ll / over_count:.5f})')
print(f'\item[Under]: {under_count / total:.5f} ({under_ll / under_count:.5f})')
