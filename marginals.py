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

gmm_dir = 'curexp/ts-v'
model_dir = 'curexp/ts-bw'
data_dir = 'data/tinyshakespeare/'

gmm_valid_file = 'gmm64valid_model.pkl'
gmm_sample_file = 'gmm64sample_model.pkl'
state_sample_file = 'states_sample.npy'
ngram_count_file = 'sample_words.count'
total_tokens = 4000063


# Load GMMs and sampled words and states
gmm_path = partial(os.path.join, gmm_dir)
with open(gmm_path(gmm_valid_file), mode='rb') as gmm_file:
    gmm_valid = pickle.load(gmm_file)
with open(gmm_path(gmm_sample_file), mode='rb') as gmm_file:
    gmm_sample = pickle.load(gmm_file)
sampled_states = np.load(gmm_path(state_sample_file))
sampled_states = sampled_states[~np.all(sampled_states == 0, axis=-1)]
get_random_state_ids = partial(np.random.choice, np.arange(len(sampled_states)))
ngram_counts = sq.ngram_stat.read_ngram_count_file(
    gmm_path(ngram_count_file), remove_sentence=True)
# TODO: Filter ngram_counts
sampling_weights = np.array(list(ngram_counts.values()))
sampling_weights = sampling_weights / np.sum(sampling_weights)
sampled_words = list(ngram_counts.keys())
word_choices = np.arange(len(sampled_words))
get_random_word_ids = partial(np.random.choice, word_choices, p=sampling_weights)
# sampled_states = np.expand_dims(sampled_states, 0)

# Load configurations and build model graph
epath = partial(os.path.join, model_dir)
dpath = partial(os.path.join, data_dir)
with open(epath('model_opt.json')) as ifp:
    model_opt = json.load(ifp)
vocab = sq.Vocabulary.from_vocab_file(dpath('vocab.txt'))
model_vocab_opt = sq.AESeqModel.get_vocab_opt(*(v.vocab_size for v in [vocab, vocab]))
model_opt = ChainMap(
    {'xxx:add_first_token': True, 'loss:eval_nll': True}, model_vocab_opt, model_opt)
bw_model = sq.AESeqModel()
bw_nodes = bw_model.build_graph(model_opt, no_dropout=True, **{'rnn:use_bw_state': True})
fw_model = sq.AESeqModel()
fw_nodes = fw_model.build_graph(model_opt, reuse=True, no_dropout=True)

# Create Session and restore model
sess_config = sq.get_tfsession_config(False, num_threads=8)
sess = tf.Session(config=sess_config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, epath('checkpoint/best'))


# Helper functions

def gmm_logprob(gmm, vectors):
    if len(vectors.shape) == 1:
        vectors = vectors.reshape((1, -1))
    probs = gmm.predict_proba(vectors)
    # print(probs.argmax(axis=-1))
    # print(probs[0])
    # print(probs[1])
    # print(probs[2])
    marginal_probs = np.matmul(probs, gmm.weights_)
    return np.log(marginal_probs)


def make_batch(seq_list):
    seq_token_id_list = vocab.w2i(seq_list)
    x, y = zip(*[(ids[:-1], ids[1:]) for ids in seq_token_id_list])
    x_arr, x_len = sq.hstack_list(x)
    y_arr, y_len = sq.hstack_list(y)
    seq_weight = np.where(y_len > 0, 1, 0).astype(np.float32)
    token_weight, num_tokens = sq.masked_full_like(
        y_arr, 1, num_non_padding=y_len)
    features = sq.SeqFeatureTuple(x_arr, x_len)
    labels = sq.SeqLabelTuple(y_arr, token_weight, seq_weight)
    batch = sq.BatchTuple(features, labels, num_tokens, False)
    return batch


def compute_nlls(batch, mode='zero', gmm=None, M=10):

    def get_states(vector):
        return tuple((vector[:, 0:128], vector[:, 128:]))

    max_steps = batch.features.inputs.shape[0] + 1
    batch_size = batch.features.inputs.shape[-1]
    model = bw_model if mode == 'q' else fw_model
    run_fn = partial(
        model.evaluate, sess=sess, features=batch.features, labels=batch.labels)
    total_token_probs = np.zeros((max_steps, batch_size), dtype=np.float32)
    q = None
    if mode == 'zero':
        result, __ = run_fn()
        nlls = result['nll']
    elif mode == 'sample':
        for __ in range(M):
            states = sampled_states[get_random_state_ids(batch_size)]
            result, __ = run_fn(state=get_states(states))
            total_token_probs += np.exp(-result['nll'])
        nlls = -np.log(total_token_probs / M)
    elif mode == 'k-means':
        means = np.reshape(gmm.means_, (gmm.means_.shape[0], 1, -1))
        for prior, mean in zip(gmm.weights_, means):
            states = np.repeat(mean, batch_size, axis=0)
            result, __ = run_fn(state=get_states(states))
            total_token_probs += np.exp(-result['nll']) * prior
        nlls = -np.log(total_token_probs)
    elif mode == 'q':
        result, extra = run_fn(extra_fetch=['bw_final_state'])
        q_states = np.concatenate(extra[0], axis=-1)
        # print(np.linalg.norm(q_states[0, :].reshape((1, -1)) - q_states, axis=-1))
        q = -gmm_logprob(gmm, q_states)
        nlls = result['nll']
    sum_nll = nlls.sum(axis=0)
    if q is not None:
        sum_nll += q
    return sum_nll, nlls, q


def display_sum_nlls(words, sum_nlls):
    for i, word in enumerate(words):
        str_nlls = [word] + [f'{nll:>7.3f}' for nll in sum_nlls[i]]
        print('|'.join(str_nlls))


random_words = [['_'] + list(sampled_words[idx][0]) for idx in get_random_word_ids(10)]
batch = make_batch(random_words)
sum_nll, token_nll, q = compute_nlls(batch, mode='q', gmm=gmm_sample)
display_sum_nlls(random_words, [sum_nll])

# def check_ll(text, prefix='', per_token=False):
#     text = f'{prefix}{text}'
#     tokens = ['_'] + list(text.strip().replace(' ', '_')) + ['_']
#     batch = make_batch(tokens)
#     result, extra = model.evaluate(
#         sess=sess, features=batch.features, labels=batch.labels)
#     marginal_ll_tokens_v = weighted_sum(batch, means_v, weight_v, result)
#     marginal_ll_tokens_s = weighted_sum(batch, means_s, weight_s, result)
#     marginal_ll_tokens_ss = sample(batch, result)
#     c = ngram_counts[(text,)]
#     if c == 0:
#         c_ll = -float('inf')
#     else:
#         c_ll = np.log(c / total_tokens)
#     fw_ll = -result['nll'].sum()
#     ll_v = marginal_ll_tokens_v.sum()
#     ll_s = marginal_ll_tokens_s.sum()
#     ll_ss = marginal_ll_tokens_ss.sum()
#     if per_token:
#         for token, nll1, nll2, nll3, nll4 in zip(
#                 tokens, result['nll'],
#                 -marginal_ll_tokens_v, -marginal_ll_tokens_s, -marginal_ll_tokens_ss):
#             nll1 = nll1[0]
#             nll2 = nll2[0]
#             nll3 = nll3[0]
#             nll4 = nll4[0]
#             print(f'{token}  {nll1:>7.3f}  {nll2:>7.3f}  {nll3:>7.3f}  {nll4:>7.3f}')
#         print((f'{text} {c}, {c_ll:.3f} vs {fw_ll:.3f} vs '
#                f'{ll_v:.3f} vs {ll_s:.3f} vs {ll_ss:.3f}'))
#     else:
#         print((f'{text.ljust(20)}| {c:5d} {c_ll:>7.3f} | {fw_ll:>7.3f} '
#                f' | {ll_v:>7.3f} | {ll_s:>7.3f} | {ll_ss:>7.3f}'))


# def random_check(num_samples=10):
#     for choice in np.random.choice(choices, num_samples, p=sampling_weights):
#         word = words[choice][0]
#         if len(word) == 1:
#             continue
#         if any((c.isupper() for c in word)):
#             continue
#         check_ll(word)


# num_samples = 10
# random_check(10)
# check_ll('it', per_token=True)
