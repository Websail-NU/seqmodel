import os
import sys
import json
import pickle
from functools import partial
from collections import ChainMap

import numpy as np
import tensorflow as tf

import seqmodel as sq

batch_size = 1000
q_mode = 'sample'
data_dir = 'data/ptb/'
eval_file = 'curexp/ptb-v/sample.count-filter-text'
count_file = 'curexp/ptb-v/sample.5.count'
trace_nll_file = 'curexp/ptb-v/sample_nll.txt'

gmm_dir = 'curexp/ptb-v'
gmm_file = 'gmm64valid_model.pkl'
# model_dir = 'curexp/ptb-cde-kmean-with_nll'
model_dir = 'curexp/ptb-cde-direct-only_native'
# model_dir = 'curexp/ptb-v'
state_sample_file = 'sample_states.npy'

# data
print('loading n-gram counts...')
ngram_counts = sq.ngram_stat.read_ngram_count_file(count_file)
print('loading data...')
dpath = partial(os.path.join, data_dir)
vocab = sq.Vocabulary.from_vocab_file(dpath('vocab.txt'))
ngram_lines = sq.read_lines(eval_file, token_split=' ')
data = sq.read_seq_data(
    ngram_lines, vocab, vocab, keep_sentence=True,
    data_include_eos=True, add_sos=False)
batches = partial(
    sq.seq_batch_iter, *data, batch_size=batch_size, shuffle=True, keep_sentence=True)

print('loading trace data...')
# gmm
gmm_path = partial(os.path.join, gmm_dir)
with open(gmm_path(gmm_file), mode='rb') as gmm_pickle:
    gmm = pickle.load(gmm_pickle)
_a = np.load(gmm_path(state_sample_file))
sampled_states = np.transpose(_a, (1, 0, 2)).reshape((-1, _a.shape[-1]))
# sampled_states = np.reshape(_a, (-1, _a.shape[-1]))
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
# nodes = model.build_graph(
#     model_opt, no_dropout=True,
#     **{'out:q_token_nll': True, 'out:no_debug': True})
nodes = model.build_graph(
    model_opt, no_dropout=True,
    **{'out:q_token_nll': not q_inactive, 'out:no_debug': True})
sess_config = sq.get_tfsession_config(True, num_threads=8)
sess = tf.Session(config=sess_config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(
    sq.filter_tfvars_in_checkpoint(tf.global_variables(), epath('checkpoint/best')))
saver.restore(sess, epath('checkpoint/best'))
# print(sq.describe_variables(tf.trainable_variables(), ''))


def make_batch(ngram):
    ids = vocab.w2i(ngram)
    x, y = ids[:-1], ids[1:]
    x = [x for __ in range(batch_size)]
    y = [y for __ in range(batch_size)]
    x_arr, x_len = sq.hstack_list(x)
    y_arr, y_len = sq.hstack_list(y)
    seq_weight = np.where(y_len > 0, 1, 0).astype(np.float32)
    token_weight, num_tokens = sq.masked_full_like(
        y_arr, 1, num_non_padding=y_len)
    features = sq.SeqFeatureTuple(x_arr, x_len)
    labels = sq.SeqLabelTuple(y_arr, token_weight, seq_weight)
    batch = sq.BatchTuple(features, labels, num_tokens, False)
    return batch


def linear_kernel(x_points, mu_points):
    dot_product = np.sum(x_points * mu_points, axis=-1)
    kernel = np.exp(dot_product)
    return kernel, dot_product


def gaussian_kernel(x_points, mu_points):
    euc_dist = np.linalg.norm(x_points - mu_points, axis=-1)
    kernel = np.exp(-(euc_dist ** 2.0) / 2.0) / (2 * np.pi)
    return kernel, euc_dist


def log_linear(x, mu):
    dot_product = np.sum(x * mu, axis=-1)
    return dot_product


def log_normal(x, mu, var=1.0):
    # var = sampled_states.var(axis=-1)
    # var = np.expand_dims(var, axis=-1)
    # var = np.expand_dims(var, axis=-1)
    k = -0.5 * (np.log(2 * np.pi) + np.log(var) + np.square(x - mu) / var)
    return np.sum(k, axis=-1)


def softmax(logit, axis=-1):
    max_logit = np.max(logit, axis=axis, keepdims=True)
    exp_logit = np.exp(logit - max_logit)
    return exp_logit / exp_logit.sum(axis=axis, keepdims=True)


def log_softmax(logit, axis=-1):
    max_logit = np.max(logit, axis=axis, keepdims=True)
    exp_logit = np.exp(logit - max_logit)
    return logit - (max_logit + np.log(exp_logit.sum(axis=axis, keepdims=True)))


def get_states(vector):
    return tuple((vector[:, 0:200], vector[:, 200:]))


def main(num_samples, ngram):
    N = len(ngram)
    if N == 1:
        ngram = tuple((ngram[0], '</s>'))
    batch = make_batch(ngram)
    run_fn = partial(
        model.evaluate, sess=sess, features=batch.features, labels=batch.labels)
    __, q = run_fn(extra_fetch=['q_states'])
    q_first_states = q[0][0, :, :]
    q_states = q[0][:N, 0, :]
    x_points = np.expand_dims(sampled_states, 1)
    mu_points = np.expand_dims(q_states, 0)
    logit = log_linear(x_points, mu_points)
    # all_p_keeps = 1 / (1 + np.exp(-dot_product))
    uniform_prob = 1.0 / len(sampled_states)
    q_h = softmax(logit, axis=0)
    # kl = np.sum(q_h[:, 0] * (np.log(q_h[:, 0]) - np.log(uniform_prob)))
    # print(kl)
    # kl += np.sum(q_h[:, 1] * (np.log(q_h[:, 1]) - np.log(q_h[:, 0])))
    # print(kl)

    nll_data = []
    total_token_probs = np.zeros((len(ngram[:N]), ), dtype=np.float32)
    total_token_counts = np.zeros((len(ngram[:N]), ), dtype=np.float32)
    for i in range(num_samples // batch_size):
        state_ids = get_random_state_ids(batch_size, p=q_h[:, N-1])
        # state_ids = get_random_state_ids(batch_size)
        init_state_ids = state_ids - N + 1
        init_states = sampled_states[init_state_ids]
        # p_keeps = all_p_keeps[state_ids]
        result, ex = run_fn(state=get_states(init_states), extra_fetch=['g_states'])
        p_states = ex[0]
        token_probs = np.exp(-result['token_nll'])

        for j in range(batch_size):
            init_state_id = init_state_ids[j]
            for n in range(len(ngram[:N])):
                count = 1.0
                if n == 0:
                    weight = uniform_prob / q_h[init_state_id + n, n]
                else:
                    weight = 1.0
                    pn = n - 1
                    p_h = log_linear(sampled_states, p_states[n, j, :])
                    p_h = softmax(p_h, axis=0)
                    # weight = q_h[init_state_id + pn, pn] / q_h[init_state_id + n, n]
                    # if np.abs(np.log(weight)) > 10:
                    #     weight = 0
                    #     count = 0
                    weight = p_h[init_state_id + n] / q_h[init_state_id + n, n]
                total_token_probs[n] += token_probs[n, j] * weight
                total_token_counts[n] += count
            if np.any(total_token_counts == 0):
                nll_data.append((0.0, 0.0))
            else:
                cur = np.log(total_token_probs / total_token_counts).sum()
                nll_data.append((-np.sum(result['token_nll'][:, j]), cur))
    # print(total_token_counts)
    return nll_data, total_token_counts


if __name__ == '__main__':
    only_predict = sys.argv[-1] == "True"
    num_samples = int(float(sys.argv[1]))
    map_file = epath('sampling_marginals.map.pkl')
    data_path = epath('sampling_marginals.data')
    sq.ensure_dir(data_path, delete=False)
    if os.path.exists(map_file):
        with open(map_file, mode='rb') as f:
            map_ = pickle.load(f)
    else:
        map_ = {}

    for argv in sys.argv[2:-1]:
        ngram = tuple(argv.split(' '))
        count = ngram_counts[ngram]
        target_ll = np.log(count / total_tokens)
        # ll, kl = main(num_samples, ngram)
        # ngram = ' '.join(ngram)
        # print(f'"{ngram}"\t{target_ll}\t{ll-kl}')
        nll_data, weights = main(num_samples, ngram)
        if ngram in map_:
            idx, __, __ = map_[ngram]
        else:
            idx = len(map_)
        nll_data = np.array(nll_data)
        np.save(os.path.join(data_path, f'{idx}.npy'), nll_data)
        ngram = ' '.join(ngram)
        # print(f'"{ngram}"\t{target_ll}\t{nll_data[-1, -1]}\t{weights}')
        if only_predict:
            print(f'{nll_data[-1, -1]}')
        else:
            print(f'"{ngram}"\t{target_ll}\t{nll_data[-1, -1]}')
        map_[ngram] = (idx, count, -target_ll)
        with open(map_file, mode='wb') as f:
            pickle.dump(map_, f)

# def main2(num_samples, ngram):
#     K = 3
#     var = 1.0
#     # var = sampled_states.var(axis=-1)
#     # var = np.expand_dims(var, axis=-1)
#     # var = np.expand_dims(var, axis=-1)
#     # logit_fn = partial(log_normal, var=var)
#     logit_fn = log_linear
#     N = len(ngram)
#     if N == 1:
#         ngram = tuple((ngram[0], '</s>'))
#     batch = make_batch(ngram)
#     run_fn = partial(
#         model.evaluate,
#         sess=sess, features=batch.features, labels=batch.labels)
#     uniform_prob = 1.0 / len(sampled_states)
#     __, (q, ) = run_fn(extra_fetch=['q_out'])

#     k_prob = softmax(q.alpha[:, 0, :], axis=-1)
#     print(k_prob)
#     total_token_probs = np.zeros((N, ), dtype=np.float32)
#     total_kl = np.zeros((N, ), dtype=np.float32)
#     x_points = np.expand_dims(sampled_states, 1)

#     for k in range(K):
#         result, (g_states, ) = run_fn(
#             extra_fetch=['g_states'],
#             state=get_states(q.k_states[0, :, k]),
#             q_states=tuple((q.k_states[:, :, k, 0:200], q.k_states[:, :, k, 200:400])))
#         token_nll = result['token_nll']
#         q_points = np.expand_dims(q.k_states[:, 0, k, :], 0)
#         # p_points = np.concatenate(
#         #     (np.zeros_like(g_states[0:1, 0, :]), g_states[:, 0, :]), 0)
#         # p_points = np.expand_dims(p_points, 0)
#         p_points = np.expand_dims(g_states[:, 0, :], 0)

#         q_logit = logit_fn(x_points, q_points)
#         p_logit = logit_fn(x_points, p_points)
#         for n in range(N):
#             weight = k_prob[n, k]
#             qn = q_logit[:, n]
#             if n == 0:
#                 log_pn = np.log(uniform_prob)
#             else:
#                 log_pn = log_softmax(p_logit[:, n-1], axis=0)
#             # log_pn = log_softmax(p_logit[:, n], axis=0)
#             kl = np.sum(
#                 softmax(qn, axis=0) * (log_softmax(qn, axis=0) - log_pn))
#             total_token_probs[n] += np.exp(-token_nll[n, 0]) * weight
#             total_kl[n] += kl * weight
#     print(np.log(total_token_probs))
#     print(total_kl)
#     sum_ll = np.log(total_token_probs).sum()
#     sum_kl = total_kl.sum()
#     # print(total_kl)
#     # print(sum_ll, sum_kl, sum_ll - sum_kl)
#     return sum_ll, sum_kl
