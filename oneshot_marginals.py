import os
import sys
import json
import pickle
from pydoc import locate
from functools import partial
from collections import ChainMap

import numpy as np
import tensorflow as tf

import seqmodel as sq

gpu = False

batch_size = 1
data_dir = 'data/ptb/'
# eval_file = 'curexp/ptb-v/sample.count-filter-text'
# count_file = 'curexp/ptb-v/sample.5.count'
# trace_nll_file = 'curexp/ptb-v/sample_nll.txt'
# model_dir = 'curexp/ptb-v'
m = 'h_star-reset-10'
sample_file = f'curexp/ptb-{m}/sample.txt'
eval_file = f'curexp/ptb-{m}/sample.count-filter-text'
count_file = f'curexp/ptb-{m}/sample.5.count'
model_dir = f'curexp/ptb-{m}'
total_tokens = 0
with open(sample_file) as lines:
    for line in lines:
        total_tokens += len(line.strip().split(' ')) + 1
with open(os.path.join(model_dir, 'basic_opt.json')) as fp:
    basic_opt = json.load(fp)
    model_class = basic_opt['model_class']
    if model_class == '':
        model_class = 'seqmodel.SeqModel'
    MODEL_CLASS = locate(model_class)
# data
print('loading n-gram counts...')
ngram_counts = sq.ngram_stat.read_ngram_count_file(count_file)
print('loading vocab...')
dpath = partial(os.path.join, data_dir)
vocab = sq.Vocabulary.from_vocab_file(dpath('vocab.txt'))

# model
print('loading model...')
epath = partial(os.path.join, model_dir)
with open(epath('model_opt.json')) as ifp:
    model_opt = json.load(ifp)
model_vocab_opt = MODEL_CLASS.get_vocab_opt(*(v.vocab_size for v in [vocab, vocab]))
model_opt = ChainMap(
    {'out:token_nll': True, 'out:eval_first_token': True}, model_vocab_opt, model_opt)
model = MODEL_CLASS()
nodes = model.build_graph(model_opt, no_dropout=True)
sess_config = sq.get_tfsession_config(gpu, num_threads=8)
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


def main(ngram):
    N = len(ngram)
    if N == 1:
        ngram = tuple((ngram[0], '</s>'))
    batch = make_batch(ngram)
    run_fn = partial(
        model.evaluate,
        sess=sess, features=batch.features, labels=batch.labels)
    # zero_states = sq.cells.nested_map(
    #     lambda x: np.zeros((batch_size, x.shape[0]), dtype=np.float32),
    #     nodes['cell']._init_vars)
    # result, __ = run_fn(state=zero_states)
    result, __ = run_fn()
    token_nll = result['token_nll'][:N]
    return -token_nll.sum()


if __name__ == '__main__':
    only_predict = sys.argv[-1] == "True"
    for argv in sys.argv[1:-1]:
        ngram = tuple(argv.split(' '))
        count = ngram_counts[ngram]
        target_ll = np.log(count / total_tokens)
        predict_ll = main(ngram)
        ngram = ' '.join(ngram)
        if only_predict:
            print(f'{predict_ll}')
        else:
            print(f'"{ngram}"\t{target_ll}\t{predict_ll}')

