import sys
import random
import argparse
import os
import copy
import numpy as np

sys.path.insert(0, '../../')
from seqmodel.dstruct import Vocabulary  # noqa

special_symbols = Vocabulary.special_symbols

# simple state-transition matrix
# row is state (position)
# column is transition label (vocab)
max_seq_len = 10
vocab_size = 11
base = [1.0 / 2**n for n in range(vocab_size)]
ending = [1.0 / 1.2**n for n in range(9-3)] + [0.0] * 3
ending.reverse()
transitions = [base]
for i in range(max_seq_len-2):
    t = list(base)
    transitions.append(t)
    random.shuffle(t)
transitions.append([0] * (vocab_size - 1) + [1.0])
transitions = np.array(transitions)
transitions[0:9, -1] = ending
transitions = transitions / np.reshape(
    transitions.sum(axis=1), [max_seq_len, 1])
choices = 'a b c d e f g h i j'.split()
valid_choices = list(choices)
choices.append(special_symbols['end_seq'])


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_sample():
    tokens = []
    for p in range(max_seq_len):
        t = np.random.choice(choices, p=transitions[p])
        if t == special_symbols['end_seq']:
            break
        tokens.append(t)
    return tokens


def generate_random_sample():
    rand_len = int(np.random.uniform(high=max_seq_len, low=2))
    return np.random.choice(valid_choices, size=rand_len)


def write_samples(path, n, mode, random_seq=False):
    with open(path, 'w') as ofp:
        for i in range(n):
            if random_seq:
                sample = generate_random_sample()
            else:
                sample = generate_sample()
            in_sample = ' '.join(sample)
            if mode == 'single':
                ofp.write('{}\n'.format(in_sample))
                continue
            if mode == 'reverse':
                sample.reverse()
            out_sample = ' '.join(sample)
            ofp.write('{} \t {}\n'.format(in_sample, out_sample))


def write_vocab(path):
    with open(path, 'w') as ofp:
        for k in special_symbols.keys():
            ofp.write('{}\n'.format(special_symbols[k]))
        for c in choices:
            if c == special_symbols['end_seq']:
                break
            ofp.write('{}\n'.format(c))


parser = argparse.ArgumentParser()
parser.add_argument("output_dir")
parser.add_argument("--mode", type=str, default="copy",
                    choices=['copy', 'reverse', 'single'])
parser.add_argument("--num_train", type=int, default=10000)
parser.add_argument("--num_valid", type=int, default=1000)
parser.add_argument("--num_test", type=int, default=1000)
parser.add_argument("--random_seq", action="store_true")
args = parser.parse_args()
ensure_dir(args.output_dir)
write_samples(os.path.join(args.output_dir, 'train.txt'),
              args.num_train, args.mode, args.random_seq)
write_samples(os.path.join(args.output_dir, 'valid.txt'),
              args.num_valid, args.mode, args.random_seq)
write_samples(os.path.join(args.output_dir, 'test.txt'),
              args.num_test, args.mode, args.random_seq)
write_vocab(os.path.join(args.output_dir, 'vocab.txt'))
