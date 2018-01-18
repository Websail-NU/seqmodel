import os
import sys
import numpy as np


big_emb_path = sys.argv[1]
big_vocab_path = sys.argv[2]
filter_vocab_path = sys.argv[3]
out_path = sys.argv[4]


print('- Loading word2vec...')
w2v = np.load(big_emb_path)
print('- Reading vocab...')


def read_vocab(path):
    vocab = {}
    with open(path) as ifp:
        for i, line in enumerate(ifp):
            parts = line.strip().split('\t')
            word = parts[0]
            vocab[word] = i
    return vocab


big_vocab = read_vocab(big_vocab_path)
filter_vocab = read_vocab(filter_vocab_path)
unk_words = []

print('- Copying word2vec...')
vocab2vec = np.random.uniform(low=-0.05, high=0.05,
                              size=(len(filter_vocab), w2v.shape[1]))
for w, i in filter_vocab.items():
    if w in big_vocab:
        vocab2vec[i] = w2v[big_vocab[w]]
    else:
        unk_words.append(w)

print('- Unknown words: {}'.format(len(unk_words)))
print('- Writing output...')
with open(out_path, 'wb') as ofp:
    np.save(ofp, vocab2vec)
