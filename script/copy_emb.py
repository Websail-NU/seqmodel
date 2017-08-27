import os
import sys
import numpy as np
from gensim.models import KeyedVectors

# _special_symbols = set(['<s>', '</enc>'])
# cap_words = set(['a', 'and', 'to', 'of'])

_special_symbols = set()
cap_words = set()

w2v_path = sys.argv[1]
binary = True
vocab_path = sys.argv[2]
out_path = sys.argv[3]


print('- Loading word2vec...')
w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=binary)
words = []
freq = {}
print('- Reading vocab...')
with open(vocab_path) as ifp:
    for line in ifp:
        parts = line.strip().split('\t')
        word = parts[0]
        if len(parts) > 1:
            freq[word] = parts[1]
        else:
            freq[word] = "0"
        words.append(word)
print('- Vocab size: {}'.format(len(words)))
print('- Copying word2vec...')
vocab2vec = np.random.uniform(low=-0.05, high=0.05,
                              size=(len(words), w2v['test'].shape[0]))
vocab2vec = vocab2vec / np.linalg.norm(vocab2vec, ord=2, axis=0)
unk_words = []
for i, corpus_word in enumerate(words):
    word = corpus_word
    # XXX: common words that are not in word2vec, but cap exists
    # if word in cap_words:
    #     word = word[0].upper() + word[1:]
    # if word == '<unk>':
    #     word = 'UNK'
    # if word == 'e.g.' or word == 'i.e.':
    #     word = 'eg'
    # if word == "'s":
    #     word = 's'
    if word in w2v:
        vocab2vec[i] = w2v[word]
    elif word in _special_symbols:
        vocab2vec[i, :] = 0.0
    else:
        unk_words.append(corpus_word)
print('- Unknown words: {}'.format(len(unk_words)))
print('- Writing output...')
with open(out_path, 'wb') as ofp:
    np.save(ofp, vocab2vec)
unk_path = out_path + '_unk.txt'
with open(unk_path, 'w') as ofp:
    for word in unk_words:
        ofp.write("{}\t{}\n".format(word, freq[word]))
