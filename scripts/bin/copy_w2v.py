import sys
import cPickle
import numpy as np
from gensim.models import KeyedVectors

w2v_path = sys.argv[1]
binary = True
vocab_path = sys.argv[2]
out_path = sys.argv[3]


print('- Loading word2vec...')
w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=binary)
words = []
print('- Reading vocab...')
with open(vocab_path) as ifp:
    for line in ifp:
        words.append(line.strip().split('\t')[0])
print('- Vocab size: {}'.format(len(words)))
print('- Copying word2vec...')
vocab2vec = np.random.uniform(low=-1.0, high=1.0,
                              size=(len(words), w2v['test'].shape[0]))
vocab2vec = vocab2vec / np.linalg.norm(vocab2vec, ord=2, axis=0)
unk = 0
for i, word in enumerate(words):
    if word in w2v:
        vocab2vec[i] = w2v[word]
    else:
        unk += 1
print('- Unknown words: {}'.format(unk))
print('- Writing output...')
with open(out_path, 'w') as ofp:
    cPickle.dump(obj=vocab2vec, file=ofp)
