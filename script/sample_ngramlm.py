import sys

import kenlm
import numpy as np

sys.path.insert(0, '../')
import seqmodel as sq  # noqa

lm_path = sys.argv[1]
vocab_path = sys.argv[2]
num_samples = int(sys.argv[3])
out_path = sys.argv[4]

lm = kenlm.Model(lm_path)
vocab = sq.Vocabulary.from_vocab_file(vocab_path)
choices = np.arange(vocab.vocab_size)
prob = np.zeros((vocab.vocab_size, ))

with open(out_path, 'w') as ofp:
    for __ in range(num_samples):
        state1, state2 = kenlm.State(), kenlm.State()
        lm.BeginSentenceWrite(state1)
        word = '<s>'
        sentence = []
        while word != '</s>':
            for i in range(vocab.vocab_size):
                prob[i] = lm.BaseScore(state1, vocab.i2w(i), state2)
            prob = np.power(10, prob)
            prob = prob / prob.sum()
            word = vocab.i2w(np.random.choice(choices, p=prob))
            sentence.append(word)
            lm.BaseScore(state1, word, state2)
            state1, state2 = state2, state1
        sentence = ' '.join(sentence[:-1])
        ofp.write(f'{sentence}\n')
