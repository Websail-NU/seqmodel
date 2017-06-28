import sys

import kenlm
import numpy as np

sys.path.insert(0, '../')
import seqmodel as sq  # noqa

lm_path = sys.argv[1]
vocab_path = sys.argv[2]
data_path = sys.argv[3]
num_iters = int(sys.argv[4])
out_path = sys.argv[5]
greedy = sys.argv[6] == 'True'

lm = kenlm.Model(lm_path)
vocab = sq.Vocabulary.from_vocab_file(vocab_path)

wbdefs = []
with open(data_path) as ifp:
    for line in ifp:
        wbdefs.append(line.split('\t')[0])

choices = np.arange(vocab.vocab_size)
prob = np.zeros((vocab.vocab_size, ))

with open(out_path, 'w') as ofp:
    for __ in range(num_iters):
        for wbdef in wbdefs:
            state1, state2 = kenlm.State(), kenlm.State()
            lm.BeginSentenceWrite(state1)
            sentence = [wbdef, '<def>']
            senprob = []
            lm.BaseScore(state1, wbdef, state2)
            state1, state2 = state2, state1
            lm.BaseScore(state1, '<def>', state2)
            state1, state2 = state2, state1
            word = '<def>'
            while word != '</s>':
                for i in range(vocab.vocab_size):
                    prob[i] = lm.BaseScore(state1, vocab.i2w(i), state2)
                prob = np.power(10, prob)
                prob = prob / prob.sum()
                if greedy:
                    wid = np.argmax(prob)
                else:
                    wid = np.random.choice(choices, p=prob)
                p = prob[wid]
                word = vocab.i2w(wid)
                sentence.append(word)
                senprob.append(p)
                if len(sentence) > 42:
                    break
                lm.BaseScore(state1, word, state2)
                state1, state2 = state2, state1
            sentence = ' '.join(sentence[:-1])
            senprob = ' '.join((str(np.log(p)) for p in senprob))
            ofp.write(f'{sentence}\t{senprob}\n')
