import sys

import numpy as np


rnn_output_path = sys.argv[1]
ngram_output_path = sys.argv[2]

rnn_output = []
ngram_output = []

with open(rnn_output_path) as lines:
    for line in lines:
        rnn_output.append(float(line.strip().split('\t')[-1]))

with open(ngram_output_path) as lines:
    for line in lines:
        if '[ ' not in line:
            continue
        s = line.index('[ ') + 2
        e = line.index(' ]') + 1
        ngram_output.append(float(line[s:e]))

rnn_output = np.array(rnn_output)
ngram_output = -1 * np.array(ngram_output) / np.log10(np.e)

for i in range(1, 11):
    w = i / 10
    output = (1 - w) * rnn_output + w * ngram_output
    print(w, np.exp(np.mean(output)))
