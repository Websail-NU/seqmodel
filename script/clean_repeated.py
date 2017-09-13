import os
import sys
from collections import Counter
from nltk.util import ngrams
import operator
import re


pattern = re.compile(r'(?P<words>.+) (or|,|, or|of|and) (?P=words)\b')


def clean_repeated(text, max_len=6, min_len=1):
    s = pattern.search(text)
    new_text = text
    if s is not None:
        new_text = text[0:s.start()] + s.groups()[0]
        if s.end() < len(text):
            new_text = new_text + text[s.end():]
    text = new_text
    tokens = text.split()
    for n in range(max_len, min_len - 1, -1):
        if n > len(tokens):
            continue
        done = False
        while not done:
            ngram = Counter(ngrams(tokens, n))
            p = sorted(ngram.items(), key=operator.itemgetter(1), reverse=True)
            for k in p:
                if k[1] == 1:
                    done = True
                    break
                r = list(k[0])
                pos = [(i, i+len(r)) for i in range(len(tokens)) if tokens[i:i+len(r)] == r]  # noqa
                prev_end = -1
                r_start = -1
                r_end = -1
                for start, end in pos:
                    if start <= prev_end:
                        if r_start == -1:
                            r_start = prev_end
                        r_end = end
                    prev_end = end
                if r_end != -1:
                    tokens = tokens[:r_start] + tokens[r_end:]
                    done = False
                    break
                else:
                    done = True
    return ' '.join(tokens)


if __name__ == '__main__':
    inputs = sys.argv[1]
    with open(inputs, mode='r') as lines, open(f'{inputs}.clean', 'w') as ofp:
        for line in lines:
            part = line.strip().split('\t')
            clean = clean_repeated(part[-1], max_len=6, min_len=1)
            ofp.write(f'{part[0]}\t{clean}\n')
