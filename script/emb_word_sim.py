import sys
import itertools
from functools import partial

import numpy as np
from scipy import stats

sys.path.insert(0, '../')
import seqmodel as sq  # noqa


def read_eval_file(filepath, vocab=None):
    pairs = []
    scores = []
    with open(filepath) as lines:
        for line in lines:
            line = line.strip()
            part = split(line, filepath)
            if vocab is not None and any((w not in vocab for w in part[:2])):
                continue
            pairs.append(part[:2])
            scores.append(sim_score(part, filepath))
    return pairs, scores


def split(l, filepath):
    if 'mc-30.txt' in filepath:
        return l.split(':')
    if any((x in filepath for x in ('MTURK-771.csv', 'wordsim353.csv'))):
        return l[:-1].split(',')
    else:
        return l.split()


def sim_score(l, filepath):
    if any((x in filepath for x in ('SimLex-999.txt', 'SimVerb-3500.txt'))):
        return float(l[3])
    return float(l[2])


def read_embeddings(emb_path, word_path):
    emb = np.load(emb_path)
    emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
    emb_map = {}
    with open(word_path, mode='r') as lines:
        for i, line in enumerate(lines):
            word = line.strip().split('\t')[0]
            idx = emb_map.setdefault(word, [])
            idx.append(i)
    return emb, emb_map


def dot(w1, w2, emb, emb_map):
    return [np.dot(emb[idx1], emb[idx2])
            for idx1, idx2 in itertools.product(emb_map[w1], emb_map[w2])]


def emb_scores(pairs, emb, emb_map, agg_fn=np.mean):
    sim = partial(dot, emb=emb, emb_map=emb_map)
    return [agg_fn(sim(w1, w2)) for w1, w2 in pairs]


def spearman_corr(sim_x, sim_y):
    rg_x = ranks(sim_x)
    rg_y = ranks(sim_y)
    return np.corrcoef(rg_x, rg_y)[0, 1]


def ranks(array):
    temp = array.argsort()
    ranks = np.empty(len(array), int)
    ranks[temp] = np.arange(len(array))
    return ranks


if __name__ == '__main__':
    emb, emb_map = read_embeddings('tmp_autodef_tran_mask_split_fix/emb.npy',
                                   'data/wn_lemma_senses/all.txt')
    pairs, scores = read_eval_file('data/wordsim_eval/SimLex-999.txt', emb_map)
    s_scores = emb_scores(pairs, emb, emb_map)
    print(stats.spearmanr(s_scores, scores))
    print(spearman_corr(np.array(s_scores), np.array(scores)))
