import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, '../')
from seqmodel.dstruct import Vocabulary  # noqa
from seqmodel.generator import read_lines  # noqa


def empirical_repetition(filepath, max_k, vocab):
    repeat_after_count = np.zeros((vocab.vocab_size, max_k), dtype=np.int32)
    last_seen_dict = defaultdict(int)
    lines = read_lines([filepath], token_split=' ')
    index = 0
    for line_parts in lines:
        line = line_parts[0]
        for token in line:
            index += 1
            last_seen_idx = last_seen_dict[token]
            last_seen_dict[token] = index
            if last_seen_idx == 0:
                continue
            distance = index - last_seen_idx
            if distance - 1 < max_k:
                repeat_after_count[vocab.w2i(token), distance - 1] += 1

    return repeat_after_count


if __name__ == '__main__':
    vocab = Vocabulary.from_vocab_file(sys.argv[2])
    re = empirical_repetition(sys.argv[1], 100, vocab)
