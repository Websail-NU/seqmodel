import sys
import numpy as np

sys.path.insert(0, '../')

import seqmodel as sq  # noqa
from seqmodel import ngram_stats as ns  # noqa

text_filepath = sys.argv[1]
vocab_filepath = sys.argv[2]
window_size = int(sys.argv[3])
output_filepath = sys.argv[4]

lines = sq.read_lines(text_filepath, token_split=' ')
vocab = sq.Vocabulary.from_vocab_file(vocab_filepath)
repeat_count, no_repeat_count = ns.count_repeat_at(lines, vocab, window_size)

np.save(f'{output_filepath}.rep.npy', repeat_count)
np.save(f'{output_filepath}.neg.npy', no_repeat_count)
