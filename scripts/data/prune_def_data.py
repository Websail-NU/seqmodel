import os
import sys

def_dir = sys.argv[1]
new_dir = sys.argv[2]
min_count = int(sys.argv[3])

removed_words = set(['(', ')', ',', "'", '.'])

if not os.path.exists(new_dir):
    os.makedirs(new_dir)

vocab = {'<unk>': 0}
with open(os.path.join(def_dir, 'dec_vocab.txt')) as ifp:
    for line in ifp:
        parts = line.split("\t")
        word = parts[0]
        if len(parts) > 1:
            freq = int(parts[1])
        else:
            freq = -1
        if freq != -1 and freq < min_count:
            vocab['<unk>'] += freq
        else:
            vocab[word] = freq

for split in ['train.txt', 'valid.txt', 'test.txt']:
    data_path = os.path.join(def_dir, split)
    new_path = os.path.join(new_dir, split)
    with open(data_path) as ifp, open(new_path, 'w') as ofp:
        for line in ifp:
            parts = line.split('\t')
            other = '\t'.join(parts[0:-1])
            definition = parts[-1]
            tokens = definition.split()
            for i in range(len(tokens)):
                if tokens[i] not in vocab:
                    tokens[i] = '<unk>'
                if tokens[i] in removed_words:
                    tokens[i] = ''
            definition = ' '.join(tokens)
            definition = definition.replace('  ', ' ')
            ofp.write("{}\t{}\n".format(other, definition))
