import sys
import argparse
import codecs
import os
import operator

sys.path.insert(0, '../../')
from seqmodel.data.vocab import _special_symbols  # noqa


def write_vocab(path, vocab, opt):
    opt = vars(opt)
    with codecs.open(path, 'w', 'utf-8') as ofp:
        for k in _special_symbols:
            if opt[k]:
                ofp.write('{}\n'.format(_special_symbols[k]))
        vocab = sorted(vocab.items(),
                       key=operator.itemgetter(1), reverse=True)
        for word, count in vocab:
            ofp.write('{}\t{}\n'.format(word, count))


def collect_vocab_from_file(vocabs, filepath, is_parallel=False):
    with codecs.open(filepath, 'r', 'utf-8') as ifp:
        for line in ifp:
            if is_parallel:
                parts = line.split('\t')
                parts = (parts[0], parts[-1])
            else:
                parts = [line]
            for i, part in enumerate(parts):
                tokens = line.strip().split()
                for token in tokens:
                    vocabs[i][token] = vocabs[i].get(token, 0) + 1


parser = argparse.ArgumentParser()
parser.add_argument("text_dir")
parser.add_argument("--vocab_filename", type=str, default="vocab.txt")
parser.add_argument("--text_filenames", type=str,
                    default="train.txt,valid.txt,test.txt")
parser.add_argument("--parallel_text", action='store_true')
parser.add_argument("--unknown", action='store_true')
parser.add_argument("--start_seq", action='store_true')
parser.add_argument("--end_seq", action='store_true')
parser.add_argument("--end_encode", action='store_true')
parser.add_argument("--start_decode", action='store_true')

args = parser.parse_args()
vocabs = [{}]
if args.parallel_text:
    vocabs.append({})
for filename in args.text_filenames.split(','):
    filepath = os.path.join(args.text_dir, filename)
    collect_vocab_from_file(vocabs, filepath, args.parallel_text)
if len(vocabs) == 1:
    write_vocab(os.path.join(args.text_dir, args.vocab_filename),
                vocabs[0], args)
else:
    write_vocab(os.path.join(args.text_dir, 'enc_' + args.vocab_filename),
                vocabs[0], args)
    write_vocab(os.path.join(args.text_dir, 'dec_' + args.vocab_filename),
                vocabs[1], args)
