import sys
import argparse
import codecs
import os
import operator

sys.path.insert(0, '../../')
from seqmodel.dstruct import Vocabulary  # noqa

special_symbols = Vocabulary.special_symbols


def write_vocab(path, vocab, opt):
    opt = vars(opt)
    with codecs.open(path, 'w', 'utf-8') as ofp:
        if not opt['char_level']:
            for k in special_symbols:
                if opt[k]:
                    ofp.write('{}\n'.format(special_symbols[k]))
        else:
            ofp.write('_\n')
            vocab.pop('_', None)
        vocab = sorted(vocab.items(),
                       key=operator.itemgetter(1), reverse=True)
        for word, count in vocab:
            ofp.write('{}\t{}\n'.format(word, count))


def collect_vocab_from_file(
        vocabs, filepath, is_parallel=False, part_indices=(0, -1), char_level=False,
        convert_word_to_chars=False):
    with codecs.open(filepath, 'r', 'utf-8') as ifp:
        for line in ifp:
            if is_parallel:
                parts = list(line.strip().split('\t'))
                parts = [parts[int(i)] for i in part_indices]
            else:
                parts = [line]
            for i, part in enumerate(parts):
                tokens = part.strip().split()
                if char_level:
                    tokens = '<' + '_'.join(tokens) + '>'
                elif convert_word_to_chars:
                    tokens = '_'.join(tokens)
                for token in tokens:
                    vocabs[i][token] = vocabs[i].get(token, 0) + 1


parser = argparse.ArgumentParser()
parser.add_argument('text_dir')
parser.add_argument('--vocab_filename', type=str, default='vocab.txt')
parser.add_argument('--text_filenames', type=str,
                    default='train.txt,valid.txt,test.txt')
parser.add_argument('--parallel_text', action='store_true')
parser.add_argument('--part_indices', type=str, default='0,-1')
parser.add_argument('--char_level', action='store_true')
parser.add_argument('--unknown', action='store_true')
parser.add_argument('--start_seq', action='store_true')
parser.add_argument('--end_seq', action='store_true')
parser.add_argument('--end_encode', action='store_true')
parser.add_argument('--convert_word_to_chars', action='store_true')

args = parser.parse_args()
vocabs = [{}]
if args.parallel_text:
    num_vocabs = len(args.part_indices.split(','))
    for __ in range(1, num_vocabs):
        vocabs.append({})
for filename in args.text_filenames.split(','):
    filepath = os.path.join(args.text_dir, filename)
    collect_vocab_from_file(
        vocabs, filepath, args.parallel_text, args.part_indices.split(','),
        args.char_level, args.convert_word_to_chars)
if len(vocabs) == 1:
    write_vocab(os.path.join(args.text_dir, args.vocab_filename),
                vocabs[0], args)
else:
    write_vocab(os.path.join(args.text_dir, 'enc_' + args.vocab_filename),
                vocabs[0], args)
    write_vocab(os.path.join(args.text_dir, 'dec_' + args.vocab_filename),
                vocabs[1], args)
