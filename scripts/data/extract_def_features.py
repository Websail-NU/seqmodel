import operator
import argparse
import os
import codecs
import sys

sys.path.insert(0, '../../')
from seqmodel.data.vocab import Vocabulary  # noqa


def read_definitions(filepath):
    entries = []
    word_set = set()
    with codecs.open(filepath, 'r', 'utf-8') as ifp:
        for line in ifp:
            parts = line.strip().split('\t')
            entries.append([parts[0], parts[-1]])
            word_set.add(parts[0])
    return entries, word_set


def write_char_vocab(filepath, word_set):
    vocab = {}
    for word in word_set:
        for c in word:
            vocab[c] = vocab.get(c, 0) + 1
    vocab = sorted(vocab.items(),
                   key=operator.itemgetter(1), reverse=True)
    with codecs.open(filepath, 'w', 'utf-8') as ofp:
        ofp.write("<\n")
        ofp.write(">\n")
        for c, count in vocab:
            ofp.write('{}\t{}\n'.format(c, count))


def longest_word(word, definition):
    tokens = definition.split()
    tokens.sort(key=lambda s: len(s))
    tokens.reverse()
    for token in tokens:
        if token != word:
            return token


def first_content_word(word, definition, stopwords):
    for token in definition.split():
        if token != word and token not in stopwords:
            return token
    # fail
    return longest_word(word, definition)


def extract_def_features(filepath, entries, stopwords, vocab, mode):
    with codecs.open(filepath, 'w', 'utf-8') as ofp:
        for word, definition in entries:
            if mode == 'first_content_word':
                feature = first_content_word(word, definition, stopwords)
            else:
                feature = longest_word(word, definition)
            ofp.write(u"{}\t# {}\t{}\t{}\n".format(vocab.w2i(feature),
                                                   feature, word, definition))


parser = argparse.ArgumentParser()
parser.add_argument("text_dir")
parser.add_argument("--char_vocab", type=str,
                    default="char_vocab.txt")
parser.add_argument("--feature_filename", type=str,
                    default="features.txt")
parser.add_argument("--function_words", type=str,
                    default="function_words.txt")
parser.add_argument("--feature_type", type=str,
                    choices=['first_content_word', 'longest_word'],
                    default="first_content_word")
parser.add_argument("--text_filenames", type=str,
                    default="train.txt,valid.txt,test.txt")
args = parser.parse_args()

entries = {}
union_words = set()
for filename in args.text_filenames.split(','):
    filepath = os.path.join(args.text_dir, filename)
    entries[filename], word_set = read_definitions(filepath)
    union_words = union_words.union(word_set)

char_vocab_path = os.path.join(args.text_dir, args.char_vocab)
write_char_vocab(char_vocab_path, union_words)

with codecs.open(os.path.join(args.text_dir,
                              args.function_words), 'r', 'utf-8') as ifp:
    stopwords = set()
    for line in ifp:
        stopwords.add(line.strip())
vocab = Vocabulary.from_vocab_file(
    os.path.join(args.text_dir, 'dec_vocab.txt'))
for filename in args.text_filenames.split(','):
    filepath = os.path.join(args.text_dir, "{}_{}".format(
        os.path.splitext(filename)[0], args.feature_filename))
    extract_def_features(filepath, entries[filename], stopwords, vocab,
                         args.feature_type)
