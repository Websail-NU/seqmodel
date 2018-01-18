# Sequence Modeling

A code base for creating and running sequence models of language. Including
language modeling, definition modeling, and common encoder-decoder stuffs.
**Required python 3.6**.

## Requirements

- python 3.6
- tensorflow 1.4
- numpy 1.13
- nltk 3.2.4
- six 1.10
- kenlm

## Overview

- dstruct.py: basic data structures and tuples
- generator.py: functions data reader and batch generator
- graph.py: functions to create various types of graphs
- model.py: runnable models from configuration
- run.py: functions to train and evaluate model (data + model -> result)
- util.py: utility functions (dictionary, array, logging, and cmd arguments)
- ngram_stat.py and ngram_stats.py: various functions for computing global statatistics

## Under development

- cells.py
- marignal.py
- scrap.py

## Usage


### Preprocessing data

By default, this code needs a vocabulary file(s), and text data split into 3 files:
`train.txt`, `valid.txt`, and `test.txt`. There are a few scripts to download and
preprocess datasets in `script/data`. You can use them as an example.

### A test run

To train and evaluate a language model on a test data, run the following command:
```
# Train an RNNLM with defaul configuration
python main_lm.py train test_data/tiny_single/ exp_dir --log_level debug

# Evaluate the same model (in exp_dir)
python main_lm.py eval test_data/tiny_single/ exp_dir --log_level debug
```

You can add `--help` to list all options.

## TODO

- [ ] Implement Bahdanau attention
- [ ] Beam search decoding
- [ ] Ensure the indices of `</s>` and `<s>` are 0 and 1,
      and index of `_` is 0 for char-level data
- [ ] Option to randomly select sequences of the same encode input
- [ ] Value Network and A2C
- [ ] Use tf.summary for tensorboard.

