# Sequence Model (Work-In-Progress)

A code base for creating and running sequence models of language. Including
language modeling, definition modeling, and common encoder-decoder stuffs.
**Required python 3.6**.

## Requirements:
- python 3.6
- tensorflow 1.2
- numpy 1.13
- nltk 3.2.4
- six 1.10

## Overview
- dstruct.py: basic data structures and tuples
- generator.py: functions data reader and batch generator
- graph.py: functions to create various types of graphs
- model.py: runnable models from configuration
- run.py: functions to train and evaluate model (data + model -> result)
- util.py: utility functions (dictionary, array, logging, and cmd arguments)

## TODO

### Model
- [x] Fix scan_rnn to support tuple input and output
- [x] Implement dot attention
- [ ] Use LSTMBlockFusedCell
- [ ] Implement Bahdanau attention
- [ ] Beam search decoding
- [ ] Decode with initial state
- [ ] Value Network and A2C

### Run
- [ ] TD(lambda)
- [ ] Bootstrap last state if not terminal

### Generator
- [ ] Option to randomly select sequences of the same encode input

### Scripts
- [ ] Option to select reward function from CLI
- [ ] Fix a bug where `--\_\_:` options are not saved!

### TensorFlow
- [ ] Take advantage of tf.Session.make_callable (need benchmark)

### Bucket list
- [ ] Compile my own TensorFlow
- [ ] Use tf.summary for tensorboard.
- [ ] It would be nice if we do not need to fecth state and feed it back in when
      training a langauge model (sentence dependent).
- [ ] Ensure the indices of `</s>` and `<s>` are 0 and 1,
      and index of `_` is 0 for char-level data
