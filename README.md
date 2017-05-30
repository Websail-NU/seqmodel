# Sequence Model (Work-In-Progress)

A code base for creating and running sequence models of language. Including
language modeling, definition modeling, and common encoder-decoder stuffs.
**Required python 3.6**.

## Requirements:
- python 3.6
- tensorflow 1.1
- numpy 1.12
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
- [ ] Decode with option to return likelihood
- [ ] Value Network

### Run
- [ ] Add teacher forcing
- [ ] TD(lambda)
- [ ] Bootstrap last state if not terminal

### Generator
- [x] Reward functions for NLG (i.e. BLEU and token match)

### Scripts
- [x] Merge _main.decode function into _main function
- [x] Decoding option group
- [ ] BLEU evaluation script
- [ ] Support multiple evaluation files

### TensorFlow
- [ ] Compile my own TensorFlow
- [ ] Upgrade to TensorFlow v1.2 (Pending for its release)
- [ ] Take advantage of tf.Session.make_callable (Pending for its release)

### Bucket list
- [ ] Use tf.summary for tensorboard.
- [ ] Wrap manual attention in definition model to RNNCell
- [ ] It would be nice if we do not need to fecth state and feed it back in when
      training a langauge model (sentence dependent).
- [ ] Ensure the indices of `</s>` and `<s>` are 0 and 1,
      and index of `_` is 0 for char-level data
