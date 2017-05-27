# Sequence Model (Work-In-Progress)

A code base for creating and running sequence models of language. Including
language modeling, definition modeling, and common encoder-decoder stuffs.
**Required python 3.6**.

## Requirements:
- python 3.6
- tensorflow 1.1
- numpy 1.12
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
- [ ] Value Network

### Run
- [ ] Add teacher forcing
- [ ] TD(lambda)
- [ ] Bootstrap last state if not terminal

### Generator
- [ ] Reward functions for NLG (i.e. BLEU and token match)

### Scripts
- [ ] Add option to put all data into tf.constant

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
