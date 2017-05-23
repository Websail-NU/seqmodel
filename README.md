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
- [x] Make it possible to share variables between 2 different models
- [x] Reuse placeholder variables when creating a model
- [x] Dynamic decode: greedy gen to fixed length
- [ ] Dynamic decode: add stop id check and multinomial sample
- [ ] DefinitionModel
- [ ] Value Network

### Run
- [ ] Sampled-based running functions
- [ ] Add teacher forcing
- [ ] TD(lambda)
- [ ] Bootstrap last state if not terminal

### Generator
- [ ] Definition data reader

### Scripts
- [ ] Scripts to get definition data
- [ ] Add option to put all data into tf.constant

### Bucket list
- [ ] Use sess.partialrun() to cache encoder state, then we can decode and update
      the networks without running encoding twice!
- [ ] It would be nice if we do not need to fecth state and feed it back in when
      training a langauge model (sentence dependent).
- [ ] Ensure the indices of `</s>` and `<s>` are 0 and 1.
