# Sequence Model (Work-In-Progress)

A code base for creating and running sequence models of language. Including
language modeling, definition modeling, and common encoder-decoder stuffs.
**Required python 3.6**.

## Overview
- dstruct.py: basic data structures and tuples
- generator.py: functions data reader and batch generator
- graph.py: functions to create various types of graphs
- model.py: runnable models from configuration
- run.py: functions to train and evaluate model (data + model -> result)
- util.py: utility functions (dictionary, array, and logging)

## TODO

### Model
- [x] Make it possible to share variables between 2 different models
- [x] Reuse placeholder variables when creating a model
- [x] Dynamic decode: greedy gen to fixed length
- [ ] Dynamic decode: add stop id check and multinomial sample
- [ ] DefinitionModel

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

### Refactor:
- [ ] Move function in contrib package and add test
- [ ] Less classes, more pythonic, and more unit test.

### Bucket list
- [ ] Use sess.partialrun() to cache encoder state
