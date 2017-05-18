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

### Refactor:
- [x] SeqModel
- [ ] Basic running function
- [ ] Scripts to get data
- [ ] Seq2SeqModel
- [ ] Sampled-based running functions
- [ ] Definition data reader
- [ ] DefinitionModel


### Model
- [ ] Make it possible to share variables between 2 different models
- [x] Reuse placeholder variables when creating a model
- [ ] Use sess.partialrun() to cache encoder state

### Run
- [x] Add teacher forcing
- [ ] TD(lambda)
- [ ] Bootstrap last state if not terminal

### Generator


### Scripts


## Goal:
Less classes, more pythonic, and more unit test.

After refactoring codes several times, I think OOP works againt my progress. It makes
the project hard to add/change functions. It is also hard to read the code because
relevant sequences of code are fragmented every where to fit the object's semantics.
