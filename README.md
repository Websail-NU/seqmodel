# Sequence Model (Work-In-Progress)

A code base for creating and running sequence models of language. Including
language modeling, definition modeling, and common encoder-decoder stuffs.

## Overview
- data: iterates data in batch, output a tuple (see common_tuple.py)
- experiment: defines training procedure and provide friendly interfaces for usage
- model: creates graph nodes that interfaces with data, and provides feed and fetch for sess.run()
- model.module: creates graph nodes

## TODO

### Refactor
- [ ] Move to python 3
- [ ] Less OOP
- [ ] Finish SeqModel

### Model
- [ ] Make it possible to share variables between 2 different models
- [ ] Pass `reuse` when building graph (not create an object)
- [ ] Reuse placeholder variables when creating a model with `reuse=True`
- [x] Replace highway-like update with GRU-like update in definition model (to replicate original paper)
- [ ] Cache encoder state in ExeSeq2SeqModel such that we don't need to fetch the same thing all the time

### Agent
- [x] Add teacher forcing
- [ ] TD(lambda)
- [ ] Bootstrap last state if not terminal

### Data
- [ ] New reward functions
- [ ] Handle different padding ids when concat data tuples

### Scripts
- [ ] More flexible scripts to create configuration templates
- [ ] Refactor policy gradient definition modeling script
