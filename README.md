# Sequence Model (Work-In-Progress)

A code base for creating and running sequence models of language. Including
language modeling, definition modeling, and common encoder-decoder stuffs.

## Overview
- data: iterates data in batch, output a tuple (see common_tuple.py)
- experiment: defines training procedure and provide friendly interfaces for usage
- model: creates graph nodes that interfaces with data, and provides feed and fetch for sess.run()
- model.module: creates graph nodes

## TODO

### Model
- [ ] Use sequence weight when computing loss
- [ ] Make it possible to share variables between 2 different models
- [ ] Use tensorflow's dynamic_decode
- [ ] Pass `reuse` when building graph (not create an object)
- [ ] Refactor dropout (replace static values of `keep_prob` with placeholders).
- [ ] Replace highway-like update with GRU-like update in definition model (to replicate original paper)

### Agent
- [ ] Rollout more than a batch before updating policy and value networks
- [ ] Bootstrap last state if not terminal

### Data
- [ ] Polysemous words should get lower weight during the training (sequence weight)
- [ ] New reward functions
