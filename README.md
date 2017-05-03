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
- [ ] Make it possible to share variables between 2 different models
- [ ] Pass `reuse` when building graph (not create an object)
- [ ] Refactor dropout (replace static values of `keep_prob` with placeholders).
- [ ] Replace highway-like update with GRU-like update in definition model (to replicate original paper)
- [ ] Cache encoder state in ExeSeq2SeqModel such that we don't need to fetch the same thing all the time
- [x] Use sequence weight when computing loss

### Agent
- [x] Rollout more than a batch before updating policy and value networks
- [x] Sample and rerank
- [x] Select argmax or sample rather than fetch distribution
- [ ] Bootstrap last state if not terminal

### Data
- [x] Polysemous words should get lower weight during the training (sequence weight)
- [x] Option to remove duplicate words
- [ ] New reward functions

### Scripts
- [ ] More flexible scripts to create configuration templates
- [ ] Refactor policy gradient definition modeling script
- [x] Compute BLEU score from top hypothesis
