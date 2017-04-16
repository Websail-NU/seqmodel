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
- [x] Create an actual model object that has a well defined API for creation (graph nodes)
- [x] Placeholders should be in a tuple corresponding to BatchIterator
- [x] Fetch functions will not take any argument
- [x] Feed dictionary functions will take data and state (need to decide on the format)
  - [x] data is in a tuple that corresponds to the graph nodes
  - [x] state is whatever the model needed for itself (possibly a tuple or bunch)
- [ ] Update Definition Model for the executable model interfaces
- [ ] Replace highway-like update with GRU-like update in definition model (to replicate original paper)
- [ ] Refactor dropout (replace static values of `keep_prob` with placeholders).

### Agent
- [ ] Update basic agent to use the executable model interfaces
- [ ] Simplify sampling interfaces, use environment interfaces

### Data
- [x] Creating BatchIterator every time is a pain. We should have a function that creates the same iterator  
  with new data.
- [x] Data in the BatchIterator is not changing, we should use tuple for performance and integrity.
  The fastest way is to have a class, but that's too much code
- [x] BatchIterator will not have a configuration for data files or lists anymore, initialize()  
  function should take argument for such thing
- [x] We should have an Environment class to wrap BatchIterator in RL.
- [ ] The environment interface on sequence iterator does not make much sense
- [ ] Perhaps, an agent should take raw data and use BatchIterator to generate batch
