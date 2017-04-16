# Sequence Model (Work-In-Progress)

A code base for creating and running sequence models of language. Including
language modeling, definition modeling, and common encoder-decoder stuffs.


## Data, Fetch, and Feed Documentation

### Fetch

#### ModelBase
- Args:
  - model: Bunch of reference graph nodes
  - \_custom_fetch: Bunch of graph nodes to fetch for debugging
- Kwargs:
  - None
- Returns:
  - Empty Bunch or \_custom_fetch

#### SeqModel (ModelBase)
- *No implementation*

#### BasicSeqModel (SeqModel)
- Args:
  - model: Bunch of reference graph nodes
  - is_sampling: a boolean indicating whether this fetch for sampling  
                 Default: False
- Kwargs:
  - \_custom_fetch: Bunch of graph nodes to fetch for debugging (*ModelBase*)
- Returns:
  - losses = model.losses
  - state = model.decoder_output.final_state
  - logit = model.decoder_output.logit, if is_sampling is True
  - distribution = model.decoder_output.distribution, if is_sampling is True
  - \_custom_fetch, if not empty or overwritten (*ModelBase*)

#### Seq2SeqModel (ModelBase)
- *No implementation*

#### BasicSeq2SeqModel (Seq2SeqModel)
- Args:
  - model: Bunch of reference graph nodes
  - is_sampling: a boolean indicating whether this fetch for sampling  
                 Default: False (*BasicSeqModel*)
- Kwargs:
  - \_custom_fetch: Bunch of graph nodes to fetch for debugging (*ModelBase*)
- Returns:
  - losses = model.losses (*BasicSeqModel*)
  - state =
    - encoder_context = model.encoder_output **This should be encoder final state**
    - decoder_state = model.decoder_output.final_state
  - logit = model.decoder_output.logit, if is_sampling is True (*BasicSeqModel*)
  - distribution = model.decoder_output.distribution, if is_sampling is True (*BasicSeqModel*)
  - \_custom_fetch (if not empty, or overwritten) (*ModelBase*)

## TODO

### Model
- [ ] Create an actual model object that has a well defined API for creation (graph nodes)
- [ ] Placeholders should be in a tuple corresponding to BatchIterator
- [ ] Fetch functions will not take any argument
- [ ] Feed dictionary functions will take data and state (need to decide on the format)
  - [ ] data is in a tuple that corresponds to the graph nodes
  - [ ] state is whatever the model needed for itself (possibly a tuple or bunch)
- [ ] Replace highway-like update with GRU-like update in definition model (to replicate original paper)
- [ ] Refactor dropout (replace static values of `keep_prob` with placeholders).

### Agent
- [ ] Simplify sampling interfaces, use environment interfaces

### Data
- [x] Creating BatchIterator every time is a pain. We should have a function that creates the same iterator  
  with new data.
- [x] Data in the BatchIterator is not changing, we should use namedtuple for performance and integrity.
  The fastest way is to have a class, but that's too much code
- [x] BatchIterator will not have a configuration for data files or lists anymore, initialize()  
  function should take argument for such thing
- [x] We should have an Environment class to wrap BatchIterator in RL.
- [ ] The environment interface on sequence iterator does not make much sense
- [ ] Perhaps, an agent should take raw data and use BatchIterator to generate batch
