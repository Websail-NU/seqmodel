# Sequence Model (Work-In-Progress)

A code base for creating and running sequence models of language. Including
language modeling, definition modeling, and common encoder-decoder stuffs.

## TODO
- Refactor dropout (replace static values of `keep_prob` with placeholders).
- Simplify sampling interfaces.
- Load best model to evaluate when the training is done
- Replace highway-like update with GRU-like update in definition model (to replicate original paper)
