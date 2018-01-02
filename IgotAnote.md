### python
- ipython auto reload

    ```python
    %load_ext autoreload
    %autoreload 2
    ```
- pytest with printing `pytest -s seqmodel/some_test.py::Class::method`


### Refactor
- I did code overhaul too many times. I think the problem stems from OOP is not quite
  suitable for research code.
- We add or change functions all the times, and every time we have to find
  an object that fits the semantics of the functions. If it happens to be in a class
  hierarchy, suddenly we have to change a lot of code.
- The logic in the code is scattered around into many different files, it is difficult
  to track down what a single function does.
- Inheritance should not be overused. Encapsulation is easier with closure and partial.
- *less OOP, fewer files, more procedural, lot of partial*.


### Dictionary default
- `dict.get()` and `dict.setdefault()` are greate for shortening code.
- But they can add unintended computation cost because of the *default*.
- In many cases such as creating a tuple, those functions will run faster than
  having `if: ... else: ...` and hashing twice.
- We need to be careful about costly functions and functions that have side-effect
  i.e. `tf.almost_anything()`


### Actor-Critic implementation
- The model should compute the advantage in tensorflow.
- Essentially we define a new loss function with different token weights.
- In code, we provide nodes and scopes to create a value network elsewhere, and then
  create new loss graph for both the policy network and the value network.
  - Value network will need encoder input nodes, and scopes if sharing variable.
  - Do not forget to stop gradient of the policy loss to value network.
- Pro: All in one sess.run().
- Con: Less flexibility on how to update policy and value network
  (mitigated by setting different weight on the loss function).


### Data in GPU memory
- In many cases, we can put all of the data into GPU memory.
- Then we only need to feed in a position vector to gather the data.
- The problems are:
    1. `tf.gather` (or `tf.nn.embedding`) does not work support integer data on GPU.
       We could change the data type of float, but then we will need to cast it back.
    2. `tf.gather` only selects from the first k dimensions, meaning we have to transpose
       the batch data.
- All-in-all, we do not gain much advantage as we do in Torch7 (lua)
  where we can just `whatever:cuda()` on almost anything.


### Sublime Text Packages
- jedi - Python autocompletion
- sublime-rsync-ssh
- SublimeLinter-pydocstyle
- Git
- Whitespace
- Figlet Big ASCII Text
