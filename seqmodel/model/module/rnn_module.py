""" A collections of useful functions to create RNN graphs """
import copy
import warnings

import tensorflow as tf

from seqmodel.bunch import Bunch
from seqmodel.model.module import graph_util
from seqmodel.model.module.graph_module import GraphModule


def default_rnn_cell_opt():
    """ Create a default options for RNN cell. """
    return Bunch(cell_class="BasicLSTMCell",
                 cell_opt=Bunch(num_units=128),
                 input_keep_prob=1.0,
                 output_keep_prob=1.0,
                 num_layers=1)


def no_dropout_if_not_training(rnn_cell_opt, is_training):
    """ Change dropout to 0.0. """
    if not is_training:
        new_opt = copy.copy(rnn_cell_opt)
        new_opt.input_keep_prob = 1.0
        new_opt.output_keep_prob = 1.0
        rnn_cell_opt = new_opt
    return rnn_cell_opt


def create_rnn_cell_from_opt(opt, module):
    """ Create an RNN cell from module name. """
    cell_class = eval("{}.{}".format(module, opt.cell_class))
    return cell_class(**opt.cell_opt)


def get_rnn_cell(opt, module="tf.contrib.rnn"):
    """ Create a homogenous RNN cell. """
    cells = []
    for _ in range(opt.num_layers):
        cell = create_rnn_cell_from_opt(opt, module)
        if opt.input_keep_prob < 1.0 or opt.output_keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(
               cell=cell,
               input_keep_prob=opt.input_keep_prob,
               output_keep_prob=opt.output_keep_prob)
        cells.append(cell)
    if opt.num_layers > 1:
        return tf.contrib.rnn.MultiRNNCell(cells)
    return cells[0]


def feed_state(feed_dict, state_vars, state_vals):
    if isinstance(state_vars, dict):
        for k in state_vars:
            feed_state(feed_dict, state_vars[k], state_vals[k])
        return feed_dict
    else:
        feed_dict[state_vars] = state_vals
        return feed_dict


class BasicRNNModule(GraphModule):
    """
    A standard RNN graph
    opt:
        rnn_cell: Option to create RNN cell.
        time_major: If true, inputs and outputs are
                    in time major [time, batch ,..].
                    Internal calculations take [time, batch, depth], so it is
                    recommended to use time_major=True.
        create_zero_initial_state: If true, create zero state and return.
        batch_size: None, only required for create_zero_initial_state.
        logit: (Optional) Options to create output logit.
    """

    def __init__(self, opt, name='basic_rnn', is_training=False):
        GraphModule.__init__(self, name)
        self.opt = opt
        self.is_training = is_training
        self.opt.rnn_cell = no_dropout_if_not_training(
            self.opt.rnn_cell, self.is_training)

    @staticmethod
    def default_opt():
        return Bunch(create_zero_initial_state=False,
                     rnn_cell=default_rnn_cell_opt(),
                     time_major=True,
                     logit=graph_util.default_logit_opt())

    def _build(self, inputs, sequence_length,
               logit_fn=graph_util.create_logit_layer,
               *args, **kwargs):
        """
        Create unrolled RNN graph. Return Decoder output and states.
        args:
            inputs: a tensor for inputs
            sequence_length: a tensor for length of the inputs
        kwargs:
            initial_state: a tensor for initial_state (Default: None or zero)
            rnn_fn: a function or template to unroll RNN cell
                    (Default: tf.nn.dynamic_rnn)
            logit_w: a tensor for computing logit (Default: create a new one)
            logit_fn: a function or template to create logit
                      (Default: graph_util.create_logit_layer)
        """
        final_output = self.step(inputs, sequence_length, *args, **kwargs)
        if self.opt.is_attr_set('logit'):
            final_output = self._add_logit(logit_fn, final_output.cell_output,
                                           final_output, *args, **kwargs)
        return final_output

    def _initialize(self, inputs, initial_state=None,
                    create_zero_initial_state=False, *args, **kwargs):
        self.cell = get_rnn_cell(self.opt.rnn_cell)
        self.initial_state = initial_state
        create_zero_initial_state = (create_zero_initial_state or
                                     self.opt.create_zero_initial_state)
        if (self.initial_state is None and
                create_zero_initial_state):
            batch_dim = 1 if self.opt.time_major else 0
            batch_size = tf.shape(inputs)[batch_dim]
            self.initial_state = self.cell.zero_state(batch_size, tf.float32)
        return inputs, self.cell, self.initial_state

    def _finalize(self, cell_output, final_state, *args, **kwargs):
        final_output = Bunch(cell_output=cell_output, final_state=final_state)
        if self.initial_state is not None:
            final_output.initial_state = self.initial_state
        return final_output

    def _add_logit(self, logit_fn, cell_output, final_output, *args, **kwargs):
        self.logit, self.logit_temp = logit_fn(
            self.opt.logit, cell_output, *args, **kwargs)
        self.prob = tf.nn.softmax(self.logit)
        final_output.logit = self.logit
        final_output.logit_temperature = self.logit_temp
        final_output.distribution = self.prob
        return final_output

    def step(self, inputs, sequence_length, rnn_fn=tf.nn.dynamic_rnn,
             *args, **kwargs):
        inputs, cell, initial_state = self._initialize(inputs, *args, **kwargs)
        cell_output, final_state = rnn_fn(
            cell=cell,
            inputs=inputs,
            sequence_length=sequence_length,
            initial_state=initial_state,
            time_major=self.opt.time_major,
            dtype=tf.float32)
        return self._finalize(cell_output, final_state)

    @property
    def output_time_major(self):
        return self.opt.time_major


class FixedContextRNNModule(BasicRNNModule):
    """
    A RNN graph that uses an additional context vector to transform input or
    output (before logit)
    opt:
        rnn_cell: Option to create RNN cell.
        time_major: If true, inputs and outputs are
                    in time major [time, batch ,..].
                    Internal calculations take [time, batch, depth], so it is
                    recommended to use time_major=True.
        create_zero_initial_state: If true, create zero state and return.
        batch_size: None, only required for create_zero_initial_state.

        logit: (Optional) Options to create output logit.
    """

    def __init__(self, opt, name='fix_context_rnn', is_training=False):
        super(FixedContextRNNModule, self).__init__(opt, name, is_training)

    @staticmethod
    def default_opt():
        return Bunch(BasicRNNModule.default_opt(),
                     transform_input_mode='ignored',
                     transform_output_mode='highway')

    def _build(self, inputs, sequence_length, context,
               logit_fn=graph_util.create_logit_layer,
               *args, **kwargs):
        """ See BasicRNNModule """
        time_dim = 1
        if self.opt.time_major:
            time_dim = 0
        _multiples = [1, 1]
        _multiples.insert(time_dim, tf.shape(inputs)[time_dim])
        tiled_context = tf.tile(tf.expand_dims(context, time_dim), _multiples)
        # TODO: transform input
        final_output = self.step(inputs, sequence_length, *args, **kwargs)
        # transform output
        _carried_output_cell = final_output.cell_output
        if self.opt.rnn_cell.output_keep_prob < 1.0:
            final_output.cell_output = tf.nn.dropout(
                final_output.cell_output, self.opt.rnn_cell.output_keep_prob)
        updated_output = graph_util.create_update_layer(
            final_output.cell_output, tiled_context, _carried_output_cell)
        if self.opt.rnn_cell.output_keep_prob < 1.0:
            updated_output = tf.nn.dropout(
                updated_output, self.opt.rnn_cell.output_keep_prob)
        final_output.updated_output = updated_output
        if self.opt.is_attr_set('logit'):
            final_output = self._add_logit(logit_fn, updated_output,
                                           final_output, *args, **kwargs)
        return final_output

    def _initialize(self, inputs, initial_state=None,
                    create_zero_initial_state=False, *args, **kwargs):

        self.cell = get_rnn_cell(Bunch(self.opt.rnn_cell,
                                       output_keep_prob=1.0))
        self.initial_state = initial_state
        create_zero_initial_state = (create_zero_initial_state or
                                     self.opt.create_zero_initial_state)
        if (self.initial_state is None and
                create_zero_initial_state):
            batch_dim = 1 if self.opt.time_major else 0
            batch_size = tf.shape(inputs)[batch_dim]
            self.initial_state = self.cell.zero_state(batch_size, tf.float32)
        return inputs, self.cell, self.initial_state
