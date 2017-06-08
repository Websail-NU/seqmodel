import warnings
from contextlib import contextmanager
import six
from pydoc import locate
from functools import partial

import numpy as np
import tensorflow as tf

from seqmodel import dstruct


__all__ = ['_safe_div', 'tfph_collection', 'create_2d_tensor', 'matmul', 'create_cells',
           'create_rnn', 'select_rnn', 'select_nested_rnn', 'create_tdnn', 'maybe_scope',
           'create_highway_layer', 'create_gru_layer', 'get_seq_input_placeholders',
           'get_seq_label_placeholders', 'create_lookup', 'get_logit_layer',
           'select_from_logit', 'create_xent_loss', 'create_ent_loss',
           'create_slow_feature_loss', 'create_l2_loss', 'create_train_op',
           'empty_tfph_collection', 'scan_rnn_no_mask', 'create_decode']


_global_collections = {}


@contextmanager
def tfph_collection(collect_key, add_to_collection):
    """return a function to get value from global collection, and update if needed."""
    global _global_collections
    collection = {}
    if collect_key is not None:
        collection = _global_collections.setdefault(collect_key, {})
    temp = {}

    def _get_and_set(name, dtype, shape):
        assert name not in temp, f'{name} is already existed in this context.'
        if name not in collection:
            ph = tf.placeholder(dtype, shape, name)
            temp[name] = ph
        else:
            ph = collection[name]
        return ph

    yield _get_and_set

    if add_to_collection and collect_key is not None:
        collection.update(temp)


def empty_tfph_collection(collect_key):
    """clean global placeholder collection that we use temporarily"""
    global _global_collections
    if collect_key == '*':
        _global_collections = {}
    else:
        _global_collections[collect_key] = {}


@contextmanager
def maybe_scope(scope=None, reuse=False):
    if scope is not None:
        with tf.variable_scope(scope, reuse=reuse) as _scope:
            yield _scope
    else:
        yield None


def create_2d_tensor(dim1, dim2, trainable=True, init=None, name='tensor'):
    if init is None:
        return tf.get_variable(name, [dim1, dim2], trainable=trainable)
    else:
        init = np.load(init) if isinstance(init, six.string_types) else init
        return tf.get_variable(
            name, trainable=trainable, initializer=tf.constant(init, dtype=tf.float32))


def _safe_div(numerator, denominator, name='safe_div'):
    """Computes a safe divide which returns 0 if the denominator is zero."""
    return tf.where(tf.equal(denominator, 0),
                    tf.zeros_like(numerator),
                    tf.div(numerator, denominator),
                    name=name)


def matmul(mat, mat2d, transpose_b=False):
    """return multiplication of 3D tensor and 2D tensor."""
    if len(mat.get_shape()) < 3:
        return tf.matmul(mat, mat2d, transpose_b=transpose_b)
    mat3d = mat
    mat3d_dim = int(mat3d.get_shape()[-1])
    if transpose_b:
        mat2d_dim = int(mat2d.get_shape()[0])
    else:
        mat2d_dim = int(mat2d.get_shape()[-1])
    output_shapes = tf.unstack(tf.shape(mat3d))
    output_shapes[-1] = mat2d_dim
    output_shape = tf.stack(output_shapes)
    flat_mat3d = tf.reshape(mat3d, [-1, mat3d_dim])
    outputs = tf.matmul(flat_mat3d, mat2d, transpose_b=transpose_b)
    return tf.reshape(outputs, output_shape)

#####################################
#     ######  ########  #######     #
#    ##    ## ##       ##     ##    #
#    ##       ##       ##     ##    #
#     ######  ######   ##     ##    #
#          ## ##       ##  ## ##    #
#    ##    ## ##       ##    ##     #
#     ######  ########  ##### ##    #
#####################################
# Most of below functions assume time major input and output, unless specified


def create_cells(num_units, num_layers, cell_class=tf.contrib.rnn.BasicLSTMCell,
                 reuse=False, in_keep_prob=1.0, out_keep_prob=1.0, state_keep_prob=1.0,
                 variational=False, input_size=None, drop_out_last_layer=True):
    """return an RNN cell with optionally DropoutWrapper and MultiRNNCell."""
    cells = []
    for layer in range(num_layers):
        if isinstance(cell_class, six.string_types):
            cell_class = locate(cell_class)
        cell = cell_class(num_units, reuse=reuse)
        any_drop = any(kp < 1.0 for kp in [in_keep_prob, out_keep_prob, state_keep_prob])
        last_layer_drop = ((layer == num_layers - 1 and drop_out_last_layer) or
                           layer != num_layers - 1)
        if any_drop and last_layer_drop:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, in_keep_prob, out_keep_prob, state_keep_prob, variational,
                input_size, tf.float32)
        input_size = cell.output_size
        if not variational:
            in_keep_prob = 1.0  # remove double dropout when stacking cells
        cells.append(cell)
    if num_layers == 1:
        final_cell = cells[0]
    else:
        final_cell = tf.contrib.rnn.MultiRNNCell(cells)
    return final_cell


def scan_rnn(cell, inputs, sequence_length, initial_state=None, dtype=tf.float32,
             scope='rnn', mask_output=True, **_kwargs):
    """dynamically unroll cell to max(len(inputs)), and select last relevant state.
    IMPORTANT sequence_length shoule be at least 1, otherwise this function will return
    the first state even thought it is not relevant."""

    def step(acc, x_t):
        output, state = cell(x_t, acc[1])
        return output, state

    with tf.variable_scope(scope):
        batch_size = tf.shape(inputs)[1]  # time major
        output, states = tf.scan(
            step, inputs, name='scan_rnn',
            initializer=(tf.zeros((batch_size, cell.output_size),
                                  dtype=dtype, name='scan_rnn_init'),
                         initial_state))
        final_state = select_nested_rnn(states, tf.nn.relu(sequence_length - 1))
        if mask_output:
            max_len = tf.shape(inputs)[0]
            mask = tf.expand_dims(
                tf.sequence_mask(sequence_length, max_len, tf.float32), -1)
            output = tf.multiply(output, tf.transpose(mask, (1, 0, 2)))
    return output, final_state


scan_rnn_no_mask = partial(scan_rnn, mask_output=False)


def create_rnn(cell, inputs, sequence_length=None, initial_state=None,
               rnn_fn=tf.nn.dynamic_rnn):
    """return output (all time steps), initial state, and final state in time major."""
    if isinstance(rnn_fn, six.string_types):
        rnn_fn = locate(rnn_fn)
    if initial_state is None:
        batch_size = tf.shape(inputs)[1]  # time major
        initial_state = cell.zero_state(batch_size, tf.float32)
    cell_output, final_state = rnn_fn(
        cell=cell, inputs=inputs, sequence_length=sequence_length,
        initial_state=initial_state, time_major=True, dtype=tf.float32)
    return cell_output, initial_state, final_state


def select_nested_rnn(maybe_tuple, time_step):
    """return possibly nested tensor at the time_step (time major)."""
    if isinstance(maybe_tuple, tuple):
        select = tuple([select_nested_rnn(item, time_step) for item in maybe_tuple])
        if hasattr(maybe_tuple, '_make'):
            select = maybe_tuple._make(select)
    else:
        select = select_rnn(maybe_tuple, time_step)
    return select


def select_rnn(tensor, time_step):
    """return tensor at the time_step (time major). This is similar to numpy
    tensor[time_step, :, :] where time_step can be 1D array."""
    time_step = tf.expand_dims(time_step, axis=-1)
    range_ = tf.expand_dims(tf.range(start=0, limit=tf.shape(tensor)[1]), axis=-1)
    idx = tf.concat([time_step, range_], axis=-1)
    return tf.gather_nd(tensor, idx)


def create_tdnn(inputs, sequence_length=None, filter_widths=[2, 3, 4, 5, 6],
                num_filters=[10, 30, 40, 40, 40], activation_fn=tf.tanh):
    """return time-delayed network as a tensor of [batch, sum num_filters].
    This function expects batch major input."""
    if isinstance(activation_fn, six.string_types):
        activation_fn = locate(activation_fn)
    input_dim = inputs.get_shape()[-1]
    if sequence_length is not None:
        max_len = tf.shape(inputs)[1]
        mask = tf.expand_dims(tf.sequence_mask(sequence_length, max_len, tf.float32), -1)
        inputs = tf.multiply(inputs, mask)
    inputs = tf.expand_dims(inputs, 1)
    layers = []
    for width, out_channels in zip(filter_widths, num_filters):
        filters = tf.get_variable(
            'filter_{}'.format(width),
            [1, width, input_dim, out_channels], dtype=tf.float32)
        conv2d = tf.nn.conv2d(inputs, filters, [1, 1, 1, 1], 'SAME')
        if activation_fn is not None:
            conv2d = activation_fn(conv2d)
        max_pool = tf.squeeze(tf.reduce_max(conv2d, 2), axis=1)
        layers.append(max_pool)
    return tf.concat(layers, axis=1) if len(layers) > 1 else layers[0]


def create_highway_layer(transform, extra, carried):
    """return updated carried using Highway-like update function.
    (https://arxiv.org/abs/1505.00387)"""
    transform_dim = int(transform.get_shape()[-1])
    carried_dim = int(carried.get_shape()[-1])
    extra_dim = int(extra.get_shape()[-1])
    assert transform_dim == carried_dim, 'transform and carried must have the same size'
    in_size = transform_dim + extra_dim
    out_size = carried_dim * 2
    gate_w = tf.get_variable('gate_w', [in_size, out_size])
    _arr = np.zeros((out_size))
    _arr[:] = -1
    gate_b = tf.get_variable('gate_b', initializer=tf.constant(
        _arr, dtype=tf.float32))
    z = matmul(tf.concat([transform, extra], -1), gate_w) + gate_b
    t = tf.sigmoid(tf.slice(z, [0, 0, 0], [-1, -1, carried_dim]))
    h = tf.tanh(tf.slice(z, [0, 0, carried_dim], [-1, -1, -1]))
    return tf.multiply(h - carried, t) + carried, t


def create_gru_layer(transform, extra, carried):
    """return updated carried using GRU-like update function.
    (https://arxiv.org/abs/1612.00394)"""
    transform_dim = int(transform.get_shape()[-1])
    carried_dim = int(carried.get_shape()[-1])
    extra_dim = int(extra.get_shape()[-1])
    assert transform_dim == carried_dim, 'transform and carried must have the same size'
    in_size = transform_dim + extra_dim
    out_size = carried_dim + extra_dim
    zr_w = tf.get_variable('gate_zr_w', [in_size, out_size])
    zr_b = tf.get_variable('gate_zr_b', [out_size])
    zr = tf.sigmoid(matmul(tf.concat([extra, transform], -1), zr_w) + zr_b)
    if len(transform.get_shape()) == 2:
        z = tf.slice(zr, [0, 0], [-1, carried_dim])
        r = tf.slice(zr, [0, carried_dim], [-1, -1])
    else:
        z = tf.slice(zr, [0, 0, 0], [-1, -1, carried_dim])
        r = tf.slice(zr, [0, 0, carried_dim], [-1, -1, -1])
    h_w = tf.get_variable('h_w', [in_size, carried_dim])
    h_b = tf.get_variable('h_b', [carried_dim])
    scaled_extra = tf.multiply(r, extra)
    h = tf.tanh(matmul(tf.concat([scaled_extra, transform], -1), h_w) + h_b)
    return tf.multiply(h - carried, z) + carried, zr


def create_decode(emb_var, cell, logit_w, initial_state, initial_inputs, initial_finish,
                  logit_b=None, logit_temperature=None, min_len=1, max_len=40, end_id=0,
                  cell_scope=None, reuse_cell=True, back_prop=False, select_fn=None,
                  late_attn_fn=None):
    if select_fn is None:
        select_fn = partial(tf.argmax, axis=-1)
    gen_ta = tf.TensorArray(dtype=tf.int32, size=min_len, dynamic_size=True)
    init_values = (tf.constant(0), initial_inputs, initial_state, gen_ta, initial_finish)

    def cond(t, _inputs, _state, _out_ta, finished):
        return tf.logical_and(t < max_len, tf.logical_not(tf.reduce_all(finished)))

    def step(t, inputs, state, out_ta, finished):
        input_emb = tf.nn.embedding_lookup(emb_var, inputs)
        with maybe_scope(cell_scope, reuse=reuse_cell):
            with tf.variable_scope('rnn', reuse=True):
                output, new_state = cell(input_emb, state)
        if late_attn_fn is not None:
            output = late_attn_fn(output)
        logit = tf.matmul(output, logit_w, transpose_b=True)
        if logit_b is not None:
            logit = logit + logit_b
        if logit_temperature is not None:
            logit = logit / logit_temperature
        next_token = tf.cast(select_fn(logit), tf.int32)
        out_ta = out_ta.write(t, next_token)
        finished = tf.logical_or(finished, tf.equal(next_token, end_id))
        return t + 1, next_token, new_state, out_ta, finished

    _t, _i, _s, result, _f = tf.while_loop(cond, step, init_values, back_prop=back_prop,
                                           parallel_iterations=10)
    # parallel_iterations does not matter much here.
    return result.stack()


#######################################
#    ####          ##     #######     #
#     ##          ##     ##     ##    #
#     ##         ##      ##     ##    #
#     ##        ##       ##     ##    #
#     ##       ##        ##     ##    #
#     ##      ##         ##     ##    #
#    ####    ##           #######     #
#######################################
# get_X() have side-effect on tensorflow collection
# if add_to_collection is True (default), the functions will add placeholders to
# global collection


def get_seq_input_placeholders(prefix='decoder', add_to_collection=True,
                               collect_key='model_inputs'):
    """return input and sequence length placeholders,
    create if not existed in collection. If add_to_collection is True, this function
    adds placeholders to tensorflow collection."""
    # collect_key = f'{tf.get_variable_scope().name}/placeholders'
    with tfph_collection(collect_key, add_to_collection) as get:
        input_key = f'{prefix}_input'
        seq_len_key = f'{prefix}_seq_len'
        input_ = get(input_key, tf.int32, (None, None))
        seq_len_ = get(seq_len_key, tf.int32, (None, ))
    return input_, seq_len_


def get_seq_label_placeholders(label_dtype=tf.int32, prefix='decoder',
                               add_to_collection=True, collect_key='model_inputs'):
    """return label, token weight, and sequence weight placeholders,
    create if not existed in collection. If add_to_collection is True, this function
    adds placeholders to tensorflow collection."""
    with tfph_collection(collect_key, add_to_collection) as get:
        label_key = f'{prefix}_label'
        tk_w_key = f'{prefix}_token_weight'
        seq_w_key = f'{prefix}_seq_weight'
        label = get(label_key, label_dtype, (None, None))
        tk_w = get(tk_w_key, tf.float32, (None, None))
        seq_w = get(seq_w_key, tf.float32, (None, ))
    return label, tk_w, seq_w


def create_lookup(inputs, emb_vars=None, onehot=False, vocab_size=None, dim=None,
                  trainable=True, init=None, prefix='input',
                  emb_name='embedding'):
    """return lookup, and embedding variable (None if onehot)"""
    if onehot:
        assert vocab_size is not None, 'onehot needs vocab_size to be set.'
        lookup = tf.one_hot(inputs, vocab_size, axis=-1, dtype=tf.float32,
                            name=f'{prefix}_lookup')
        return lookup, None  # RETURN IS HERE TOO!
    if emb_vars is None:
        size_is_not_none = vocab_size is not None and dim is not None
        assert size_is_not_none or init is not None,\
            'If emb_vars and init is None, vocab_size and dim must be set.'
        emb_vars = create_2d_tensor(vocab_size, dim, trainable, init, emb_name)
    lookup = tf.nn.embedding_lookup(emb_vars, inputs, name=f'{prefix}_lookup')
    return lookup, emb_vars


def get_logit_layer(inputs, logit_w=None, logit_b=None, output_size=None,
                    use_bias=True, temperature=None, trainable=True,
                    init=None, add_project=False, project_size=-1, project_act=tf.tanh,
                    prefix='output', add_to_collection=True, collect_key='model_inputs'):
    """return logit with temperature layer and variables"""
    if logit_w is None:
        input_dim = int(inputs.get_shape()[-1])
        logit_w = create_2d_tensor(output_size, input_dim, trainable, init=init,
                                   name=f'{prefix}_logit_w')
    if add_project:
        project_size = project_size if project_size > 0 else input_dim
        proj_w = tf.get_variable(f'{prefix}_logit_proj', shape=(input_dim, project_size),
                                 dtype=tf.float32)
        logit_w = tf.matmul(logit_w, proj_w)
        if isinstance(project_act, six.string_types):
            project_act = locate(project_act)
        if project_act is not None:
            logit_w = project_act(logit_w)
    logit = matmul(inputs, logit_w, transpose_b=True)
    if use_bias:
        if logit_b is None:
            logit_b = tf.get_variable(f'{prefix}_logit_b', [output_size],
                                      dtype=tf.float32)
        logit = logit + logit_b
    if temperature is None:
        with tfph_collection(collect_key, add_to_collection) as get:
            temp_key = f'{prefix}_logit_temperature'
            temperature = get(temp_key, tf.float32, shape=[])
    logit = logit / temperature
    return logit, temperature, logit_w, logit_b


def select_from_logit(logit, distribution=None):
    if distribution is None:
        distribution = tf.nn.softmax(logit)
    max_idx = tf.argmax(logit, axis=-1)
    max_prob = tf.reduce_max(distribution, axis=-1)
    logit_shape = tf.shape(logit)
    logit_dim = logit_shape[-1]
    logit_2d = tf.reshape(logit, [-1, logit_dim])
    dist_2d = tf.reshape(distribution, [-1, logit_dim])
    sample_idx = tf.cast(tf.multinomial(logit_2d, 1), dtype=tf.int32)
    gather_idx = tf.expand_dims(
        tf.range(start=0, limit=tf.shape(sample_idx)[0]), axis=-1)
    gather_idx = tf.concat([gather_idx, sample_idx], axis=-1)
    sample_prob = tf.gather_nd(dist_2d, gather_idx)
    sample_idx = tf.reshape(sample_idx, logit_shape[:-1])
    sample_prob = tf.reshape(sample_prob, logit_shape[:-1])
    max_tuple = dstruct.IndexScoreTuple(max_idx, max_prob)
    sample_tuple = dstruct.IndexScoreTuple(sample_idx, sample_prob)
    return distribution, max_tuple, sample_tuple


##############################################
#    ##        #######   ######   ######     #
#    ##       ##     ## ##    ## ##    ##    #
#    ##       ##     ## ##       ##          #
#    ##       ##     ##  ######   ######     #
#    ##       ##     ##       ##       ##    #
#    ##       ##     ## ##    ## ##    ##    #
#    ########  #######   ######   ######     #
##############################################


def create_xent_loss(logit, label, weight, seq_weight=None, loss_denom=None):
    """return negative log likelihood (cross-entropy loss).
    Return includes NLL/sum weight, NLL[/loss_denom], token NLL"""
    # Internally logits and labels are reshaped into 2D and 1D...
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logit, labels=label)
    if seq_weight is not None:
        weight = tf.multiply(weight, seq_weight)
    sum_loss = tf.reduce_sum(tf.multiply(loss, weight))
    mean_loss = _safe_div(sum_loss, tf.reduce_sum(weight))
    if loss_denom is not None:
        training_loss = _safe_div(sum_loss, loss_denom)
    else:
        training_loss = sum_loss
    return mean_loss, training_loss, loss


def create_ent_loss(distribution, weight, seq_weight=None):
    """return average negative entropy."""
    if seq_weight is not None:
        weight = tf.multiply(weight, seq_weight)
    neg_entropy = tf.reduce_sum(
        distribution * tf.log(distribution + 1e-6), axis=-1)
    sum_loss = tf.reduce_sum(neg_entropy * weight)
    num_dist = tf.reduce_sum(weight)
    mean_loss = _safe_div(sum_loss, num_dist)
    return mean_loss


def create_slow_feature_loss(feature, weight, delta=1.0):
    """return a constrastive slow feature analysis loss
    Args:
        feature: A tensor of shape [batch, time, dim]
        weight: A tensor of shape [batch, time, time].
                For correctness, this tensor should only contain 0.0 and 1.0;
                and each [i, time, time] matrix should be an upper triangular
                matrix.
    Return:
        loss: A tensor of shape [batch, time, time]
        batch_loss: sum loss, averaged by batch size"""
    r = tf.expand_dims(tf.reduce_sum(feature * feature, -1), axis=-1)
    D = r - 2 * tf.matmul(A, tf.transpose(A, perm=[0, 2, 1]))
    D = D + tf.transpose(r, perm=[0, 2, 1])
    R2 = tf.multiply(D, weight)
    if delta > 0:
        n_weight = 1 - weight
        n_weight = tf.matrix_band_part(n_weight, 0, -1)
        n_weight = n_weight - tf.matrix_band_part(n_weight, 0, 0)
        R2_n = tf.multiply(tf.nn.relu(delta - D), n_weight)
        R2 = R2 + R2_n
    batch_size = tf.shape(feature)[0]
    return R2, tf.reduce_sum(R2) / tf.cast(batch_size, tf.float32)


def create_l2_loss(var_list):
    """return L2 norm of all variables in var_list."""
    l2_loss = tf.reduce_sum(tf.add_n(
        [tf.nn.l2_loss(var) for var in var_list]))
    return l2_loss


def create_train_op(loss, optim_class=tf.train.AdamOptimizer, learning_rate=0.001,
                    clip_gradients=5.0, **optim_kwarg):
    """return train operation graph"""
    if isinstance(optim_class, six.string_types):
        optim_class = locate(optim_class)
    optim = optim_class(learning_rate=learning_rate, **optim_kwarg)
    g_v_pairs = optim.compute_gradients(loss)
    grads, tvars = [], []
    for g, v in g_v_pairs:
        if g is None:
            continue
        tvars.append(v)
        grads.append(g)
    clipped_grads, _norm = tf.clip_by_global_norm(grads, clip_gradients)
    train_op = optim.apply_gradients(zip(clipped_grads, tvars))
    return train_op
