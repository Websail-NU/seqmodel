""" A collections of useful functions to create common graphs """
import cPickle
import numpy as np
import tensorflow as tf

from seqmodel.bunch import Bunch


def default_logit_opt():
    """
    Create a default options for logit. Technically, logit is a fully
    connected layer.
    """
    return Bunch(out_vocab_size=15, use_bias=True, name_prefix='logit',
                 trainable=True)


def create_logit_layer(opt, inputs, logit_w=None, logit_b=None,
                       temperature=None, *args, **kwargs):
    """
    Args:
        opt: Option to create logit layer
        inputs: A tensor
        kwargs:
            logit_w: A tensor of logit weight to use [output_dim, input_dim]
    """
    if logit_w is None:
        input_dim = int(inputs.get_shape()[-1])
        logit_w = tf.get_variable('{}_w'.format(opt.name_prefix),
                                  [opt.out_vocab_size, input_dim],
                                  dtype=tf.float32, trainable=opt.trainable)
    logit = matmul(inputs, logit_w, transpose_b=True)
    if opt.use_bias:
        if logit_b is None:
            logit_b = tf.get_variable('{}_b'.format(opt.name_prefix),
                                      [opt.out_vocab_size],
                                      dtype=tf.float32)
        logit = logit + logit_b
    if temperature is None:
        temperature = tf.placeholder_with_default(
            1.0, shape=None, name="{}_temperature".format(opt.name_prefix))
    logit = logit / temperature
    return logit, temperature, logit_w, logit_b


def create_embedding_var(vocab_size, dim, trainable=True, name='embedding',
                         init_filepath=None):
    if init_filepath is None:
        return tf.get_variable(name, [vocab_size, dim], trainable=trainable)
    else:
        with open(init_filepath) as ifp:
            init_emb = cPickle.load(ifp)
        return tf.get_variable(
            name, trainable=trainable, initializer=tf.constant(
                init_emb, dtype=tf.float32))


def create_update_layer(transform, extra, carried):
    transform_dim = int(transform.get_shape()[-1])
    carried_dim = int(carried.get_shape()[-1])
    extra_dim = int(extra.get_shape()[-1])
    in_size = transform_dim + extra_dim
    out_size = carried_dim * 2
    gate_w = tf.get_variable("gate_w", [in_size, out_size])
    _arr = np.zeros((out_size))
    _arr[:] = -1
    gate_b = tf.get_variable("gate_b", initializer=tf.constant(
        _arr, dtype=tf.float32))
    z = matmul(tf.concat([transform, extra], -1), gate_w) + gate_b
    t = tf.sigmoid(tf.slice(z, [0, 0, 0], [-1, -1, carried_dim]))
    h = tf.tanh(tf.slice(z, [0, 0, carried_dim], [-1, -1, -1]))
    return tf.multiply(h - carried, t) + carried


# def create_update_layer(self, transform, extra, carried):
#         dim = len(transform.get_shape())
#         transform_dim = int(transform.get_shape()[-1])
#         carried_dim = int(carried.get_shape()[-1])
#         extra_dim = int(extra.get_shape()[-1])
#         in_size = transform_dim + extra_dim
#         out_size = carried_dim * 2
#         gate_w = tf.get_variable("gate_w", [in_size, out_size])
#         _arr = np.zeros((out_size))
#         _arr[:] = self._init_gate_bias
#         gate_b = tf.get_variable("gate_b", initializer=tf.constant(
#             _arr, dtype=tf.float32))
#         if dim == 3:
#             z = self.helper.fancy_matmul(
#                 tf.concat([transform, extra], -1), gate_w) + gate_b
#             t = tf.sigmoid(tf.slice(z, [0, 0, 0], [-1, -1, carried_dim]))
#             h = tf.tanh(tf.slice(z, [0, 0, carried_dim], [-1, -1, -1]))
#         else:
#             z = tf.matmul(tf.concat([transform, extra], -1), gate_w) + gate_b
#             t = tf.sigmoid(tf.slice(z, [0, 0], [-1, carried_dim]))
#             h = tf.tanh(tf.slice(z, [0, carried_dim], [-1, -1]))
#         self._transform_gate = t
#         o = tf.multiply(h - carried, t) + carried
#         self._final_rnn_output = o
#         if self._opt.keep_prob < 1.0 and self.is_training:
#             o = tf.nn.dropout(o, self._opt.keep_prob)
#         return o


def matmul(mat, mat2d, transpose_b=False):
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
