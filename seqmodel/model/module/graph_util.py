""" A collections of useful functions to create common graphs """
import tensorflow as tf

from seqmodel.bunch import Bunch


def default_logit_opt():
    """
    Create a default options for logit. Technically, logit is a fully
    connected layer.
    """
    return Bunch(out_vocab_size=15, use_bias=True, name_prefix='logit',
                 trainable=True)


def create_logit_layer(opt, inputs, *args, **kwargs):
    """
    Args:
        opt: Option to create logit layer
        inputs: A tensor
        kwargs:
            logit_w: A tensor of logit weight to use [output_dim, input_dim]
    """
    if 'logit_w' in kwargs:
        logit_w = kwargs['logit_w']
    else:
        input_dim = int(inputs.get_shape()[-1])
        logit_w = tf.get_variable('{}_w'.format(opt.name_prefix),
                                  [opt.out_vocab_size, input_dim],
                                  dtype=tf.float32, trainable=opt.trainable)
    logit = matmul(inputs, logit_w, transpose_b=True)
    if opt.use_bias:
        logit_b = tf.get_variable('{}_b'.format(opt.name_prefix),
                                  [opt.out_vocab_size],
                                  dtype=tf.float32)
        logit = logit + logit_b
    temperature = tf.placeholder_with_default(
        1.0, shape=None, name="{}_temperature".format(opt.name_prefix))
    logit = logit / temperature
    return logit, temperature


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
