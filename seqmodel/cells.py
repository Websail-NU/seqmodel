import tensorflow as tf


def _tensor2gaussian(
        tensor, residual_mu=None, residual_logvar=None,
        activation=None, name='gaussian'):
    if activation is None:
        activation = tf.nn.tanh
    out_dim = int(tensor.get_shape()[-1])
    # g_hidden = tf.layers.dense(
    #     tensor, out_dim*2, activation=None, name=f'{name}_hidden')
    # g_params = tf.layers.dense(
    #     g_hidden, out_dim*2, activation=None, name=f'{name}_output')
    g_params = tf.layers.dense(
        tensor, out_dim*2, activation=None, name=f'{name}_output')
    mu = activation(tf.slice(g_params, [0, 0], [-1, out_dim]))
    if residual_mu is not None:
        mu += residual_mu
    logvar = tf.slice(g_params, [0, out_dim], [-1, -1])
    if residual_logvar is not None:
        logvar += residual_logvar
    logvar = tf.log(tf.exp(logvar) + 1e-8)
    return mu, logvar


def _sample_normal(mu, logvar):
    epsilon = tf.random_normal(tf.shape(logvar))
    std = tf.exp(0.5 * logvar)
    std = tf.Print(std, [tf.reduce_mean(std)])
    z = mu + tf.multiply(std, epsilon)
    return z


class SGRU(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units):
        self.num_units = num_units
        self.cell = tf.nn.rnn_cell.GRUCell(num_units)

    @property
    def output_size(self):
        return self.cell.output_size

    @property
    def state_size(self):
        return self.cell.state_size

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scope:
            output, new_state = self.cell(inputs, state, scope=scope)
            output = tf.contrib.layers.layer_norm(output)
            mu, logvar = _tensor2gaussian(output, name='gaussian')
            z = _sample_normal(mu, logvar)
            return z, z
