""" A collection of functions that define losses for training """
import tensorflow as tf


def xent_loss(logit, label, weight, seq_weight=None):
    """
    Compute negative likelihood (cross-entropy loss)
    Args:
        logit: A tensor of logits
        label: A tensor of labels (same shape as logit)
        weight: A tensor of weights
    Return:
        batch losses
        traning loss: NLL / loss_denom
        loss_denom: a placeholder for training loss denominator
        eval loss: average NLL
    """
    # Internally logits and labels are reshaped into 2D and 1D...
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logit, labels=label)
    sum_loss = tf.reduce_sum(tf.multiply(loss, weight))
    mean_loss = _safe_div(sum_loss, tf.reduce_sum(weight))
    if seq_weight is not None:
        sum_loss = tf.reduce_sum(tf.multiply(
            loss, tf.multiply(weight, seq_weight)))
    loss_denom = tf.placeholder_with_default(
        1.0, shape=None, name="training_loss_denom")
    training_loss = _safe_div(sum_loss, loss_denom)
    return loss, training_loss, loss_denom, mean_loss


def ent_loss(distribution, weight, seq_weight=None):
    """
    Compute entropy loss (negative entropy)
    Args:
        logit: A tensor of logits
        distribution: A tensor of distribution (same shape as logit)
        weight: A tensor of weights
    Return:
        mean entropy loss
    """
    if seq_weight is not None:
        weight = tf.multiply(weight, seq_weight)
    neg_entropy = tf.reduce_sum(
        distribution * tf.log(distribution + 1e-6), axis=-1)
    sum_loss = tf.reduce_sum(neg_entropy * weight)
    num_dist = tf.reduce_sum(weight)
    mean_loss = _safe_div(sum_loss, num_dist)
    return mean_loss


def slow_feature_loss(feature, weight, delta=1.0):
    """
    Compute a constrastive slow feature analysis loss
    Args:
        feature: A tensor of shape [batch, time, dim]
        weight: A tensor of shape [batch, time, time].
                For correctness, this tensor should only contain 0.0 and 1.0;
                and each [i, time, time] matrix should be an upper triangular
                matrix.
    Return:
        loss: A tensor of shape [batch, time, time]
        batch_loss: sum loss, averaged by batch size
    """
    D = _all_pair_sq_dist(feature)
    R2 = tf.multiply(D, weight)
    if delta > 0:
        n_weight = 1 - weight
        n_weight = tf.matrix_band_part(n_weight, 0, -1)
        n_weight = n_weight - tf.matrix_band_part(n_weight, 0, 0)
        R2_n = tf.multiply(tf.nn.relu(delta - D), n_weight)
        R2 = R2 + R2_n
    batch_size = tf.shape(feature)[0]
    return R2, tf.reduce_sum(R2) / tf.cast(batch_size, tf.float32)


def _all_pair_sq_dist(A):

    r = tf.expand_dims(tf.reduce_sum(A*A, -1), axis=-1)
    D = r - 2 * tf.matmul(A, tf.transpose(A, perm=[0, 2, 1]))\
        + tf.transpose(r, perm=[0, 2, 1])
    # for stablity reason, we will not put on sqrt
    # D = tf.sqrt(D - tf.matrix_band_part(D, 0, 0))
    return D


def l2_loss(var_list):
    l2_loss = tf.reduce_sum(tf.add_n(
        [tf.nn.l2_loss(var) for var in var_list]))
    return l2_loss


def _safe_div(numerator, denominator, name="value"):
    """
    Adapted from Tensorflow losses

    Computes a safe divide which returns 0 if the denominator is zero.
    Note that the function contains an additional conditional check that is
    necessary for avoiding situations where the loss is zero causing NaNs to
    creep into the gradient computation.
    Args:
    numerator: An arbitrary `Tensor`.
    denominator: `Tensor` whose shape matches `numerator`
    name: An optional name for the returned op.
    Returns:
    The element-wise value of the numerator divided by the denominator.
    """
    return tf.where(tf.equal(denominator, 0),
                    tf.zeros_like(numerator),
                    tf.div(numerator, denominator),
                    name=name)
