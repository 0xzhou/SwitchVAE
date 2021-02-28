import numpy as np
import tensorflow as tf


def evaluate_voxel_prediction(predictions, gt, threshold=1):
    """
    Calculate metrics based on the output of model
    Args:
        predictions: the ouput of voxel decoder
        gt: the ground truth of object
    Returns:
    """
    predtions_occupy = predictions >= threshold
    gt = gt >= 1
    inverse_gt = gt < 1

    intersection = np.sum(np.logical_and(predtions_occupy, gt))
    union = np.sum(np.logical_or(predtions_occupy, gt))
    num_fp = np.sum(np.logical_and(predtions_occupy, inverse_gt))  # false positive
    num_fn = np.sum(np.logical_and(np.logical_not(predtions_occupy), gt))  # false negative

    precision = intersection / (intersection + num_fp)
    IoU = intersection / union
    recall = intersection / (intersection + num_fn)

    return precision, IoU, recall


def get_precision(y_true, y_pred):
    """
    Calculate metrics in the training process
    """
    ones = tf.ones_like(y_true)
    zero = tf.zeros_like(y_true)

    y_pred = tf.where(y_pred > 0, ones, zero)
    inverse_y_ture = tf.where(y_true > 0, zero, ones)

    y_pred = tf.cast(y_pred, dtype=tf.bool)
    y_true = tf.cast(y_true, dtype=tf.bool)
    inverse_y_ture = tf.cast(inverse_y_ture, dtype=tf.bool)

    intersection = tf.reduce_sum(tf.cast(tf.math.logical_and(y_pred, y_true), dtype=tf.float32))
    num_fp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_pred, inverse_y_ture), dtype=tf.float32))
    precision = intersection / (intersection + num_fp)

    return precision


def get_IoU(y_true, y_pred):
    """
    Calculate metrics in the training process
    """
    ones = tf.ones_like(y_true)
    zero = tf.zeros_like(y_true)

    y_pred = tf.where(y_pred > 0, ones, zero)

    y_pred = tf.cast(y_pred, dtype=tf.bool)
    y_true = tf.cast(y_true, dtype=tf.bool)

    union = tf.reduce_sum(tf.cast(tf.math.logical_or(y_pred, y_true), dtype=tf.float32))
    intersection = tf.reduce_sum(tf.cast(tf.math.logical_and(y_pred, y_true), dtype=tf.float32))

    IoU = intersection / union

    return IoU
