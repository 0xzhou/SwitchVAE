import numpy as np
import sklearn.metrics
import time
def evaluate_voxel_prediction(predictions, gt, threshold=1):
    """
    Calculate metrics based on the output of model
    Args:
        predictions: the ouput of voxel decoder
        gt: the ground truth of object
    Returns:
    """
    predtions_occupy = predictions >= threshold
    gt = gt >=1
    inverse_gt = gt < 1

    intersection = np.sum(np.logical_and(predtions_occupy, gt))
    union = np.sum(np.logical_or(predtions_occupy, gt))
    num_fp = np.sum(np.logical_and(predtions_occupy, inverse_gt))  # false positive
    num_fn = np.sum(np.logical_and(np.logical_not(predtions_occupy), gt))  # false negative

    precision = intersection/(intersection+num_fp)
    IoU = intersection / union
    recall = intersection/(intersection+num_fn)

    return precision, IoU, recall
