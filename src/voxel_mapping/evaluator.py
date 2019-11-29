import torch
import numpy as np


def bbox_inside(pred: np.ndarray, bboxes: np.ndarray):
    """
    Return
    :param pred: prediction for one sample, shape (3,)
    :param bboxes: set of bounding boxes of k organs inside the sample sentence, shape (k, 3, 2)
    :return: Array with k entries
     1.0 at i-th entry - pred is inside the i-th bounding box,
     0.0 at i-th entry - prediction is outside of the i-th bounding box
    """
    corrects = (
        np.concatenate((pred - bboxes[:, :, 0], bboxes[:, :, 1] - pred), axis=-1) >= 0
    ).all(axis=-1)

    return 1 if np.count_nonzero(corrects) > 0 else 0


def bbox_distance(
    pred: torch.Tensor, bboxes: torch.Tensor, norm_p: int = 2
) -> torch.Tensor:
    """
        Return
        :param pred: prediction for one sample, shape (3,)
        :param bboxes: set of bounding boxes of k organs inside the sample sentence, shape (k, 3, 2)
        :param norm_p: norm order - 1 is for Manhattan distance, 2 is for Euclidean
        :return: Array with k entries, i-th entry is distance from the point to the i-th bounding box
        """
    centers = bboxes.mean(axis=-1)
    dists = (
        (torch.abs(pred - centers) - (bboxes[:, :, 1] - bboxes[:, :, 0]) / 2)
        .clamp(min=0)
        .norm(p=norm_p, dim=-1)
    )
    return dists
