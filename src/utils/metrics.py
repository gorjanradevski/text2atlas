import torch
import numpy as np
from typing import List, Union


def bbox_volume(bbox: np.ndarray):
    """
        Return
        :param bbox: bounding box (3, 2)
        :return: Volume of bounding box
        """
    return (bbox[:, 1] - bbox[:, 0]).prod()


def bbox_intersection_volume(bboxes: List[np.ndarray]):
    mins = np.empty((3, 0))
    maxs = np.empty((3, 0))

    for bbox in bboxes:
        mins = np.concatenate((mins, bbox[:, 0][..., None]), axis=1)
        maxs = np.concatenate((maxs, bbox[:, 1][..., None]), axis=1)

    intersection = np.concatenate(
        (mins.max(axis=1)[..., None], maxs.min(axis=1)[..., None]), axis=1
    )

    return max((float(bbox_volume(intersection))), 0.0)


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
    return corrects.astype(float)


def bbox_shrink(
    bboxes: Union[List[List], List[np.ndarray], np.ndarray], bbox_shrink: float = 0.0
) -> np.ndarray:
    bboxes = np.array(bboxes)
    for bbox in bboxes:
        dims = bbox[:, 1] - bbox[:, 0]
        bbox[:, 0] += (bbox_shrink / 2) * dims
        bbox[:, 1] -= (bbox_shrink / 2) * dims
    return bboxes


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
