import torch
import numpy as np
from typing import List, Union


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
