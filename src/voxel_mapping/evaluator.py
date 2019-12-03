import json
import torch
import numpy as np
from typing import List


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


class RecallPerClass(object):
    def __init__(self, number_of_classes: int, organ2ind_path: str):
        self.number_of_classes = number_of_classes
        self.totals = np.zeros(number_of_classes)
        self.corrects = np.zeros(number_of_classes)
        self.accuracies = np.zeros(number_of_classes)
        self.organ2ind = json.load(open(organ2ind_path))

    @staticmethod
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

    def update(self, true_indices: List[int], pred: np.ndarray, bboxes: np.ndarray):

        bbox_corrects = bbox_inside(pred, bboxes)

        for bbox_correct, true_index in zip(bbox_corrects, true_indices):
            self.totals[true_index] += 1
            if bbox_correct:
                self.corrects[true_index] += 1

    def calculate_recalls(self):
        self.accuracies = self.corrects / self.totals

    def reset(self):
        self.totals = np.zeros(self.number_of_classes)
        self.corrects = np.zeros(self.number_of_classes)
        self.accuracies = np.zeros(self.number_of_classes)

    def print_recalls(self):
        for organ, index in self.organ2ind.items():
            print(f'Organ: {organ}, Recall {100 * self.accuracies[index]}%')


