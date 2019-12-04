import json
import os
from typing import List, Union

import natsort
import numpy as np
import tifffile
import torch

from utils.constants import VOXELMAN_CENTER


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
    def __init__(
        self,
        number_of_classes: int,
        ind2organ_path: str,
        organ2ind_path: str,
        organ2label_path: str,
        voxelman_images_path: str,
        method: str = "bboxes",
    ):
        self.number_of_classes = number_of_classes
        self.totals = np.zeros(number_of_classes)
        self.corrects = np.zeros(number_of_classes)
        self.accuracies = np.zeros(number_of_classes)
        self.organ2ind = json.load(open(organ2ind_path))
        self.ind2organ = json.load(open(ind2organ_path))
        self.organ2label = json.load(open(organ2label_path))

        image_files = natsort.natsorted(
            [
                os.path.join(voxelman_images_path, f)
                for f in os.listdir(voxelman_images_path)
                if os.path.isfile(os.path.join(voxelman_images_path, f))
                and f.endswith(".tif")
            ]
        )[::-1]
        self.voxelman = tifffile.imread(image_files)
        self.voxelman = self.voxelman.transpose(1, 2, 0)

        assert method in [
            "voxels",
            "bboxes",
        ], 'Parameter "method" needs to be one of {"voxels", "bboxes"}'
        self.method = method

    @staticmethod
    def bbox_inside(pred: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """
        :param pred: prediction for one sample, shape (3,)
        :param bboxes: set of bounding boxes of k organs inside the sample sentence, shape (k, 3, 2)
        :return: Array with k entries
         1.0 at i-th entry - pred is inside the i-th bounding box,
         0.0 at i-th entry - pred is outside of the i-th bounding box
        """
        corrects = (
            np.concatenate((pred - bboxes[:, :, 0], bboxes[:, :, 1] - pred), axis=-1)
            >= 0
        ).all(axis=-1)
        return corrects.astype(float)

    def voxels_inside(
        self, pred: np.ndarray, organ_indices: Union[List, np.ndarray]
    ) -> np.ndarray:
        """
        :param pred: prediction for one sample, numpy array or shape (3,)
        :param organ_indices: set of true organ indices for the sample, list or numpy array
        :return: Array with k entries
         1.0 at i-th entry - pred is inside the voxels of the i-th organ,
         0.0 at i-th entry - pred is outside of the voxels of the i-th organ
        """
        organ_indices = np.array(organ_indices)
        corrects = np.zeros(organ_indices.size)
        pred = np.round(pred * 2) / 2 + VOXELMAN_CENTER
        for i, organ_index in enumerate(organ_indices):
            labels = self.organ2label[self.ind2organ[str(organ_index)]]
            x, y, z = pred.astype(int)
            corrects[i] = int(self.voxelman[x, y, z] in labels)
        return corrects.astype(float)

    def update(self, true_indices: List[int], pred: np.ndarray, bboxes: np.ndarray):

        if self.method == "bboxes":
            corrects = self.bbox_inside(pred, bboxes)
        else:
            corrects = self.voxels_inside(pred, true_indices)

        for correct, true_index in zip(corrects, true_indices):
            self.totals[true_index] += 1
            if correct:
                self.corrects[true_index] += 1

    def calculate_recalls(self):
        self.accuracies = self.corrects / self.totals

    def reset(self):
        self.totals = np.zeros(self.number_of_classes)
        self.corrects = np.zeros(self.number_of_classes)
        self.accuracies = np.zeros(self.number_of_classes)

    def print_recalls(self):
        for organ, index in self.organ2ind.items():
            print(f"Organ: {organ}, Recall {100 * self.accuracies[index]}%")
