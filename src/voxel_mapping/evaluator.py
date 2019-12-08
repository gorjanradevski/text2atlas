import json
import os
from typing import List, Union

import natsort
import numpy as np
import tifffile
import torch

from utils.constants import VOXELMAN_CENTER


class Evaluator:
    def __init__(
        self, ind2organ_path: str, organ2label_path: str, voxelman_images_path: str
    ):
        self.organ2label = json.load(open(organ2label_path))
        self.ind2organ = json.load(open(ind2organ_path))
        image_files = natsort.natsorted(
            [
                os.path.join(voxelman_images_path, f)
                for f in os.listdir(voxelman_images_path)
                if os.path.isfile(os.path.join(voxelman_images_path, f))
                and f.endswith(".tif")
            ]
        )[::-1]
        self.voxelman = tifffile.imread(image_files).transpose(1, 2, 0)
        self.total = 0
        self.correct = 0

    def reset_counters(self):
        self.total = 0
        self.correct = 0

    def update_counters(self, output_mapping: np.ndarray, organ_indices: np.ndarray):
        self.total += 1
        self.correct += self.voxels_inside(output_mapping, organ_indices)

    def get_current_accuracy(self):
        return self.correct / self.total

    def voxels_inside(
        self, pred: np.ndarray, organ_indices: Union[List, np.ndarray]
    ) -> int:
        """
        :param pred: prediction for one sample, numpy array or shape (3,)
        :param organ_indices: set of true organ indices for the sample, list or numpy array
        :return: Array with k entries
         1.0 at i-th entry - pred is inside the voxels of the i-th organ,
         0.0 at i-th entry - pred is outside of the voxels of the i-th organ
        """
        organ_indices = np.array(organ_indices)
        corrects = np.zeros(organ_indices.size)
        pred = np.round(pred + VOXELMAN_CENTER)
        pred = np.clip(
            pred, a_min=np.array([0, 0, 0]), a_max=(np.array(VOXELMAN_CENTER) * 2 - 1)
        )
        for i, organ_index in enumerate(organ_indices):
            labels = self.organ2label[self.ind2organ[str(organ_index)]]
            x, y, z = pred.astype(int)
            corrects[i] = int(self.voxelman[x, y, z] in labels)

        return 1 if np.count_nonzero(corrects) > 0 else 0

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
            np.concatenate((pred - bboxes[:, :, 0], bboxes[:, :, 1] - pred), axis=-1)
            >= 0
        ).all(axis=-1)

        return 1 if np.count_nonzero(corrects) > 0 else 0


class InferenceEvaluator(Evaluator):
    def __init__(
        self,
        ind2organ_path: str,
        organ2label_path: str,
        voxelman_images_path: str,
        organ2summary_path: str,
    ):
        super().__init__(ind2organ_path, organ2label_path, voxelman_images_path)
        self.organ2summary = json.load(open(organ2summary_path))
        self.distance = 0.0

    def reset_counters(self):
        super().reset_counters()
        self.distance = 0.0

    def update_counters(self, output_mapping: np.ndarray, organ_indices: np.ndarray):
        super().update_counters(output_mapping, organ_indices)
        self.distance += self.voxels_distance(output_mapping, organ_indices)

    def voxels_distance(
        self, pred: np.ndarray, organ_indices: Union[List, np.ndarray]
    ) -> np.ndarray:
        """
        :param pred: prediction for one sample, numpy array or shape (3,)
        :param organ_indices: set of true organ indices for the sample, list or numpy array
        :return: Array with k entries
         x at i-th entry - pred is x away from nearset voxel of the i-th organ,
         0.0 at i-th entry - pred is inside of the voxels of the i-th organ
        """
        organ_indices = np.array(organ_indices)
        distances = np.zeros(organ_indices.size)
        pred_ind = np.round(pred + VOXELMAN_CENTER)
        pred_ind = np.clip(
            pred_ind,
            a_min=np.array([0, 0, 0]),
            a_max=(np.array(VOXELMAN_CENTER) * 2 - 1),
        )
        for i, organ_index in enumerate(organ_indices):
            labels = self.organ2label[self.ind2organ[str(organ_index)]]
            x, y, z = pred_ind.astype(int)
            inside = int(self.voxelman[x, y, z] in labels)
            if inside:
                distances[i] = 0.0
            else:
                summary_points = np.array(
                    self.organ2summary[self.ind2organ[str(organ_index)]]
                )
                distances[i] = np.sqrt(
                    np.square(pred - summary_points).sum(axis=1)
                ).min()

        return distances.min()

    def get_current_distance(self):
        return self.distance / self.total


class TrainingEvaluator(Evaluator):
    def __init__(
        self, ind2organ_path: str, organ2label_path: str, voxelman_images_path: str
    ):
        super().__init__(ind2organ_path, organ2label_path, voxelman_images_path)
        self.best_avg_accuracy = -1
        self.current_average_accuracy = 0

    def reset_current_average_accuracy(self):
        self.current_average_accuracy = 0

    def reset_counters(self):
        super().reset_counters()

    def update_counters(self, output_mapping: np.ndarray, organ_indices: np.ndarray):
        super().update_counters(output_mapping, organ_indices)

    def update_current_average_accuracy(self):
        self.current_average_accuracy += self.correct / self.total

    def finalize_current_average_accuracy(self):
        self.current_average_accuracy /= 2

    def is_best_avg_accuracy(self):
        return self.current_average_accuracy > self.best_avg_accuracy

    def update_best_avg_accuracy(self):
        self.best_avg_accuracy = self.current_average_accuracy


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
