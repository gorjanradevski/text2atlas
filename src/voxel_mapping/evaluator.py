import os
from typing import Dict, List, Union

import natsort
import numpy as np
import tifffile
from utils.constants import VOXELMAN_CENTER


class Evaluator:
    def __init__(
        self,
        ind2organ: Dict[int, str],
        organ2label: Dict,
        organ2sum_vox: Dict,
        voxelman_images_path: str,
        total_samples: int,
    ):
        self.total_samples = total_samples
        self.organ2label = organ2label
        self.ind2organ = ind2organ
        self.organ2sum_vox = organ2sum_vox
        image_files = natsort.natsorted(
            [
                os.path.join(voxelman_images_path, f)
                for f in os.listdir(voxelman_images_path)
                if os.path.isfile(os.path.join(voxelman_images_path, f))
                and f.endswith(".tif")
            ]
        )[::-1]
        self.voxelman = tifffile.imread(image_files).transpose(1, 2, 0)
        self.corrects = np.zeros(self.total_samples)
        self.distances = np.zeros(self.total_samples)
        self.index = 0

    def reset_counters(self):
        self.index = 0
        self.corrects = np.zeros(self.total_samples)
        self.distances = np.zeros(self.total_samples)

    def update_counters(self, output_mapping: np.ndarray, organ_indices: np.ndarray):
        self.distances[self.index] = self.voxels_distance(output_mapping, organ_indices)
        self.corrects[self.index] = 1 if self.distances[self.index] < 10.0 else 0
        self.index += 1

    def get_current_ior(self):
        return np.round((np.sum(self.corrects) / self.total_samples) * 100, decimals=1)

    def get_current_distance(self):
        return np.round((np.sum(self.distances) / self.total_samples) / 10, decimals=1)

    def get_current_miss_distance(self):
        return np.round(
            (np.sum(self.distances) / (np.count_nonzero(self.distances) + 1e-15)) / 10,
            decimals=1,
        )

    def voxels_distance(
        self, pred: np.ndarray, organ_indices: Union[List, np.ndarray]
    ) -> np.ndarray:
        # Setting distances to large number so that we can take minimum
        distances = np.ones(organ_indices.size, dtype=np.float) * 1e15
        pred_ind = np.round(pred + VOXELMAN_CENTER)
        pred_ind = np.clip(pred_ind, a_min=[0, 0, 0], a_max=(VOXELMAN_CENTER * 2) - 1)
        x, y, z = pred_ind.astype(int)
        for i, organ_index in enumerate(organ_indices):
            if organ_index < 0:
                continue
            str_organ_index = str(organ_index)
            if (
                self.voxelman[x, y, z]
                in self.organ2label[self.ind2organ[str_organ_index]]
            ):
                # If it is inside, the minimum will always be 0
                return 0.0
            else:
                summary_points = np.array(
                    self.organ2sum_vox[self.ind2organ[str_organ_index]]
                )
                distances[i] = np.linalg.norm(pred - summary_points, axis=-1).min()

        return distances.min()


class InferenceEvaluator(Evaluator):
    def __init__(
        self,
        ind2organ: Dict,
        organ2label: Dict,
        organ2sum_vox: Dict,
        voxelman_images_path: str,
        total_samples: int,
    ):
        super().__init__(
            ind2organ, organ2label, organ2sum_vox, voxelman_images_path, total_samples,
        )

    def get_ior_error_bar(self):
        return np.round(
            np.std(self.corrects, ddof=1) / np.sqrt(self.total_samples) * 100,
            decimals=1,
        )

    def get_distance_error_bar(self):
        return np.round(
            np.std(self.distances, ddof=1) / np.sqrt(self.total_samples) / 10,
            decimals=1,
        )

    def get_miss_distance_error_bar(self):
        return np.round(
            np.std(self.distances[np.nonzero(self.distances)], ddof=1)
            / np.sqrt(np.count_nonzero(self.distances))
            / 10,
            decimals=1,
        )


class TrainingEvaluator(Evaluator):
    def __init__(
        self,
        ind2organ: Dict,
        organ2label: Dict,
        organ2sum_vox: Dict,
        voxelman_images_path: str,
        total_samples: int,
        best_distance: float,
    ):
        super().__init__(
            ind2organ, organ2label, organ2sum_vox, voxelman_images_path, total_samples,
        )
        self.best_distance = best_distance
        self.current_distance = 0

    def reset_current_distance(self):
        self.current_distance = 0

    def update_current_distance(self):
        self.current_distance += self.get_current_distance()

    def is_best_distance(self):
        return self.current_distance <= self.best_distance

    def update_best_distance(self):
        self.best_distance = self.current_distance


class InferenceEvaluatorPerOrgan(InferenceEvaluator):
    def __init__(
        self,
        ind2organ: str,
        organ2label: str,
        organ2sum_vox: str,
        voxelman_images_path: str,
        organ2count: Dict[str, int],
        total_samples: int,
    ):
        super().__init__(
            ind2organ, organ2label, organ2sum_vox, voxelman_images_path, total_samples
        )
        self.organ2count = organ2count
        self.organ_totals = {organ_name: 0 for organ_name in self.organ2count.keys()}
        self.organ_corrects = {
            organ_name: np.zeros((organ_count,))
            for organ_name, organ_count in self.organ2count.items()
        }
        self.organ_distances = {
            organ_name: np.zeros((organ_count,))
            for organ_name, organ_count in self.organ2count.items()
        }
        self.index_organs = {organ_name: 0 for organ_name in self.organ2count.keys()}

    def reset_counters(self):
        self.organ_totals = {organ_name: 0 for organ_name in self.organ2count.keys()}
        self.organ_corrects = {
            organ_name: np.zeros((organ_count,))
            for organ_name, organ_count in self.organ2count.items()
        }
        self.organ_distances = {
            organ_name: np.zeros((organ_count,))
            for organ_name, organ_count in self.organ2count.items()
        }
        self.index_organs = {organ_name: 0 for organ_name in self.organ2count.keys()}
        super().reset_counters()

    def update_counters(self, output_mapping: np.ndarray, organ_indices: np.ndarray):
        for organ_index in organ_indices:
            if organ_index < 0:
                continue
            organ_name = self.ind2organ[str(organ_index)]
            self.organ_totals[organ_name] += 1
            sample_distance = self.voxels_distance(
                output_mapping, np.array([organ_index])
            )
            self.organ_distances[organ_name][
                self.index_organs[organ_name]
            ] += sample_distance
            self.organ_corrects[organ_name][self.index_organs[organ_name]] = (
                1 if sample_distance < 10.0 else 0
            )
            self.index_organs[organ_name] += 1
        super().update_counters(output_mapping, organ_indices)

    def get_current_ior_for_organ(self, organ: str):
        if self.organ_totals[organ]:
            return np.round(
                np.sum(self.organ_corrects[organ]) / self.organ_totals[organ] * 100,
                decimals=1,
            )
        else:
            return -1

    def get_current_distance_for_organ(self, organ: str):
        if self.organ_totals[organ]:
            return np.round(
                np.sum(self.organ_distances[organ]) / self.organ_totals[organ] / 10,
                decimals=1,
            )
        else:
            return -1

    def get_current_miss_distance_for_organ(self, organ):
        if self.organ_totals[organ]:
            return np.round(
                (
                    np.sum(self.organ_distances[organ])
                    / (np.count_nonzero(self.distances) + 1e-15)
                )
                / 10,
                decimals=1,
            )
        else:
            return -1

    def get_ior_error_bar_for_organ(self, organ):
        if self.organ_totals[organ]:
            return np.round(
                np.std(self.organ_corrects[organ], ddof=1)
                / np.sqrt(self.organ_totals[organ])
                * 100,
                decimals=1,
            )
        else:
            return -1

    def get_distance_error_bar_for_organ(self, organ):
        if self.organ_totals[organ]:
            return np.round(
                np.std(self.organ_distances[organ], ddof=1)
                / np.sqrt(self.organ_totals[organ])
                / 10,
                decimals=1,
            )
        else:
            return -1

    def get_miss_distance_error_bar_for_organ(self, organ):
        return np.round(
            np.std(
                self.organ_distances[organ][np.nonzero(self.organ_distances[organ])],
                ddof=1,
            )
            / np.sqrt(np.count_nonzero(self.organ_distances[organ]) + 1e-15)
            / 10,
            decimals=1,
        )
