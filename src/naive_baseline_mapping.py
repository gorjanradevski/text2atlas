import os
import numpy as np
import argparse
from tqdm import tqdm
import json
from typing import List

from voxel_mapping.evaluator import InferenceEvaluator


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


def center_naive(organs_dir_path: str, train_samples, test_samples, mode: str):
    # Prepare paths
    organ2mass_path = os.path.join(organs_dir_path, "organ2mass_new.json")
    organ2ind_path = os.path.join(organs_dir_path, "organ2ind_new.json")
    ind2organ_path = os.path.join(organs_dir_path, "ind2organ_new.json")
    organ2label_path = os.path.join(organs_dir_path, "organ2label_new.json")
    organ2summary_path = os.path.join(organs_dir_path, "organ2voxels_new.json")
    voxelman_images_path = os.path.join(organs_dir_path, "voxelman_images")
    # Load necessary jsons
    organ2mass = json.load(open(organ2mass_path))
    organ2ind = json.load(open(organ2ind_path))
    ind2organ = json.load(open(ind2organ_path))
    # Create evaluator
    evaluator = InferenceEvaluator(
        ind2organ_path,
        organ2label_path,
        voxelman_images_path,
        organ2summary_path,
        len(test_samples),
    )
    # Compute frequences
    frequencies = np.zeros(len(organ2ind))
    for sample in tqdm(train_samples):
        indices = sample["organ_indices"]
        for index in indices:
            frequencies[index] += 1
    most_frequent_ind = frequencies.argmax()
    prediction = organ2mass[ind2organ[str(most_frequent_ind)]]

    for sample in tqdm(test_samples):
        evaluator.update_counters(
            np.array(prediction), np.array(sample["organ_indices"])
        )

    print(
        f"At {mode} mode the average probability of hitting: {evaluator.get_current_accuracy()}"
    )
    print(
        f"At {mode} mode the average distance to hit: {evaluator.get_current_distance()}"
    )

    print(
        f"At {mode} mode the accuracy error-bar of hitting: {evaluator.get_accuracy_error_bar()}"
    )
    print(
        f"At {mode} mode the distance error-bar to hit: {evaluator.get_distance_error_bar()}"
    )


def random_naive(organs_dir_path: str, samples, mode, bbox_shrink=0.0):
    if mode == "random_easy":
        organs_bbox = json.load(
            open(os.path.join(organs_dir_path, "organ2bbox_new.json"))
        )
        bboxes = np.concatenate(
            [np.array(box) for organ, box in organs_bbox.items()], axis=1
        )
        volume_box = np.concatenate(
            (
                bboxes[:, 0::2].min(axis=1)[..., None],
                bboxes[:, 1::2].max(axis=1)[..., None],
            ),
            axis=1,
        )
    else:
        body_hull = json.load((os.path.join(organs_dir_path, "body_hull.json")))
        body_hull = np.array(body_hull["body_hull"])
        volume_box = np.concatenate(
            (body_hull.min(axis=0)[..., None], body_hull.max(axis=0)[..., None]), axis=1
        )

    volume = bbox_volume(volume_box)
    hit_probs = []

    for sample in tqdm(samples):
        # bboxes = sample["bboxes"]
        bboxes = sample["bboxes"]
        bboxes = np.array(bboxes)
        for bbox in bboxes:
            dims = bbox[:, 1] - bbox[:, 0]
            bbox[:, 0] += (bbox_shrink / 2) * dims
            bbox[:, 1] -= (bbox_shrink / 2) * dims
        """
        CALCULATING AREA OF UNION OF CUBOIDS
        FORMULA (V - volume, U - union)
        sum(V(singles)) - sum(V(overlapping pairs)) + sum(V(overlapping triplets)) - sum(V(overlapping quads)) ...
        We go to triplets, which should be approximately correct
        """
        union_volume = 0.0
        bboxes_seen = []
        for bbox in bboxes:
            bbox = np.array(bbox)
            union_volume += bbox_volume(bbox)
            for i, bbox_i in enumerate(bboxes_seen):
                union_volume -= bbox_intersection_volume([bbox, bbox_i])
                for j, bbox_j in enumerate(bboxes_seen[:i] + bboxes_seen[i + 1 :]):
                    union_volume += bbox_intersection_volume([bbox, bbox_i, bbox_j])
            bboxes_seen.append(bbox)

        hit_probs.append(union_volume / volume)

    print(
        f"At blind guessing in {mode} mode, the average probability of hitting is: {00 * sum(hit_probs) / len(hit_probs)}"
    )


def main():
    args = parse_args()
    assert args.mode in [
        "center",
        "random_easy",
        "random_hard",
    ], 'Parameter "difficulty" needs to be one of {"center", "random_easy", "random_hard"}'

    if args.mode in ["random_easy", "random_hard"]:
        test_samples = json.load(open(args.test_samples_path))
        random_naive(args.organs_dir_path, test_samples, args.mode, args.bbox_shrink)
    else:
        train_samples = json.load(open(args.train_samples_path))
        test_samples = json.load(open(args.test_samples_path))
        center_naive(args.organs_dir_path, train_samples, test_samples, args.mode)


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Performs naive baseline evaluation.")
    parser.add_argument(
        "--organs-dir-path",
        type=str,
        default="data/data_organs_new",
        help="Path to the data organs directory.",
    )
    parser.add_argument(
        "--train-samples-path",
        "-trp",
        type=str,
        default="",
        help="Path to the json with data samples",
    )
    parser.add_argument(
        "--test-samples-path",
        "-tsp",
        type=str,
        help="Path to the json with data samples",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        help='Mode - one of {"center", "random_easy", "random_hard"}, setting for naive evaluation',
    )
    parser.add_argument(
        "--bbox-shrink",
        "-bs",
        type=float,
        default=0.0,
        help="Fraction (from 0.0 to 1.0) by which we decrease each dimension of the bounding boxes",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
