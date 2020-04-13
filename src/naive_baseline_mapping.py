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


def frequency_naive(
    organs_dir_path: str,
    voxelman_images_path: str,
    train_samples,
    test_samples,
    mode: str,
):
    # Prepare paths
    organ2mass_path = os.path.join(organs_dir_path, "organ2mass.json")
    organ2ind_path = os.path.join(organs_dir_path, "organ2ind.json")
    ind2organ_path = os.path.join(organs_dir_path, "ind2organ.json")
    organ2label_path = os.path.join(organs_dir_path, "organ2label.json")
    organ2summary_path = os.path.join(organs_dir_path, "organ2summary.json")
    # Load necessary jsons
    organ2mass = json.load(open(organ2mass_path))
    organ2ind = json.load(open(organ2ind_path))
    ind2organ = json.load(open(ind2organ_path))
    # Create evaluator
    evaluator = InferenceEvaluator(
        ind2organ_path,
        organ2label_path,
        organ2summary_path,
        voxelman_images_path,
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
        f"At {mode} mode the average probability of hitting: {evaluator.get_current_ior():.3f}%"
    )
    print(
        f"At {mode} mode the average distance to hit: {evaluator.get_current_distance():.3f}"
    )

    print(
        f"At {mode} mode the IOR error-bar of hitting: {evaluator.get_ior_error_bar()}"
    )
    print(
        f"At {mode} mode the distance error-bar to hit: {evaluator.get_distance_error_bar()}"
    )


def center_naive(
    organs_dir_path: str, voxelman_images_path: str, test_samples, mode: str
):
    # Prepare paths
    ind2organ_path = os.path.join(organs_dir_path, "ind2organ.json")
    organ2label_path = os.path.join(organs_dir_path, "organ2label.json")
    organ2summary_path = os.path.join(organs_dir_path, "organ2summary.json")
    # Create evaluator
    evaluator = InferenceEvaluator(
        ind2organ_path,
        organ2label_path,
        organ2summary_path,
        voxelman_images_path,
        len(test_samples),
    )

    prediction = [0.0, 0.0, 0.0]

    for sample in tqdm(test_samples):
        evaluator.update_counters(
            np.array(prediction), np.array(sample["organ_indices"])
        )

    print(
        f"At {mode} mode the average probability of hitting: {100 * evaluator.get_current_ior():.3f}%"
    )
    print(
        f"At {mode} mode the average distance to hit: {evaluator.get_current_distance():.3f}"
    )

    print(
        f"At {mode} mode the IOR error-bar of hitting: {evaluator.get_ior_error_bar()}"
    )
    print(
        f"At {mode} mode the distance error-bar to hit: {evaluator.get_distance_error_bar()}"
    )


def random_naive(organs_dir_path: str, samples, mode):
    organ2voxels = json.load(open(os.path.join(organs_dir_path, "organ2voxels.json")))
    ind2organ = json.load(open(os.path.join(organs_dir_path, "ind2organ.json")))
    body_hull = json.load(open(os.path.join(organs_dir_path, "body_hull.json")))
    body_hull = np.array(body_hull["body_hull"])
    volume_box = np.concatenate(
        (body_hull.min(axis=0)[..., None], body_hull.max(axis=0)[..., None]), axis=1
    )

    volume = bbox_volume(volume_box)
    hit_probs = []

    for sample in tqdm(samples):
        union_volume = 0.0
        organ_indices = sample["organ_indices"]
        for organ_index in organ_indices:
            union_volume += len(organ2voxels[ind2organ[str(organ_index)]])

        hit_probs.append(union_volume / volume)

    print(
        f"At blind guessing in {mode} mode, the average probability of hitting is: {100 * sum(hit_probs) / len(hit_probs):.3f}%"
    )
    print(f"Error bound: {np.std(np.array(hit_probs), ddof=1)/np.sqrt(len(hit_probs))}")


def main():
    args = parse_args()
    assert args.mode in [
        "frequency",
        "center",
        "random",
    ], 'Parameter "difficulty" needs to be one of {"frequency", "center", "random"}'

    if args.mode == "random":
        test_samples = json.load(open(args.test_samples_path))
        random_naive(args.organs_dir_path, test_samples, args.mode)
    elif args.mode == "center":
        test_samples = json.load(open(args.test_samples_path))
        center_naive(
            args.organs_dir_path, args.voxelman_images_path, test_samples, args.mode
        )
    else:
        train_samples = json.load(open(args.train_samples_path))
        test_samples = json.load(open(args.test_samples_path))
        frequency_naive(
            args.organs_dir_path,
            args.voxelman_images_path,
            train_samples,
            test_samples,
            args.mode,
        )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Performs naive baseline evaluation.")
    parser.add_argument(
        "--organs-dir-path",
        type=str,
        default="data/data_organs_sages",
        help="Path to the data organs directory.",
    )
    parser.add_argument(
        "--voxelman-images-path",
        type=str,
        default="data/voxelman_images",
        help="Path to the voxel-man images",
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
        help='Mode - one of {"frequency", "center", "random"}, setting for naive evaluation',
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
