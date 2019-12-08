import os
import numpy as np
import argparse
from tqdm import tqdm

from utils.loadsave import load_json
from utils.constants import organs_dir
from utils.metrics import bbox_volume, bbox_intersection_volume, bbox_inside


def center_naive(train_samples, test_samples, mode, bbox_shrink=0.0):
    avg_center = np.zeros(3)
    n = 0
    for sample in tqdm(train_samples):
        centers = sample["centers"]
        for center in centers:
            avg_center += center
            n += 1
    avg_center /= n

    corrects = 0
    for sample in tqdm(test_samples):
        bboxes = sample["bboxes"]
        bboxes = np.array(bboxes)
        for bbox in bboxes:
            dims = bbox[:, 1] - bbox[:, 0]
            bbox[:, 0] += (bbox_shrink / 2) * dims
            bbox[:, 1] -= (bbox_shrink / 2) * dims

        if bbox_inside(avg_center, bboxes).any():
            corrects += 1

    print(
        "At {} mode, with bboxes shrunk by {}%, average probability of hitting : {}%".format(
            mode, 100 * bbox_shrink, 100 * corrects / len(test_samples)
        )
    )


def random_naive(samples, mode, bbox_shrink=0.0):
    if mode == "random_easy":
        organs_bbox = load_json(os.path.join(organs_dir, "organ2bbox.json"))
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
        body_hull = load_json(os.path.join(organs_dir, "body_hull.json"))
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
        "At blind guessing in {} mode, with bboxes shrunk by {}%, the average probability of hitting is: {}%".format(
            mode, 100 * bbox_shrink, 100 * sum(hit_probs) / len(hit_probs)
        )
    )


def main():
    args = parse_args()

    train_samples_path = args.train_samples_path
    test_samples_path = args.test_samples_path
    mode = args.mode
    bbox_shrink = args.bbox_shrink
    assert mode in [
        "center",
        "random_easy",
        "random_hard",
    ], 'Parameter "difficulty" needs to be one of {"center", "random_easy", "random_hard"}'

    if mode in ["random_easy", "random_hard"]:
        test_samples = load_json(test_samples_path)
        assert test_samples, "Invalid test data provided: {}".format(test_samples_path)
        random_naive(test_samples, mode, bbox_shrink)
    else:
        train_samples = load_json(train_samples_path)
        assert train_samples, "Invalid train data provided: {}".format(
            train_samples_path
        )
        test_samples = load_json(test_samples_path)
        assert test_samples, "Invalid test data provided: {}".format(test_samples_path)
        center_naive(train_samples, test_samples, mode, bbox_shrink)


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Trains an image-text matching model.")
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
