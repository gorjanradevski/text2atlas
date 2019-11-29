import os
import numpy as np
import argparse
from tqdm import tqdm

from utils.loadsave import load_json
from utils.constants import organs_dir
from utils.metrics import bbox_volume, bbox_intersection_volume


def main():
    args = parse_args()

    json_path = args.json_path
    difficulty = args.difficulty
    assert difficulty in ["easy", "hard"], "Parameter \"difficulty\" needs to be one of {\"easy\", \"hard\"}"

    samples = load_json(json_path)

    if difficulty == "easy":
        organs_bbox = load_json(os.path.join(organs_dir, "organ2bbox.json"))
        bboxes = np.concatenate([np.array(box) for organ, box in organs_bbox.items()], axis=1)
        volume_box = np.concatenate((bboxes[:, 0::2].min(axis=1)[..., None],
                                     bboxes[:, 1::2].max(axis=1)[..., None]),
                                    axis=1)
    else:
        body_hull = load_json(os.path.join(organs_dir, "body_hull.json"))
        body_hull = np.array(body_hull["body_hull"])
        volume_box = np.concatenate((body_hull.min(axis=0)[..., None],
                                     body_hull.max(axis=0)[..., None]),
                                    axis=1)

    volume = bbox_volume(volume_box)
    hit_probs = []

    for sample in tqdm(samples):
        bboxes = sample["bboxes"]
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
                for j, bbox_j in enumerate(bboxes_seen[:i]+bboxes_seen[i+1:]):
                    union_volume += bbox_intersection_volume([bbox, bbox_i, bbox_j])
            bboxes_seen.append(bbox)

        hit_probs.append(union_volume/volume)

    print('At {} difficulty, average probability of blind guessing: {}%'.format(
        difficulty, 100 * sum(hit_probs)/len(hit_probs))
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Trains an image-text matching model.")
    parser.add_argument(
        "--json-path",
        "-jp",
        type=str,
        help="Path to the json with data samples",
    )
    parser.add_argument(
        "--difficulty",
        "-d",
        type=str,
        help="Difficulty - one of {\"easy\", \"hard\"}, whether to run easier or harder setting for naive evaluation"
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()


