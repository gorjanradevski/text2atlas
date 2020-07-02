import argparse
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
import numpy as np
import random

from voxel_mapping.datasets import VoxelSentenceMappingTestRegDataset
from voxel_mapping.evaluator import InferenceEvaluatorPerOrgan


def naive_evaluation(
    naive_type: str,
    train_json_path: str,
    test_json_path: str,
    organs_dir_path: str,
    voxelman_images_path: str,
):
    print(f"Using {naive_type} baseline!")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_json_dataset = json.load(open(train_json_path))
    test_dataset = VoxelSentenceMappingTestRegDataset(test_json_path, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1)
    ind2organ = json.load(open(os.path.join(organs_dir_path, "ind2organ.json")))
    organ2label = json.load(open(os.path.join(organs_dir_path, "organ2label.json")))
    organ2voxels = json.load(open(os.path.join(organs_dir_path, "organ2summary.json")))
    # Create evaluator
    evaluator = InferenceEvaluatorPerOrgan(
        ind2organ, organ2label, organ2voxels, voxelman_images_path, len(test_dataset),
    )
    # Compute frequencies
    print("Aggregating from training set...")
    frequencies = np.zeros(len(ind2organ))
    for sample in tqdm(train_json_dataset):
        indices = sample["organ_indices"]
        for index in indices:
            frequencies[index] += 1
    most_freq_organ = ind2organ[str(frequencies.argmax())]
    print("Evaluating...")
    for _, organs_indices, _ in tqdm(test_loader):
        if naive_type == "center":
            pred = np.array([0.0, 0.0, 0.0])
        elif naive_type == "frequency":
            pred = np.array(random.sample(organ2voxels[most_freq_organ], 1))[0]
        elif naive_type == "random":
            random_organ = random.sample(list(organ2voxels.keys()), 1)[0]
            pred = np.array(random.sample(organ2voxels[random_organ], 1))[0]
        else:
            raise ValueError(f"{naive_type} doesn't exist!")
        evaluator.update_counters(pred, organs_indices[0].numpy())

    print(
        "The avg IOR on the test set is: "
        f"{evaluator.get_current_ior()} +/- {evaluator.get_ior_error_bar()}"
    )
    print(
        "The avg distance on the test set is: "
        f"{evaluator.get_current_distance()} +/- {evaluator.get_distance_error_bar()}"
    )
    print(
        "The avg miss distance on the test set is: "
        f"{evaluator.get_current_miss_distance()} +/- {evaluator.get_miss_distance_error_bar()}"
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Performs naive baseline evaluation.")
    parser.add_argument(
        "--naive_type", type=str, default="center", help="The baseline type.",
    )
    parser.add_argument(
        "--voxelman_images_path",
        type=str,
        default="data/voxelman_images",
        help="Path to the voxelman images",
    )
    parser.add_argument(
        "--train_json_path",
        type=str,
        default="data/mesh_dataset_train.json",
        help="Path to the test set",
    )
    parser.add_argument(
        "--test_json_path",
        type=str,
        default="data/mesh_dataset_test.json",
        help="Path to the test set",
    )
    parser.add_argument(
        "--organs_dir_path",
        type=str,
        default="data/data_organs_mesh",
        help="Path to the data organs directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    naive_evaluation(
        args.naive_type,
        args.train_json_path,
        args.test_json_path,
        args.organs_dir_path,
        args.voxelman_images_path,
    )


if __name__ == "__main__":
    main()
