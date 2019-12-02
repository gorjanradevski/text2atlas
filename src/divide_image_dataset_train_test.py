import json
import math
import os
from tqdm import tqdm
import argparse


def create_train_test_datasets(
    directory_path: str,
    datasets_path: str,
    organ2center_path: str,
    organ2bbox_path: str,
    train_percentage: float = 0.9,
):
    organ2center = json.load(open(organ2center_path))
    organ2bbox = json.load(open(organ2bbox_path))
    train_dataset = []
    test_dataset = []
    for directory in tqdm(os.listdir(directory_path)):
        dir_path = os.path.join(directory_path, directory)
        if not os.path.isdir(dir_path):
            print(f"{dir_path} is not a directory")
            continue
        frames_dir_path = os.path.join(dir_path, "frames")
        image_names = os.listdir(frames_dir_path)
        train_size = math.floor(len(image_names) * train_percentage)
        train_names = image_names[:train_size]
        test_names = image_names[train_size:]
        for image_name in train_names:
            image_path = os.path.join(os.path.join(directory, "frames"), image_name)
            organ_name = " ".join(directory.split("_")[:-1])
            center = organ2center[organ_name]
            bbox = organ2bbox[organ_name]
            train_dataset.append(
                {"image_path": image_path, "centers": [center], "bboxes": [bbox]}
            )
        for image_name in test_names:
            image_path = os.path.join(os.path.join(directory, "frames"), image_name)
            organ_name = " ".join(directory.split("_")[:-1])
            center = organ2center[organ_name]
            bbox = organ2bbox[organ_name]
            test_dataset.append(
                {"image_path": image_path, "centers": [center], "bboxes": [bbox]}
            )
    json.dump(
        train_dataset,
        open(os.path.join(datasets_path, "train_image_dataset.json"), "w"),
    )
    json.dump(
        test_dataset, open(os.path.join(datasets_path, "test_image_dataset.json"), "w")
    )


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    create_train_test_datasets(
        args.directory_path,
        args.datasets_path,
        args.organ2center_path,
        args.organ2bbox_path,
        args.train_percentage,
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Divides the images dataset in train and test."
    )
    parser.add_argument(
        "--directory_path",
        type=str,
        default=".",
        help="Path to the directory with the organ files.",
    )
    parser.add_argument(
        "--datasets_path", type=str, default="data/", help="Where to dump the datasets."
    )
    parser.add_argument(
        "--organ2center_path",
        type=str,
        default="data/data_organs/organ2center.json",
        help="Path to the organ2center json file.",
    )
    parser.add_argument(
        "--organ2bbox_path",
        type=str,
        default="data/data_organs/organ2bbox.json",
        help="Path to the organ2bbox json file.",
    )
    parser.add_argument(
        "--train_percentage",
        type=float,
        default=0.9,
        help="The percentage of images to go to the train set.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
