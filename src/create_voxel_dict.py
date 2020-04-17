import json
import os
import random
from typing import List, Dict

import natsort
import numpy as np
import tifffile
from scipy.ndimage import generate_binary_structure, binary_erosion
from tqdm import tqdm


def get_images(images_dir, extension=".tif") -> List:
    """Return file names of image files inside a folder.

    Args:
        folder: str - path to folder
        extension: str - acceptable extension of files
    """
    return natsort.natsorted(
        [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if os.path.isfile(os.path.join(images_dir, f)) and f.endswith(extension)
        ]
    )[::-1]


def read_images(images_dir, extension=".tif") -> np.ndarray:
    """Return a 3D numpy array of stacked images in folder

    Args:
        folder: str - path to folder
        extension: str - acceptable extension of files
    """
    image_files = get_images(images_dir, extension)
    images = tifffile.imread(image_files)
    images = images.transpose(1, 2, 0)
    return images


def return_voxels(
    images: np.ndarray, labels: List[int], centering: bool = True
) -> List:
    if centering:
        center = np.array(images.shape).reshape(1, -1) / 2
    else:
        center = np.zeros(shape=(1, 3))

    voxels = np.zeros(shape=(0, 3))

    for label in labels:
        if type(label) is list:
            indices = np.logical_or.reduce([images == item for item in label])
        else:
            indices = images == label
        x, y, z = np.where(indices)
        points = np.vstack((x, y, z)).T - center
        voxels = np.concatenate((voxels, points), axis=0)
    return voxels.tolist()


def return_voxels_eroded(
    images: np.ndarray,
    labels: List[int],
    mask_size=3,
    mask_connectivity=1,
    centering: bool = True,
) -> List:
    if centering:
        center = np.array(images.shape).reshape(1, -1) / 2
    else:
        center = np.zeros(shape=(1, 3))

    indices = np.logical_or.reduce([images == label for label in labels])
    images[indices] = 1
    images[~indices] = 0
    erosion_mask = generate_binary_structure(mask_size, mask_connectivity)
    images = binary_erosion(images, erosion_mask).astype(int)
    x, y, z = np.where(images == 1)
    voxels_eroded = np.array((x, y, z)).T - center
    return voxels_eroded.tolist()


def generate_organ2voxels(images_path: str, organ2label: Dict):
    images = read_images(images_path)
    organ2voxels = {}
    for organ, labels in tqdm(organ2label.items()):
        organ2voxels[organ] = return_voxels(images.copy(), labels)
    return organ2voxels


def generate_organ2voxels_eroded(images_path: str, organ2label: Dict):
    images = read_images(images_path)
    organ2voxels = {}
    for organ, labels in tqdm(organ2label.items()):
        organ2voxels[organ] = return_voxels_eroded(images.copy(), labels)
    return organ2voxels


def generate_organ2summary(organ2voxels: Dict):
    organ2summary = {}
    for organ, voxels in tqdm(organ2voxels.items()):
        if len(voxels) > 1000:
            organ2summary[organ] = random.sample(voxels, 1000)
        else:
            organ2summary[organ] = np.array(voxels)[
                np.random.choice(range(len(organ2voxels[organ])), 1000)
            ].tolist()
    return organ2summary


import argparse


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Trains atlas reg mapping model.")
    parser.add_argument(
        "--organs_dir_path",
        type=str,
        default="data/data_organs_cord",
        help="Path to the data organs directory path.",
    )
    parser.add_argument(
        "--voxelman_images_path",
        type=str,
        default="data/labels",
        help="Path to the voxel-man images",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    organs_dir_path = args.organs_dir_path
    voxelman_images_path = args.voxelman_images_path

    organ2label = json.load(open(os.path.join(organs_dir_path, "organ2label.json")))

    organ2voxels = generate_organ2voxels(voxelman_images_path, organ2label)
    with open(os.path.join(organs_dir_path, "organ2voxels.json"), "w") as f:
        json.dump(organ2voxels, f)

    organ2summary = generate_organ2summary(organ2voxels)
    with open(os.path.join(organs_dir_path, "organ2summary.json"), "w") as f:
        json.dump(organ2summary, f)

    organ2voxels_eroded = generate_organ2voxels_eroded(
        voxelman_images_path, organ2label
    )
    with open(os.path.join(organs_dir_path, "organ2voxels_eroded.json"), "w") as f:
        json.dump(organ2voxels_eroded, f)
