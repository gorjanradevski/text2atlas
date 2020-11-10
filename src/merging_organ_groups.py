import argparse
import json
import os
import random
from typing import List

import natsort
import numpy as np
import tifffile
from scipy.ndimage import binary_erosion, generate_binary_structure
from skimage.measure import label

from utils.constants import VOXELMAN_CENTER


def store_json(data, path: str):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except ValueError as e:
        print("Invalid path: %s" % e)
        return None


def get_images(folder, extension):
    """Return file names of image files inside a folder.

    Args:
        folder: str - path to folder
        extension: str - acceptable extension of files
    """
    return natsort.natsorted(
        [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and f.endswith(extension)
        ]
    )[::-1]


def read_images(folder, extension=".tif"):
    """Return a 3D numpy array of stacked images in folder

        Args:
            folder: str - path to folder
            extension: str - acceptable extension of files
        """
    image_files = get_images(folder, extension)
    images = tifffile.imread(image_files)
    images = images.transpose(1, 2, 0)
    return images


def getLargestCC(points):
    labels = label(points)
    assert labels.max() != 0, "No connected regions"
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return np.where(largestCC == True)  # noqa: E712


def get_center_of_mass(labels, images_path):

    images_in = read_images(images_path, extension=".tif")

    images = np.zeros(images_in.shape, dtype=int)
    for _label in labels:
        images[images_in == _label] = 1

    erosion_mask = generate_binary_structure(3, 1)
    i = 0
    while True:
        last_points = np.where(images != 0)
        images = binary_erosion(images, erosion_mask).astype(int)
        i += 1
        if not images.sum():
            print(f"Eroded all voxels after {i} erosions")
            break
    images[last_points] = 1
    last_points = getLargestCC(images)
    mass_center = np.array(last_points).transpose().mean(axis=0)
    mass_center = mass_center - VOXELMAN_CENTER
    return mass_center.tolist()


def point_within_organ(point, labels, images_path):
    images = read_images(images_path, extension=".tif")
    point = np.round(point + VOXELMAN_CENTER)
    x, y, z = point.astype(int)
    correct = int(images[x, y, z] in labels)
    return correct


def get_organ2summary(organ2voxels: str, num_anchors: int = 1000):
    organ2summary = {}

    for organ, voxels in organ2voxels.items():
        if len(voxels) > num_anchors:
            organ2summary[organ] = random.sample(voxels, num_anchors)
        else:
            organ2summary[organ] = np.array(voxels)[
                np.random.choice(range(len(voxels)), num_anchors)
            ].tolist()

    return organ2summary


def merge_organ_groups(
    src_dir,
    dst_dir,
    organ_groups: List[List],
    superorgan_names: List,
    superorgan_indices: List,
    images_path,
):
    assert any(
        isinstance(item, List) for item in organ_groups
    ), "Organ groups need to be a list of lists"
    assert isinstance(superorgan_names, List), "Superorgan names needs to be a list"
    assert isinstance(superorgan_indices, List), "Superorgan indices needs to be a list"
    assert (
        len(organ_groups) == len(superorgan_names) == len(superorgan_indices)
    ), "The number of organ groups, superorgan names and superorgan indices needs to match"

    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

    ind2organ = json.load(open(os.path.join(src_dir, "ind2organ.json")))
    organ2ind = json.load(open(os.path.join(src_dir, "organ2ind.json")))
    organ2label = json.load(open(os.path.join(src_dir, "organ2label.json")))
    organ2alias = json.load(open(os.path.join(src_dir, "organ2alias.json")))
    organ2center = json.load(open(os.path.join(src_dir, "organ2center.json")))
    organ2voxels = json.load(open(os.path.join(src_dir, "organ2voxels.json")))
    organ2voxels_eroded = json.load(
        open(os.path.join(src_dir, "organ2voxels_eroded.json"))
    )

    for organs_to_merge, superorgan_name, superorgan_index in zip(
        organ_groups, superorgan_names, superorgan_indices
    ):

        print(f"Merging: {organs_to_merge} into a superorgan: {superorgan_name}")

        ind2organ[superorgan_index] = superorgan_name
        organ2ind[superorgan_name] = int(superorgan_index)

        aliases = []
        labels = []
        voxels = []
        voxels_eroded = []
        for organ_to_merge in organs_to_merge:
            aliases = aliases + organ2alias[organ_to_merge]
            labels = labels + organ2label[organ_to_merge]
            voxels = voxels + organ2voxels[organ_to_merge]
            voxels_eroded = voxels + organ2voxels_eroded[organ_to_merge]

        organ2alias[superorgan_name] = aliases
        organ2label[superorgan_name] = labels
        organ2voxels[superorgan_name] = voxels
        organ2voxels_eroded[superorgan_name] = voxels_eroded
        organ2center[superorgan_name] = get_center_of_mass(
            organ2label[superorgan_name], images_path
        )

        if point_within_organ(
            organ2center[superorgan_name], organ2label[superorgan_name], images_path
        ):
            print("Center of mass is inside merged organ")
        else:
            print("Center of mass is not inside merged organ, that is an error")

        for organ_to_merge in organs_to_merge:
            del ind2organ[str(organ2ind[organ_to_merge])]
            del organ2ind[organ_to_merge]
            del organ2label[organ_to_merge]
            del organ2alias[organ_to_merge]
            del organ2center[organ_to_merge]
            del organ2voxels[organ_to_merge]
            del organ2voxels_eroded[organ_to_merge]

    organ2summary = get_organ2summary(organ2voxels, num_anchors=1000)

    store_json(ind2organ, os.path.join(dst_dir, "ind2organ.json"))
    store_json(organ2ind, os.path.join(dst_dir, "organ2ind.json"))
    store_json(organ2label, os.path.join(dst_dir, "organ2label.json"))
    store_json(organ2alias, os.path.join(dst_dir, "organ2alias.json"))
    store_json(organ2center, os.path.join(dst_dir, "organ2center.json"))
    store_json(organ2voxels, os.path.join(dst_dir, "organ2voxels.json"))
    store_json(organ2voxels_eroded, os.path.join(dst_dir, "organ2voxels_eroded.json"))
    store_json(organ2summary, os.path.join(dst_dir, "organ2summary.json"))


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Group organs into superorgans.")
    parser.add_argument(
        "--src_dir", type=str, help="Path to the source data organs directory path."
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        help="Path to the destination data organs directory path.",
    )
    parser.add_argument(
        "-og",
        "--organ_groups",
        type=str,
        nargs="+",
        action="append",
        help="List of lists - a list of organ groups that will be merged",
    )
    parser.add_argument(
        "-sn",
        "--superorgan_names",
        type=str,
        action="append",
        help="List of names of superorgans that will be formed",
    )
    parser.add_argument(
        "-si",
        "--superorgan_indices",
        type=str,
        action="append",
        help="List of indices of superorgans that will be formed",
    )
    parser.add_argument("--images_path", type=str, help="Path to the voxel-man images")
    return parser.parse_args()


def main():
    args = parse_args()
    merge_organ_groups(
        args.src_dir,
        args.dst_dir,
        args.organ_groups,
        args.superorgan_names,
        args.superorgan_indices,
        args.images_path,
    )


"""
respiratory_organs = ["bronchi", "diaphragm", "trachea", "lung", "larynx"]
digestive_organs_solid = ["gallbladder", "liver", "pancreas"]
digestive_organs_hollow = ["ampulla", "ascending colon", "duodenum", "cystic duct", "rectum", "sigmoid colon", "stomach", "transverse colon", "small intestine", "descending colon", "caecum"]
"""

if __name__ == "__main__":
    main()
