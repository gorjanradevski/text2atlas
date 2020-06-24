import json
import random
from typing import Dict

import numpy as np


def create_ind2anchors(
    organ2ind_path: str, organ2voxels_path: str, num_anchors: int = 1000
) -> Dict:
    """Create a dictionary where organ anchor points are indexed by the organ index
    Arguments:
        organ2ind_path {str} -- Path to the dictionary with the organ names and their corresponding indices
        organ2voxels_path {str} -- Path to the dictionary with the organ names and their corresponding anchor points
    Keyword Arguments:
        num_anchors {int} -- [description] (default: {1000})
    Returns:
        Dict -- Dictionary containing organ indices and their corresponding random subset of anchors
    """  # noqa: E501
    organ2ind = json.load(open(organ2ind_path))
    organ2voxels = json.load(open(organ2voxels_path))
    ind2voxels = {}

    for organ, ind in organ2ind.items():
        if len(organ2voxels[organ]) > num_anchors:
            ind2voxels[ind] = random.sample(organ2voxels[organ], num_anchors)
        else:
            ind2voxels[ind] = np.array(organ2voxels[organ])[
                np.random.choice(range(len(organ2voxels[organ])), num_anchors)
            ].tolist()

    ind2voxels[-1] = np.zeros(shape=(num_anchors, 3), dtype=np.float)

    return ind2voxels


def create_ind2centers(organ2ind_path: str, organ2center_path: str) -> Dict:
    """Create a dictionary where organ centers are indexed by the organ index
    Arguments:
        organ2ind_path {str} -- Path to the dictionary with the organ names and their corresponding indices
        organ2center_path {str} -- Path to the dictionary with the organ names and their corresponding centers
    Returns:
        Dict -- Dictionary containing organ indices and their corresponding centers
    """  # noqa: E501
    organ2ind = json.load(open(organ2ind_path))
    organ2center = json.load(open(organ2center_path))
    ind2centers = {}

    for organ, ind in organ2ind.items():
        ind2centers[ind] = np.array(organ2center[organ])

    ind2centers[-1] = np.zeros(shape=(3,), dtype=np.float)

    return ind2centers
