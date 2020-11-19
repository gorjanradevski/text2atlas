import argparse
import json
import os
import random
import re
from typing import Dict, List

import natsort
import numpy as np
import scispacy  # noqa: F401
import spacy
import tifffile
from scipy.ndimage import binary_erosion, generate_binary_structure
from scispacy.umls_linking import UmlsEntityLinker
from skimage.measure import label

from voxel_mapping.constants import VOXELMAN_CENTER


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

    for _label in labels:
        if type(label) is list:
            indices = np.logical_or.reduce([images == item for item in _label])
        else:
            indices = images == _label
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
        
    indices = np.zeros(images.shape).astype(bool)
    for _label in labels:
        indices = np.logical_or.reduce([indices, images == _label])

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
    for organ, labels in organ2label.items():
        organ2voxels[organ] = return_voxels(images.copy(), labels)
    return organ2voxels


def generate_organ2voxels_eroded(images_path: str, organ2label: Dict):
    images = read_images(images_path)
    organ2voxels = {}
    for organ, labels in organ2label.items():
        organ2voxels[organ] = return_voxels_eroded(images.copy(), labels)
    return organ2voxels


def create_organ2summary(organ2voxels, num_anchors: int = 1000):
    organ2summary = {}

    for organ, voxels in organ2voxels.items():
        if len(organ2voxels[organ]) > num_anchors:
            organ2summary[organ] = random.sample(voxels, num_anchors)
        else:
            organ2summary[organ] = np.array(voxels)[
                np.random.choice(range(len(voxels)), num_anchors)
            ].tolist()

    return organ2summary


def retrieve_alias_terms(organ_names: List[str], nlp, linker):
    
    organ_name_aliases = {}
    for organ in organ_names:
        names = [organ]
        organ_doc = nlp(organ)
        print(organ)
        if len(organ_doc.ents) == 1:
            entity = organ_doc.ents[0]
            print("     {} {}".format(entity, (entity.start_char, entity.end_char)))
            if entity._.umls_ents:
                umls_entity = linker.umls.cui_to_entity[entity._.umls_ents[0][0]]
                names.extend(umls_entity.aliases)
        names = list(set([name.lower() for name in names]))
        print(names)
        names.append(organ)
        names = list(set(names))
        organ_name_aliases[organ] = names

    return organ_name_aliases


def create_organ_dicts(sio_atlas_path, organs_dir_path):

    voxelman_images_path = os.path.join(sio_atlas_path, "labels")
    organ_list_path = os.path.join(sio_atlas_path, "objectlist.txt")

    organ_list = open(organ_list_path).read().strip().split("\n")[17:]

    """Extract list of labels"""
    organ2label = {}
    for entry in organ_list:
        name, labels = entry.split('" ')
        labels = labels.split()
        organ2label[name[1:]] = [int(label) for label in labels]

    """Keep track of mergers"""
    organ2alias = {}
    for organ in organ2label.keys():
        organ2alias[organ] = [organ]

    """Removal of bones, limb tissues and location unspecific tissues"""
    organs_to_remove = "bones of the left hand, bones of the right hand, cervical vertebra C5, cervical vertebra C6, cervical vertebra C7, coccyx, grey matter, intervertebral disc C6/C7, intervertebral disc C7/T1, intervertebral disc L1/L2, intervertebral disc L2/L3, intervertebral disc L3/L4, intervertebral disc L4/L5, intervertebral disc L5/S1, intervertebral disc S1/S2, intervertebral disc T1/T2, intervertebral disc T2/T3, intervertebral disc T3/T4, intervertebral disc T4/T5, intervertebral disc T5/T6, intervertebral disc T6/T7, intervertebral disc T7/T8, intervertebral disc T8/T9, intervertebral disc T9/T10, intervertebral disc T10/T11, intervertebral disc T11/T12, intervertebral disc T12/L1, left rib 1, left rib 2, left rib 3, left rib 4, left rib 5, left rib 6, left rib 7, left rib 8, left rib 9, left rib 10, left rib 11, left rib 12, left ulna, left scapula, left radius, left humerus, left hip bone, left femur, left clavicle, muscles of the left arm, muscles of the right arm, lumbar vertebra L1, lumbar vertebra L2, lumbar vertebra L3, lumbar vertebra L4, lumbar vertebra L5, marker 1, marker 2, marker 3, right rib 1, right rib 2, right rib 3, right rib 4, right rib 5, right rib 6, right rib 7, right rib 8, right rib 9, right rib 10, right rib 11, right rib 12, right ulna, right scapula, right radius, right humerus, right hip bone, right femur, right clavicle, skin of the left arm, skin of the right arm, thoracic vertebra T1, thoracic vertebra T2, thoracic vertebra T3, thoracic vertebra T4, thoracic vertebra T5, thoracic vertebra T6, thoracic vertebra T7, thoracic vertebra T8, thoracic vertebra T9, thoracic vertebra T10, thoracic vertebra T11, thoracic vertebra T12, unclassified bones, unclassified cartilage, unclassified muscles, unclassified skin, unclassified tissue, unclassified tissue of the left arm, unclassified tissue of the right arm, unclassified veins, white matter, sternum, sacrum, left costal cartilage 1, left costal cartilage 2, left costal cartilage 3, left costal cartilage 4, left costal cartilage 5, left costal cartilage 6-9, right costal cartilage 1, right costal cartilage 2, right costal cartilage 3, right costal cartilage 4, right costal cartilage 5, right costal cartilage 6-9, right clavicular cartilage, left clavicular cartilage"  # noqa: E501
    organs_to_remove = organs_to_remove.split(", ")
    for item in organs_to_remove:
        del organ2label[item]
        del organ2alias[item]

    """Removal of bilateral organs on the right side"""
    organs_to_remove_right = "right atrium, right external oblique, right iliacus, right internal oblique, right jugular vein, right kidney, right lung, right obturator internus, right psoas, right rectus abdominis, right renal medulla, right renal vein, right subclavian vein, right transversus abdominis, right ventricle"  # noqa: E501
    organs_to_remove_right = organs_to_remove_right.split(", ")
    for item in organs_to_remove_right:
        del organ2label[item]
        del organ2alias[item]

    """Removal of thorax muscles, scrotum visceral fat"""
    organs_to_remove_muscles = "scrotum, visceral fat, left psoas, left iliacus, left external oblique, left rectus abdominis, left internal oblique, left transversus abdominis, left obturator internus, ischiocavernosus, pelvic diaphragm, rectus sheath"  # noqa: E501
    organs_to_remove_muscles = organs_to_remove_muscles.split(", ")
    for item in organs_to_remove_muscles:
        del organ2label[item]
        del organ2alias[item]

    """Removal of blood vessels"""
    organs_to_remove_blood_vessels = "superior vena cava, superior mesenteric vein, splenic vein, pulmonary veins, pulmonary trunk, pulmonary arteries, portal vein, left subclavian vein, left jugular vein, inferior vena cava, inferior mesenteric vein, hepatic veins, descending aorta, brachiocephalic vein, azygos vein, arch of aorta, abdominal aorta, left renal vein, ascending aorta"  # noqa: E501
    organs_to_remove_blood_vessels = organs_to_remove_blood_vessels.split(", ")
    for item in organs_to_remove_blood_vessels:
        del organ2label[item]
        del organ2alias[item]

    """Removal of small organs with less than 1000 voxels"""
    organs_to_remove_small = "cystic duct"
    organs_to_remove_small = organs_to_remove_small.split(", ")
    for item in organs_to_remove_small:
        del organ2label[item]
        del organ2alias[item]
        
    """Mergers of stomach segments into "stomach"""
    organs_to_merge_stomach = "fundus of stomach, greater curvature, lesser curvature, body of stomach, cardia, stomach"
    organs_to_merge_stomach = organs_to_merge_stomach.split(", ")
    dest_organ = "stomach"
    labels = []
    names = []
    for item in organs_to_merge_stomach:
        labels += organ2label[item]
        names.append(item)
        del organ2label[item]
        del organ2alias[item]
    organ2label[dest_organ] = labels
    organ2alias[dest_organ] = list(set([dest_organ] + names))

    """Mergers of colon segments into "colon"""
    organs_to_merge_colon = "ascending colon, descending colon, transverse colon, sigmoid colon, left colic flexure, right colic flexure"  # noqa: E501
    organs_to_merge_colon = organs_to_merge_colon.split(", ")
    dest_organ = "colon"
    labels = []
    names = []
    for item in organs_to_merge_colon:
        labels += organ2label[item]
        names.append(item)
        del organ2label[item]
        del organ2alias[item]
    organ2label[dest_organ] = labels
    organ2alias[dest_organ] = list(set([dest_organ] + names))

    """Mergers of penis segments into "penis"""
    organs_to_merge_penis = "penis, corpus cavernosum penis, corpus spongiosum penis"
    organs_to_merge_penis = organs_to_merge_penis.split(", ")
    dest_organ = "penis"
    labels = []
    names = []
    for item in organs_to_merge_penis:
        labels += organ2label[item]
        names.append(item)
        del organ2label[item]
        del organ2alias[item]
    organ2label[dest_organ] = labels
    organ2alias[dest_organ] = list(set([dest_organ] + names))

    """Mergers of trachea and trachea lumen into "trachea"""
    organs_to_merge_trachea = "trachea, trachea lumen"
    organs_to_merge_trachea = organs_to_merge_trachea.split(", ")
    dest_organ = "trachea"
    labels = []
    names = []
    for item in organs_to_merge_trachea:
        labels += organ2label[item]
        names.append(item)
        del organ2label[item]
        del organ2alias[item]
    organ2label[dest_organ] = labels
    organ2alias[dest_organ] = list(set([dest_organ] + names))

    """Mergers of left kidney and left renal medulla into "left kidney"""
    organs_to_merge_kidney = "left renal medulla, left kidney"
    organs_to_merge_kidney = organs_to_merge_kidney.split(", ")
    dest_organ = "left kidney"
    labels = []
    names = []
    for item in organs_to_merge_kidney:
        labels += organ2label[item]
        names.append(item)
        del organ2label[item]
        del organ2alias[item]
    organ2label[dest_organ] = labels
    organ2alias[dest_organ] = list(set([dest_organ] + names))

    """Renaming of paired organs to just the name of the organ"""
    organs_to_rename_left = "left ventricle, left atrium, left kidney, left lung"
    organs_to_rename_left = organs_to_rename_left.split(", ")
    target_names = "ventricle, atrium, kidney, lung"
    target_names = target_names.split(", ")

    for organ_to_rename, target_name in zip(organs_to_rename_left, target_names):
        organ2label[target_name] = organ2label[organ_to_rename]
        organ2alias[target_name] = organ2alias[organ_to_rename]
        if organ_to_rename in organ2alias[target_name]:
            organ2alias[target_name].remove(organ_to_rename)
            organ2alias[target_name].append(target_name)
        del organ2label[organ_to_rename]
        del organ2alias[organ_to_rename]

    """Renaming duodenum (retroperitoneal part) to duodenum"""
    organs_to_rename_duodenum = "duodenum (retroperitoneal part)"
    organs_to_rename_duodenum = organs_to_rename_duodenum.split(", ")
    target_names = "duodenum"
    target_names = target_names.split(", ")

    for organ_to_rename, target_name in zip(organs_to_rename_duodenum, target_names):
        organ2label[target_name] = organ2label[organ_to_rename]
        organ2alias[target_name] = organ2alias[organ_to_rename]
        if organ_to_rename in organ2alias[target_name]:
            organ2alias[target_name].remove(organ_to_rename)
            organ2alias[target_name].append(target_name)
        del organ2label[organ_to_rename]
        del organ2alias[organ_to_rename]

    """
    Adding jejunum and ileum aliases to small intestine
    Perhaps later we can check if sentences with jejunum is above ileum
    (as it should be)
    """
    target_organ = "small intestine"
    aliases = "jejunum, ileum"
    aliases = aliases.split(", ")
    for alias in aliases:
        organ2alias[target_organ].append(alias)

    """
    Adding heart atria alias to atrium
    """
    target_organ = "atrium"
    aliases = "heart atria"
    aliases = aliases.split(", ")
    for alias in aliases:
        organ2alias[target_organ].append(alias)

    """
    Adding heart ventricles alias to ventricle
    """
    target_organ = "ventricle"
    aliases = "heart ventricles"
    aliases = aliases.split(", ")
    for alias in aliases:
        organ2alias[target_organ].append(alias)

    """
    Adding cecum alias to caecum
    """
    target_organ = "caecum"
    aliases = "cecum"
    aliases = aliases.split(", ")
    for alias in aliases:
        organ2alias[target_organ].append(alias)

    """
    Adding ampulla of vater alias to ampulla
    """
    target_organ = "ampulla"
    aliases = "ampulla of vater"
    aliases = aliases.split(", ")
    for alias in aliases:
        organ2alias[target_organ].append(alias)

    """
    Adding ampulla of vater alias to ampulla
    """
    target_organ = "ampulla"
    aliases = "ampulla of vater"
    aliases = aliases.split(", ")
    for alias in aliases:
        organ2alias[target_organ].append(alias)

    """
    Adding seminal vesicles alias to seminal gland
    """
    target_organ = "seminal gland"
    aliases = "seminal vesicles"
    aliases = aliases.split(", ")
    for alias in aliases:
        organ2alias[target_organ].append(alias)

    """
    Adding colon, ascending, colon, descending, colon, transverse, colon, sigmoid alias, and colic flexure names to colon  # noqa: E501
    """
    target_organ = "colon"
    aliases = [
        "colon, ascending",
        "colon, descending",
        "colon, transverse",
        "colon, sigmoid",
        "hepatic flexure",
        "splenic flexure",
        "colic flexure",
    ]
    for alias in aliases:
        organ2alias[target_organ].append(alias)

    """Random fixes"""
    organ2alias["kidney"] = ["renal medulla", "kidney"]
    organ2alias["colon"].remove("right colic flexure")
    organ2alias["colon"].remove("left colic flexure")
    organ2alias["bronchi"].append("bronchus")
    organ2alias["ampulla"] = ["ampulla", "ampulla of vater"]

    """Generate alias terms"""
    print("Generating Alias Terms...")
    nlp = spacy.load("en_core_sci_sm")
    linker = UmlsEntityLinker(resolve_abbreviations=True)
    nlp.add_pipe(linker)

    all_organ_words = list(organ2alias.values())
    all_organ_words = [item for sublist in all_organ_words for item in sublist]
    organ_name_aliases = retrieve_alias_terms(all_organ_words, nlp, linker)
    for organ, aliases in organ2alias.items():
        new_aliases = []
        for alias in aliases:
            new_aliases.extend(organ_name_aliases[alias])
        organ2alias[organ] = list(set(aliases + new_aliases))

    for organ, aliases in organ2alias.items():
        aliases = [
            re.sub(r"[\(\[][^)\]]+[\)\]]", r"", alias).strip() for alias in aliases
        ]
        aliases = [re.sub(r"(, )*nos$", r"", alias).strip() for alias in aliases]
        aliases = [re.sub(r"structure$", r"", alias).strip() for alias in aliases]
        aliases = [re.sub(r"structure of", r"", alias).strip() for alias in aliases]
        aliases = [alias for alias in aliases if ">" not in alias]
        aliases = [alias for alias in aliases if not re.search(r"\d+", alias)]

        aliases = list(set(aliases))
        organ2alias[organ] = aliases

    """Generate voxels"""
    print("Generating Dictionaries...")

    if not os.path.exists(organs_dir_path):
        os.makedirs(organs_dir_path)

    organ2ind = dict(zip(organ2alias.keys(), range(len(organ2alias))))
    ind2organ = dict(zip(range(len(organ2alias)), organ2alias.keys()))

    with open(os.path.join(organs_dir_path, "organ2ind.json"), "w") as outfile:
        json.dump(organ2ind, outfile)
    with open(os.path.join(organs_dir_path, "ind2organ.json"), "w") as outfile:
        json.dump(ind2organ, outfile)
    with open(os.path.join(organs_dir_path, "organ2label.json"), "w") as outfile:
        json.dump(organ2label, outfile)
    with open(os.path.join(organs_dir_path, "organ2alias.json"), "w") as outfile:
        json.dump(organ2alias, outfile)

    organ2voxels = generate_organ2voxels(voxelman_images_path, organ2label)
    organ2center = {}
    for organ, labels in organ2label.items():
        organ2center[organ] = get_center_of_mass(labels, voxelman_images_path)
        in_organ = point_within_organ(organ2center[organ], labels, voxelman_images_path)
        if in_organ:
            print("Center of mass is inside organ")
        else:
            print("Center of mass is not inside organ, that is an error")
    organ2summary = create_organ2summary(organ2voxels, 1000)

    with open(os.path.join(organs_dir_path, "organ2center.json"), "w") as outfile:
        json.dump(organ2center, outfile)
    with open(os.path.join(organs_dir_path, "organ2voxels.json"), "w") as outfile:
        json.dump(organ2voxels, outfile)
    with open(os.path.join(organs_dir_path, "organ2summary.json"), "w") as outfile:
        json.dump(organ2summary, outfile)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create organ dictionaries based on the human atlas."
    )
    parser.add_argument(
        "--sio_atlas_path",
        type=str,
        help="Path to the directory with a subdirectory containing images ('labels/')\
            and the text file with atlas classes ('objectlist.txt').",
    )
    parser.add_argument(
        "--organs_dir_path",
        type=str,
        help="Path to the directory where the organ dictionaries will be stored",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    create_organ_dicts(args.sio_atlas_path, args.organs_dir_path)


if __name__ == "__main__":
    main()
