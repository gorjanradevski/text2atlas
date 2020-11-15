import argparse
import json
import ndjson
import os
import random
import logging

from sklearn.model_selection import train_test_split as dataset_split
from tqdm import tqdm
from transformers import BertTokenizer
from typing import List, Dict
from utils.text import count_occurrences, detect_occurrences

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


def fix_keyword_detection_issues(
    dset: List, organ2ind: Dict
):
    """Fix potential issues with keyword detection.

    Args:
        dset (List): List of dataset samples
        organ2ind (Dict): Dictionary with organ index for each organ

    Returns:
        (List): List of dataset samples with fixed keywords.
    """   
    
    """SOLVE CARDIA""" 

    for ind, abstract in tqdm(enumerate(dset)):
        keywords = abstract["keywords"]
        occ_organ_indices = abstract["occ_organ_indices"]
        occ_organ_names = abstract["occ_organ_names"]
        organ_names = abstract["organ_names"]
        if (
            "cardiac" in keywords
            and "stomach" in occ_organ_names
            and any(
                [
                    item in organ_names
                    for item in ["atrium", "ventricle", "myocardium", "pericardium"]
                ]
            )
        ):
            occ_organ_indices.remove(organ2ind["stomach"])
            occ_organ_names.remove("stomach")
        if (
            "cardia" in keywords
            and "myocardium" in occ_organ_names
            and any([item in organ_names for item in ["stomach"]])
        ):
            occ_organ_indices.remove(organ2ind["myocardium"])
            occ_organ_names.remove("myocardium")
        abstract["occ_organ_indices"] = occ_organ_indices
        abstract["occ_organ_names"] = occ_organ_names

    inds = []
    for ind, abstract in tqdm(enumerate(dset)):
        keywords = abstract["keywords"]
        occ_organ_indices = abstract["occ_organ_indices"]
        occ_organ_names = abstract["occ_organ_names"]
        organ_names = abstract["organ_names"]
        if "cardiac" in keywords and "stomach" in occ_organ_names:
            inds.append(ind)

    """SOLVE THE LIVER - DELIVER PROBLEM"""

    for ind, abstract in tqdm(enumerate(dset)):
        keywords = abstract["keywords"]
        occ_organ_indices = abstract["occ_organ_indices"]
        occ_organ_names = abstract["occ_organ_names"]
        organ_names = abstract["organ_names"]
        if (
            any(
                [
                    keyword in keywords
                    for keyword in ["delivery", "delivered", "deliver", "delivering"]
                ]
            )
            and "liver" not in organ_names
        ):
            occ_organ_indices.remove(organ2ind["liver"])
            occ_organ_names.remove("liver")
        keywords = [
            keyword
            for keyword in keywords
            if keyword not in ["delivery", "delivered", "deliver", "delivering"]
        ]
        abstract["occ_organ_indices"] = occ_organ_indices
        abstract["occ_organ_names"] = occ_organ_names
        abstract["keywords"] = keywords

    """SOLVE THE COLON - COLONISE PROBLEM"""

    inds = []
    for ind, abstract in tqdm(enumerate(dset)):
        keywords = abstract["keywords"]
        occ_organ_indices = abstract["occ_organ_indices"]
        occ_organ_names = abstract["occ_organ_names"]
        organ_names = abstract["organ_names"]
        if (
            any(
                [
                    keyword in keywords
                    for keyword in [
                        "colonize",
                        "colonise",
                        "colonized",
                        "colonised",
                        "colonies",
                    ]
                ]
            )
            and "colon" not in organ_names
        ):
            occ_organ_indices.remove(organ2ind["colon"])
            occ_organ_names.remove("colon")
        keywords = [
            keyword
            for keyword in keywords
            if keyword
            not in ["colonize", "colonise", "colonized", "colonised", "colonies"]
        ]
        abstract["occ_organ_indices"] = occ_organ_indices
        abstract["occ_organ_names"] = occ_organ_names
        abstract["keywords"] = keywords

    """SOLVE THE BLADDER - GALLBLADDER PROBLEM"""

    """Gallbladder doesn't cause the bladder keyword"""
    """Bladder does cause problems"""

    for ind, abstract in tqdm(enumerate(dset)):
        keywords = abstract["keywords"]
        occ_organ_indices = abstract["occ_organ_indices"]
        occ_organ_names = abstract["occ_organ_names"]
        organ_names = abstract["organ_names"]
        if (
            any([keyword in keywords for keyword in ["bladder", "bladders"]])
            and any(
                [
                    keyword in keywords
                    for keyword in [
                        "gall",
                        "gallbladder",
                        "gall-bladder",
                        "gallbladders",
                        "gall-bladders",
                    ]
                ]
            )
            and "gallbladder" in organ_names
        ):
            occ_organ_indices.remove(organ2ind["urinary bladder"])
            occ_organ_names.remove("urinary bladder")
            keywords = [
                keyword
                for keyword in keywords
                if keyword not in ["bladder", "bladders"]
            ]
        abstract["occ_organ_indices"] = occ_organ_indices
        abstract["occ_organ_names"] = occ_organ_names
        abstract["keywords"] = keywords

    return dset


def create_all_mesh_dataset(
    all_mesh_path: str,
    dst_dset_path: str,
    organs_dir_path: str,
    organ_cap_single: int,
    organ_cap_multi: int,
    train_percentage: float,
    max_tokens: int = 512,
):
    """Create a dataset based on a directory containing json files with allMesh dataset abstracts
    Arguments:
        all_mesh_path (str): Path to the directory with json files containing abstracts.
        dst_dset_path (str): Path under which the dataset is saved.
        organs_dir_path (str): Path to the directory with organ dictionaries.
        organ_cap_single (int): Maximum number of organ occurrences in aggregated samples with single organs.
        organ_cap_multi (int): Maximum number of organ occurrences in aggregated samples with multiple organs.
        train_percentage (float): Percentage of training set samples.
        max_tokens (bool): Maximum number of word tokens.
    """

    if not os.path.exists(os.path.dirname(dst_dset_path)):
        os.makedirs(os.path.dirname(dst_dset_path))

    dset = []
    for json_file in tqdm(os.listdir(all_mesh_path)):
        abstracts = ndjson.load(open(os.path.join(all_mesh_path, json_file)))
        dset.extend(abstracts)
        
    dset = dset[:10000]

    organ2alias = json.load(open(os.path.join(organs_dir_path, "organ2alias.json")))
    organ2ind = json.load(open(os.path.join(organs_dir_path, "organ2ind.json")))
    
    for abstract in tqdm(dset):
        organ_names = []
        organ_indices = []
        mesh_terms = abstract["meshMajor"]
        for organ, aliases in organ2alias.items():
            if any([mesh_term.lower() in aliases for mesh_term in mesh_terms]):
                organ_names.append(organ)
                organ_indices.append(organ2ind[organ])
        if "organMeshCount" in abstract:
            del abstract["organMeshCount"]
        abstract["text"] = abstract["abstractText"]
        del abstract["abstractText"]
        abstract["organ_names"] = organ_names
        abstract["organ_indices"] = organ_indices
        abstract["mesh_terms"] = abstract["meshMajor"]
        abstract["keywords"] = []
        del abstract["meshMajor"]
        
    """Remove abstracts with animal related mesh terms"""
    animal_mesh_terms = [
        "Animals",
        "Rats",
        "Mice",
        "Rats, Sprague-Dawley",
        "Rats, Wistar",
        "Mice, Inbred C57BL",
        "Rats, Inbred Strains",
        "Disease Models, Animal",
        "Dogs",
        "Rabbits",
        "Swine",
        "Mice, Inbred BALB C",
        "Guinea Pigs",
        "Mice, Knockout",
        "Cattle",
        "Animals, Newborn",
        "Mice, Transgenic",
        "Chickens",
        "Sheep",
        "Mice, Inbred Strains",
        "Rats, Inbred F344",
    ]
    dset = [
        item
        for item in dset
        if not any([mesh_term in animal_mesh_terms for mesh_term in item["mesh_terms"]])
    ]
    
    """Subsample dataset"""
    print("Subsampling the dataset...")
    random.shuffle(dset)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    dset_single = [item for item in dset if len(item["organ_names"]) == 1]
    dset_multi = [item for item in dset if len(item["organ_names"]) > 1]

    random.shuffle(dset_single)
    random.shuffle(dset_multi)

    seen_organs = dict(zip(organ2alias.keys(), len(organ2alias) * [0]))
    dset_sample_single = []
    print("Taking samples with single organ occurrence...")

    for i, abstract in enumerate(dset_single):
        organs = abstract["organ_names"]
        if any([seen_organs[organ] < organ_cap_single for organ in organs]):
            for organ in organs:
                seen_organs[organ] += 1
            dset_sample_single.append(abstract)

            if all([item == organ_cap_single for item in seen_organs.values()]):
                break

    seen_organs = dict(zip(organ2alias.keys(), len(organ2alias) * [0]))
    dset_sample_multi = []
    print("Taking samples with multiple organ occurrences...")

    for i, abstract in enumerate(dset_multi):
        organs = abstract["organ_names"]
        if any([seen_organs[organ] < organ_cap_multi for organ in organs]):
            for organ in organs:
                seen_organs[organ] += 1
            dset_sample_multi.append(abstract)

            if all([item == organ_cap_multi for item in seen_organs.values()]):
                break

    dset_sample = dset_sample_single + dset_sample_multi

    """Taking only short abstracts"""
    print("Taking only short abstracts...")
    dset_sample = [
        sample
        for sample in dset_sample
        if len(tokenizer.encode(sample["text"])) < max_tokens
    ]

    """Create mask words"""
    print("Generating maskwords...")
    organ2alias = json.load(open(os.path.join(organs_dir_path, "organ2alias.json")))
    all_aliases = [item for sublist in organ2alias.values() for item in sublist]
    for abstract in tqdm(dset_sample):
        abstract["keywords"] = detect_occurrences(abstract["text"], all_aliases)

    """Count organ occurrences"""
    for abstract in tqdm(dset_sample):
        text = abstract["text"]
        occ_organ_names = []
        occ_organ_indices = []
        for organ, aliases in organ2alias.items():
            if count_occurrences(text, aliases):
                occ_organ_names.append(organ)
                occ_organ_indices.append(organ2ind[organ])
        abstract["occ_organ_names"] = occ_organ_names
        abstract["occ_organ_indices"] = occ_organ_indices
        
    """Fix keyword issues"""
    dset_sample = fix_keyword_detection_issues(dset_sample, organ2ind) 

    """Count organ appearances via mesh terms"""
    organ_count_dict = {}
    for abstract in tqdm(dset_sample):
        organ_names = abstract["organ_names"]
        for organ_name in organ_names:
            if organ_name not in organ_count_dict:
                organ_count_dict[organ_name] = 1
            else:
                organ_count_dict[organ_name] += 1
    print("Organ mesh term appearance counts in dataset...")
    print(organ_count_dict)

    """Count organ appearances via organ occurrences"""
    occ_organ_count_dict = {}
    for abstract in tqdm(dset_sample):
        organ_names = abstract["occ_organ_names"]
        for organ_name in organ_names:
            if organ_name not in occ_organ_count_dict:
                occ_organ_count_dict[organ_name] = 1
            else:
                occ_organ_count_dict[organ_name] += 1
    print("Organ occurrence counts in dataset...")
    print(occ_organ_count_dict)

    dset_train, dset_val_test = dataset_split(dset_sample, train_size=train_percentage)
    dset_val, dset_test = dataset_split(dset_val_test, test_size=0.5)

    with open(dst_dset_path, "w") as outfile:
        json.dump(dset_sample, outfile)

    with open(
        os.path.splitext(dst_dset_path)[0]
        + "_train"
        + os.path.splitext(dst_dset_path)[1],
        "w",
    ) as outfile:
        json.dump(dset_train, outfile)

    with open(
        os.path.splitext(dst_dset_path)[0]
        + "_val"
        + os.path.splitext(dst_dset_path)[1],
        "w",
    ) as outfile:
        json.dump(dset_val, outfile)

    with open(
        os.path.splitext(dst_dset_path)[0]
        + "_test"
        + os.path.splitext(dst_dset_path)[1],
        "w",
    ) as outfile:
        json.dump(dset_test, outfile)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create and split a dataset from a directory of ndjsons containing AllMesh articles."
    )
    parser.add_argument(
        "--all_mesh_path",
        type=str,
        help="Path to the directory with json files containing abstracts.",
    )
    parser.add_argument(
        "--dst_dset_path", 
        type=str, 
        help="Path under which the dataset is saved, should be a file path with json extension."
    )
    parser.add_argument(
        "--organs_dir_path",
        type=str,
        help="Path to the directory with organ dictionaries.",
    )
    parser.add_argument(
        "--organ_cap_single",
        type=int,
        default=750,
        help="Soft cap to the number of organ appearances in the dataset.",
    )
    parser.add_argument(
        "--organ_cap_multi",
        type=int,
        default=500,
        help="Soft cap to the number of organ appearances in the dataset.",
    )
    parser.add_argument(
        "--train_percentage",
        type=float,
        default=0.7,
        help="Percentage of training set samples.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens in an abstract, longer abstracts are discarded."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    create_all_mesh_dataset(
        args.all_mesh_path,
        args.dst_dset_path,
        args.organs_dir_path,
        args.organ_cap_single,
        args.organ_cap_multi,
        args.train_percentage,
        args.max_tokens
    )


if __name__ == "__main__":
    main()
