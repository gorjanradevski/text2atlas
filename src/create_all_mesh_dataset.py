import argparse
import json
import os
import random

import ndjson
from sklearn.model_selection import train_test_split as dataset_split
from tqdm import tqdm
from transformers import BertTokenizer

from utils.text import detect_occurrences, count_occurrences


def create_all_mesh_dataset(
    dset_path: str,
    all_mesh_path: str,
    organs_dir_path: str,
    organ_cap: int,
    train_percentage: float,
):
    """Create a dataset based on a directory containing json files with allMesh dataset abstracts
    Arguments:
        dset_path (str): Path to store dataset.
        all_mesh_path (str): Path to the directory with json files constaining allMesh dataset abstracts.
        organs_dir_path (str): Path to the directory with organ dictionaries.
        organ_cap (int): Maximum number of organ occurrences in dataset subset.
        train_percentage (float): Percentage of training set samples.
    """

    if not os.path.exists(os.path.dirname(dset_path)):
        os.makedirs(os.path.dirname(dset_path))

    organ2alias = json.load(open(os.path.join(organs_dir_path, "organ2alias.json")))
    organ2ind = json.load(open(os.path.join(organs_dir_path, "organ2ind.json")))

    # for organ, aliases in organ2alias.items():
    #     organ2alias[organ] = [alias.strip() for alias in aliases]

    dset = []
    for json_file in tqdm(os.listdir(all_mesh_path)):
        abstracts = ndjson.load(open(os.path.join(all_mesh_path, json_file)))
        dset.extend(abstracts)

    all_aliases = list(organ2alias.values())
    all_aliases = [item for sublist in all_aliases for item in sublist]
    mesh_term_count_dict = {}
    for abstract in tqdm(dset):
        mesh_terms = abstract["meshMajor"]
        count = len([item for item in mesh_terms if item.lower() in all_aliases])
        if count not in mesh_term_count_dict:
            mesh_term_count_dict[count] = 1
        else:
            mesh_term_count_dict[count] += 1
        abstract["organMeshCount"] = count

    num_single_organ = len(
        [abstract for abstract in dset if abstract["organMeshCount"] == 1]
    )

    num_multiple_organ = len(
        [abstract for abstract in dset if abstract["organMeshCount"] > 1]
    )

    print(f"Number of abstracts pertaining to one organ: {num_single_organ}")
    print(
        f"Number of abstracts pertaining to more than one organ: {num_multiple_organ}"
    )

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

    seen_organs = dict(zip(organ2alias.keys(), len(organ2alias) * [0]))
    dset_sample = []

    for i, abstract in enumerate(dset):
        organs = abstract["organ_names"]
        if any([seen_organs[organ] < organ_cap for organ in organs]):
            for organ in organs:
                seen_organs[organ] += 1
            dset_sample.append(abstract)
            if not i % 1000:
                print(
                    f"Percent complete: {sum(seen_organs.values())/(organ_cap * len(seen_organs))*100 :.2f}%"
                )
                print(f"Percent dset seen: {i/len(dset)*100 :.2f}%")

            if all([item == organ_cap for item in seen_organs.values()]):
                break

    """Taking only short abstracts"""
    print("Taking only short abstracts...")
    dset_sample = [
        sample
        for sample in dset_sample
        if len(tokenizer.encode(abstract["text"])) < 512
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

    dset_train, dset_val_test = dataset_split(dset_sample, test_size=train_percentage)
    dset_val, dset_test = dataset_split(dset_val_test, test_size=0.5)

    with open(dset_path, "w") as outfile:
        json.dump(dset, outfile)

    with open(
        os.path.splitext(dset_path)[0] + "_train" + os.path.splitext(dset_path)[1], "w"
    ) as outfile:
        json.dump(dset_train, outfile)

    with open(
        os.path.splitext(dset_path)[0] + "_val" + os.path.splitext(dset_path)[1], "w"
    ) as outfile:
        json.dump(dset_val, outfile)

    with open(
        os.path.splitext(dset_path)[0] + "_test" + os.path.splitext(dset_path)[1], "w"
    ) as outfile:
        json.dump(dset_test, outfile)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a dataset based on a directory containing json files with allMesh dataset abstracts and a directory of organs."
    )
    parser.add_argument(
        "--dset_path", type=str, help="Path under which the dataset is saved."
    )
    parser.add_argument(
        "--all_mesh_path",
        type=str,
        help="Path to the directory with json files containing abstracts.",
    )
    parser.add_argument(
        "--organs_dir_path",
        type=str,
        help="Path to the directory with organ dictionaries.",
    )
    parser.add_argument(
        "--organ_cap",
        type=int,
        default=1000,
        help="Soft cap to the number of organ appearances in the dataset.",
    )
    parser.add_argument(
        "--train_percentage",
        type=float,
        default=0.7,
        help="Percentage of training set samples.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    create_all_mesh_dataset(
        args.dset_path,
        args.all_mesh_path,
        args.organs_dir_path,
        args.organ_cap,
        args.train_percentage,
    )


if __name__ == "__main__":
    main()
