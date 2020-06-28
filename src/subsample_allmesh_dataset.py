import argparse
import json
import os
import random
import logging

import seaborn as sns  # noqa: F401
from matplotlib import pyplot as plt  # noqa: F401
from sklearn.model_selection import train_test_split as dataset_split
from tqdm import tqdm
from transformers import BertTokenizer

from utils.text import count_occurrences, detect_occurrences

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


def create_all_mesh_dataset(
    src_dset_path: str,
    dst_dset_path: str,
    organs_dir_path: str,
    organ_cap_single: int,
    organ_cap_multi: int,
    train_percentage: float,
    generate_maskwords: bool = False,
    take_short: bool = True,
):
    """Create a dataset based on a directory containing json files with allMesh dataset abstracts
    Arguments:
        src_dset_path (str): Path to the source dataset.
        dst_dset_path (str): Path under which the dataset is saved.
        organs_dir_path (str): Path to the directory with organ dictionaries.
        organ_cap (int): Maximum number of organ occurrences in dataset subset.
        train_percentage (float): Percentage of training set samples.
        generate_maskwords (bool): Whether to recalculate keywords.
    """

    if not os.path.exists(os.path.dirname(dst_dset_path)):
        os.makedirs(os.path.dirname(dst_dset_path))

    dset = json.load(open(src_dset_path))

    organ2alias = json.load(open(os.path.join(organs_dir_path, "organ2alias.json")))
    organ2ind = json.load(open(os.path.join(organs_dir_path, "organ2ind.json")))

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
            if not i % 1000:
                print(
                    f"Percent complete: {sum(seen_organs.values())/(organ_cap_single * len(seen_organs))*100 :.2f}%"
                )
                print(f"Percent dset seen: {i/len(dset_single)*100 :.2f}%")

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
            if not i % 1000:
                print(
                    f"Percent complete: {sum(seen_organs.values())/(organ_cap_multi * len(seen_organs))*100 :.2f}%"
                )
                print(f"Percent dset seen: {i/len(dset_multi)*100 :.2f}%")

            if all([item == organ_cap_multi for item in seen_organs.values()]):
                break

    dset_sample = dset_sample_single + dset_sample_multi

    """Taking only short abstracts"""
    if take_short:
        print("Taking only short abstracts...")
        dset_sample = [
            sample
            for sample in dset_sample
            if len(tokenizer.encode(sample["text"])) < 512
        ]

    """Create mask words"""
    if generate_maskwords:
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
        description="Subsample a given dataset, pick only abstracts shorter than 512 tokens."
    )
    parser.add_argument("--src_dset_path", type=str, help="Path to the source dataset.")
    parser.add_argument(
        "--dst_dset_path", type=str, help="Path under which the dataset is saved."
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

    return parser.parse_args()


def main():
    args = parse_args()
    create_all_mesh_dataset(
        args.src_dset_path,
        args.dst_dset_path,
        args.organs_dir_path,
        args.organ_cap_single,
        args.organ_cap_multi,
        args.train_percentage,
    )


if __name__ == "__main__":
    main()
