import argparse
import json
import os


def merge_dataset_groups(
    src_dset_path,
    dst_dset_path,
    src_dir,
    organ_groups_names,
    superorgan_indices,
    superorgan_names,
):
    samples = json.load(open(src_dset_path))

    organ2ind = json.load(open(os.path.join(src_dir, "organ2ind.json")))
    organ_groups_indices = [
        [organ2ind[organ] for organ in organ_group_names]
        for organ_group_names in organ_groups_names
    ]
    index_mapping_dict = {}
    name_mapping_dict = {}
    for (
        organ_group_indices,
        organ_group_names,
        superorgan_index,
        superorgan_name,
    ) in zip(
        organ_groups_indices, organ_groups_names, superorgan_indices, superorgan_names
    ):
        for organ_group_index, organ_group_name in zip(
            organ_group_indices, organ_group_names
        ):
            index_mapping_dict[organ_group_index] = int(superorgan_index)
            name_mapping_dict[organ_group_name] = superorgan_name

    for sample in samples:

        organ_indices = sample["organ_indices"]
        organ_names = sample["organ_names"]
        organ_indices = [
            index_mapping_dict[organ_index]
            if organ_index in index_mapping_dict
            else organ_index
            for organ_index in organ_indices
        ]
        organ_names = [
            name_mapping_dict[organ_name]
            if organ_name in name_mapping_dict
            else organ_name
            for organ_name in organ_names
        ]
        sample["organ_indices"] = list(set(organ_indices))
        sample["organ_names"] = list(set(organ_names))

        occ_organ_indices = sample["occ_organ_indices"]
        occ_organ_names = sample["occ_organ_names"]
        occ_organ_indices = [
            index_mapping_dict[organ_index]
            if organ_index in index_mapping_dict
            else organ_index
            for organ_index in occ_organ_indices
        ]
        occ_organ_names = [
            name_mapping_dict[organ_name]
            if organ_name in name_mapping_dict
            else organ_name
            for organ_name in occ_organ_names
        ]
        sample["occ_organ_indices"] = list(set(occ_organ_indices))
        sample["occ_organ_names"] = list(set(occ_organ_names))

    with open(dst_dset_path, "w") as outfile:
        json.dump(samples, outfile)


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Group organs into superorgans.")
    parser.add_argument("--src_dset_path", type=str, help="Path to the input dataset")
    parser.add_argument("--dst_dset_path", type=str, help="Path to the output dataset")
    parser.add_argument(
        "--src_dir", type=str, help="Path to the source data organs directory path."
    )
    parser.add_argument(
        "-og",
        "--organ_groups_names",
        type=str,
        nargs="+",
        action="append",
        help="List of lists - a list of organ groups that will be merged",
    )
    parser.add_argument(
        "-si",
        "--superorgan_indices",
        type=str,
        action="append",
        help="List of indices of superorgans that will be formed",
    )
    parser.add_argument(
        "-sn",
        "--superorgan_names",
        type=str,
        action="append",
        help="List of names of superorgans that will be formed",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    merge_dataset_groups(
        args.src_dset_path,
        args.dst_dset_path,
        args.src_dir,
        args.organ_groups_names,
        args.superorgan_indices,
        args.superorgan_names,
    )


if __name__ == "__main__":
    main()
