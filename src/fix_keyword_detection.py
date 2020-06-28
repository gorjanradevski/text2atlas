import argparse
import json
import os

from tqdm.notebook import tqdm


def fix_keyword_detection_issues(
    src_dset_path: str, dst_dset_path: str, organs_dir_path: str
):

    dset = json.load(open(src_dset_path))

    organ2ind = json.load(open(os.path.join(organs_dir_path, "organ2ind.json")))

    # # SOLVE CARDIA PROBLEM

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

    # # SOLVE THE LIVER - DELIVER PROBLEM

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

    # # SOLVE THE COLON - COLONISE PROBLEM

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

    # # SOLVE THE BLADDER - GALLBLADDER PROBLEM

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

    with open(dst_dset_path, "w") as outfile:
        json.dump(dset, outfile)


def parse_args():
    parser = argparse.ArgumentParser(description="Testing loss function")
    parser.add_argument(
        "--src_dset_path", type=str, help="Path to json file with dataset"
    )
    parser.add_argument(
        "--dst_dset_path", type=str, help="Path under which the fixed dataset is saved"
    )
    parser.add_argument(
        "--organs_dir_path",
        type=str,
        help="Path to the directory with organ dictionaries",
    )


def main():
    args = parse_args()
    fix_keyword_detection_issues(
        args.src_dset_path, args.dst_dset_path, args.organs_dir_path
    )


if __name__ == "__main__":
    main()
