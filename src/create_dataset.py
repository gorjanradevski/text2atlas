import os
import argparse
from utils.constants import dataset_dir, json_data_dir
from utils.loadsave import aggregate_jsons


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        "-d",
        type=str,
        default=os.path.join(dataset_dir, "dataset.json"),
        help="specify folder to save dataset in",
    )
    parser.add_argument(
        "--json-root-dir",
        "-j",
        type=str,
        default=json_data_dir,
        help="specify folder with json files (can be in subdirectories as well)"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    aggregate_jsons(root_path=args.json_root_dir, path_out=args.dataset_path)


if __name__ == '__main__':
    main()

