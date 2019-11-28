import os
import argparse
from utils.constants import dataset_dir, json_data_dir
from utils.loadsave import aggregate_jsons, load_json, store_json
from data_processing.image_filtering import is_natural
from data_processing.text_filtering import process_caption


def parse_args():
    parser = argparse.ArgumentParser(description="Collects image-caption pairs from jsons in json root dir.")
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
    parser.add_argument(
        "--filt-nnat",
        "--fnn",
        type=bool,
        default=True,
        help="whether to filter out non-natural images (graphs, sketches), default True"
    )
    parser.add_argument(
        "--filt-no-title",
        "--fnt",
        type=bool,
        default=False,
        help="whether to filter out samples with no title, by default False"
    )
    parser.add_argument(
        "--filt-no-keywords",
        "--fnk",
        type=bool,
        default=False,
        help="whether to filter out samples with no keywords, by default False"
    )
    parser.add_argument(
        "--filt-no-abstract",
        "--fna",
        type=bool,
        default=False,
        help="whether to filter out samples with no abstract, by default False"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    root_path = args.json_root_dir
    path_out = args.dataset_path
    filt_nnat = args.filt_nnat
    filt_no_title = args.filt_no_title
    filt_no_keywords = args.filt_no_keywords
    filt_no_abstract = args.filt_no_abstract
    aggregate_jsons(root_path=root_path, path_out=path_out)

    data = load_json(path_out)

    if filt_nnat:
        data = [item for item in data if is_natural(item["figure"])]
    if filt_no_title:
        data = [item for item in data if item["title"]]
    if filt_no_keywords:
        data = [item for item in data if item["keywords"]]
    if filt_no_abstract:
        data = [item for item in data if item["abstract"]]

    for item in data:
        item["pdf"] = os.path.join('data', '/'.join(item["pdf"].split('/')[5:]))
        item["figure"] = os.path.join('data', '/'.join(item["figure"].split('/')[5:]))
        item["caption"] = process_caption(item["caption"])

    store_json(data, path_out)


if __name__ == '__main__':
    main()

