import argparse
import os
from data_processing_matching.pdf_tree_extraction import get_pdf_tree
from utils.general import get_doc_filenames, ensure_dir
from utils.constants import trees_data_dir, html_data_dir
from utils.loadsave import store_json

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf-dir",
        "-pd",
        type=str,
        help="specify folder with pdf files",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    dirname = args.pdf_dir
    tree_root = os.path.join(trees_data_dir, os.path.basename(dirname))
    ensure_dir(tree_root)

    pdf_files = get_doc_filenames(dirname, extension=".pdf")

    for pdf_path in tqdm(pdf_files):
        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        html_path = os.path.join(html_data_dir, os.path.basename(dirname), filename)
        tree_path = os.path.join(tree_root, filename + ".json")
        if not os.path.exists(tree_path):
            tree = get_pdf_tree(pdf_path, html_path)
            store_json(tree, tree_path)


if __name__ == '__main__':
    main()


