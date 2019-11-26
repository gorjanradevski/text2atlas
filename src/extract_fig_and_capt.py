import os
import argparse
from data_processing.image_extraction import get_bb_img, get_bb_text, get_figure_bbs
from utils.general import get_doc_filenames, ensure_dir
from utils.constants import trees_data_dir, image_data_dir
from utils.loadsave import load_json

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
    img_root = os.path.join(image_data_dir, os.path.basename(dirname))
    ensure_dir(tree_root)

    pdf_files = get_doc_filenames(dirname, extension=".pdf")

    for pdf_path in tqdm(pdf_files):
        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        tree_path = os.path.join(tree_root, filename + ".json")
        tree = load_json(tree_path)
        bbxes_fig, bbxes_capt = get_figure_bbs(pdf_path=pdf_path, tree=tree)



if __name__ == '__main__':
    main()