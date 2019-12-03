import os
import argparse
from tqdm import tqdm
from typing import List
from data_processing_matching.image_caption_pairs import (
    get_bb_img,
    get_bb_text,
    get_figure_bbs,
)
from utils.general import get_doc_filenames, ensure_dir, ensure_parent_dir
from utils.constants import trees_data_dir, image_data_dir, json_data_dir
from utils.loadsave import load_json, store_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf-dir", "-pd", type=str, help="specify folder with pdf files"
    )

    return parser.parse_args()


def group_figures_and_captions(bboxes_fig: List, bboxes_capt: List) -> List:
    pairs = []
    for bbox_fig in bboxes_fig:
        pdf_path_fig, data_type_fig, pnum_fig, figure_ind, _, _, _, _, _, _ = bbox_fig
        for bbox_capt in bboxes_capt:
            pdf_path_capt, data_type_capt, pnum_capt, caption_ind, _, _, _, _, _, _ = (
                bbox_capt
            )
            if (
                pdf_path_fig == pdf_path_capt
                and pnum_fig == pnum_capt
                and figure_ind == caption_ind
            ):
                pairs.append((bbox_fig, bbox_capt))
    return pairs


def main():
    args = parse_args()
    dirname = args.pdf_dir
    tree_root = os.path.join(trees_data_dir, os.path.basename(dirname))
    img_root = os.path.join(image_data_dir, os.path.basename(dirname))
    json_root = os.path.join(json_data_dir, os.path.basename(dirname))
    ensure_dir(tree_root)
    ensure_dir(img_root)
    ensure_dir(json_root)

    pdf_files = get_doc_filenames(dirname, extension=".pdf")

    for pdf_path in tqdm(pdf_files):
        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        tree_path = os.path.join(tree_root, filename + ".json")
        tree = None
        if os.path.exists(tree_path):
            tree = load_json(tree_path)
        if tree:
            bboxes_fig, bboxes_capt = get_figure_bbs(pdf_path=pdf_path, tree=tree)
            pairs = group_figures_and_captions(bboxes_fig, bboxes_capt)

            for pair in pairs:
                bbox_fig = pair[0]
                bbox_capt = pair[1]

                pdf_path, data_type, pnum, ind, pwidth, pheight, left, top, right, bottom = (
                    bbox_fig
                )
                try:
                    img = get_bb_img(
                        pdf_path=pdf_path,
                        page_number=pnum,
                        left=left,
                        top=top,
                        right=right,
                        bottom=bottom,
                    )
                    img_path = os.path.join(
                        img_root, filename, "{}_{}.png".format(pnum, ind)
                    )
                    ensure_parent_dir(img_path)
                    img.save(filename=img_path)
                except ValueError:
                    continue

                pdf_path, data_type, pnum, ind, pwidth, pheight, left, top, right, bottom = (
                    bbox_capt
                )
                text = get_bb_text(
                    pdf_path=pdf_path,
                    page_number=pnum,
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                )
                json_path = os.path.join(
                    json_root, filename, "{}_{}.json".format(pnum, ind)
                )
                ensure_parent_dir(json_path)
                json_data = {
                    "pdf": pdf_path,
                    "page": pnum,
                    "caption": text,
                    "figure": img_path,
                }
                store_json(json_data, json_path)


if __name__ == "__main__":
    main()
