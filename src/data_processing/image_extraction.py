import os
from typing import Tuple, List, Union

import pdftotree
import subprocess
from math import floor, ceil
from wand.image import Image

from utils.general import ensure_parent_dir


def extract_images(pdf_path: str, images_root: str = "", start_page: int = 2) -> None:
    if images_root:
        ensure_parent_dir(images_root)
    subprocess.run(
        ["pdfimages", "-p", "-f", "{}".format(start_page), pdf_path, images_root]
    )


def get_pdf_tree(pdf_path: str, html_path: str) -> dict:
    return pdftotree.parse(
        pdf_path, html_path=html_path, model_type=None, model_path=None, favor_figures=True, visualize=False
    )


def get_figure_bbs(pdf_path: str, tree: dict = None, html_path: str = None) -> Tuple[List, List]:
    if not tree:
        if not html_path:
            html_path = './data_dump_'
        tree = pdftotree.parse(
            pdf_path,
            html_path=html_path,
            model_type=None,
            model_path=None,
            favor_figures=True,
            visualize=False,
        )
    captions = []
    figures = []

    for page_num, page in tree.items():
        for data_type, sections in page.items():
            i = 0
            for section in sections:
                pnum, pwidth, pheight, top, left, bottom, right = section
                if data_type == "figure":
                    figures.append(
                        (
                            pdf_path,
                            data_type,
                            pnum,
                            i,
                            pwidth,
                            pheight,
                            left,
                            top,
                            right,
                            bottom,
                        )
                    )
                elif data_type == "figure_caption":
                    captions.append(
                        (
                            pdf_path,
                            data_type,
                            pnum,
                            i,
                            pwidth,
                            pheight,
                            left,
                            top,
                            right,
                            bottom,
                        )
                    )
                i += 1
    return figures, captions


def get_bb_text(
    pdf_path: str,
    page_number: int,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> str:
    text = subprocess.check_output(
        [
            "pdftotext",
            "-f",
            "{}".format(page_number),
            "-l",
            "{}".format(page_number),
            "-x",
            "{}".format(int(left)),
            "-y",
            "{}".format(int(top)),
            "-W",
            "{}".format(int(right - left)),
            "-H",
            "{}".format(int(bottom - top)),
            pdf_path,
            "-",
        ]
    )

    return text.decode("utf-8")


def get_bb_img(
    pdf_path: str,
    base_path: str,
    page_number: int,
    index: int,
    pwidth: float,
    pheight: float,
    left: float,
    top: float,
    right: float,
    bottom: float,
    return_image=True,
    save_image=False,
) -> Union[None, Image]:
    img = Image(filename="{}[{}]".format(pdf_path, page_number - 1))
    img.resize(pwidth, pheight)
    img.crop(ceil(left), ceil(top), floor(right), floor(bottom))
    save_path = os.path.join(
        base_path,
        os.path.splittext(os.path.basename(pdf_path))[0],
        page_number + "_" + index + "_image.png",
    )
    if save_image:
        img.save(filename=save_path)
    if return_image:
        return img
    else:
        return None
