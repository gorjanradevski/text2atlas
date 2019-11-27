from typing import Tuple, List

import pdftotree
import subprocess

from math import floor, ceil
from wand.image import Image


def get_figure_bbs(
    pdf_path: str, tree: dict = None, html_path: str = None
) -> Tuple[List, List]:
    if not tree:
        if not html_path:
            html_path = "./data_dump_"
        tree = pdftotree.parse(
            pdf_path,
            html_path=html_path,
            model_type=None,
            model_path=None,
            favor_figures=True
        )
    captions = []
    figures = []

    for page_num, page in tree.items():
        for data_type, sections in page.items():
            figure_ind = 0
            caption_ind = 0
            for section in sections:
                pnum, pwidth, pheight, top, left, bottom, right = section
                if data_type == "figure":
                    figures.append(
                        (
                            pdf_path,
                            data_type,
                            pnum,
                            figure_ind,
                            pwidth,
                            pheight,
                            left,
                            top,
                            right,
                            bottom,
                        )
                    )
                    figure_ind += 1
                elif data_type == "figure_caption":
                    captions.append(
                        (
                            pdf_path,
                            data_type,
                            pnum,
                            caption_ind,
                            pwidth,
                            pheight,
                            left,
                            top,
                            right,
                            bottom,
                        )
                    )
                    caption_ind += 1

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
    page_number: int,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> Image:
    img = Image(filename="{}[{}]".format(pdf_path, page_number - 1))
    img.crop(ceil(left), ceil(top), floor(right), floor(bottom))
    return img
