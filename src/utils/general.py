import os
import sys
import natsort
from typing import List


def eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)
    return


def flatten_list(lst: List[List]) -> List:
    """
    :param lst: list containing sublists
    :return: flattened list
    """
    return [item for sublist in lst for item in sublist]


def get_doc_filenames(folder: str, extension: str = ".txt") -> list:
    """Return file names of text files inside a folder.

    Args:
        folder: str - folder name
        :param extension: str - file extension
    """
    extensions = (".vtt", ".srt", ".txt", ".pdf", ".json", ".ppm", ".pbm")
    assert extension in extensions, "Wrong extension provided, expected one of {}".format(extensions)
    return natsort.natsorted(
        [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and f.endswith(extension)
        ]
    )


def ensure_dir(path: str):
    """
    Ensure that a directory exists.
    Args:
        path (str): Directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_parent_dir(path: str):
    """
    Ensure that a parent directory of a file exists.
    Args:
        path (str): File path.
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))