import subprocess

from utils.general import ensure_parent_dir


def extract_images(pdf_path: str, images_root: str = "", start_page: int = 2) -> None:
    if images_root:
        ensure_parent_dir(images_root)
    subprocess.run(
        ["pdfimages", "-p", "-f", "{}".format(start_page), pdf_path, images_root]
    )
