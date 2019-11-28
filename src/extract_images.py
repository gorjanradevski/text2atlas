import os
import natsort
from data_processing.image_extraction import extract_images
from utils.constants import pdf_data_dir, image_data_dir
from utils.general import get_doc_filenames


if __name__ == "__main__":

    dirnames = natsort.natsorted(
        [os.path.join(pdf_data_dir, dirname) for dirname in os.listdir(pdf_data_dir)]
    )

    for dirname in dirnames[-18:-14]:
        pdf_files = get_doc_filenames(dirname, extension=".pdf")
        for pdf_path in pdf_files:
            filename = os.path.splitext(os.path.basename(pdf_path))[0]
            images_root = os.path.join(image_data_dir, os.path.basename(dirname), filename, 'image')
            if not os.path.exists(os.path.dirname(images_root)):
                extract_images(pdf_path, images_root)
