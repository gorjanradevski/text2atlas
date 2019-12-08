from google_images_download import google_images_download  # importing the library
import argparse
import time


def query_image_download(query: str, images_per_query: int, output_directory: str):
    response = google_images_download.googleimagesdownload()  # class instantiation

    arguments = {
        "keywords": query,
        "limit": images_per_query,
        "print_urls": True,
        "output_directory": output_directory,
        "color_type": "full-color",
        "format": "jpg",
    }  # creating list of arguments

    response.download(arguments)  # passing the arguments to the function

    time.sleep(3)


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Downloads images from google.")
    parser.add_argument("--query", type=str, help="Query term.")
    parser.add_argument(
        "--images-per-query",
        type=int,
        default=2,
        help="The number of images per query.",
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        default="data/scraped_images",
        help="Path to store images.",
    )

    return parser.parse_args()


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    query_image_download(args.query, args.images_per_query, args.output_directory)


if __name__ == "__main__":
    main()
