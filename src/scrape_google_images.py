from google_images_download import google_images_download  # importing the library
import json
import os
import argparse
import time


def download_images(
    aliases_path: str,
    keyword_suffix: str,
    images_per_keyword: int,
    output_directory: str,
):
    response = google_images_download.googleimagesdownload()  # class instantiation
    json_file = json.load(open(aliases_path))

    keywords = []
    directory_names = {}
    for element in json_file:
        aliases = json_file[element]
        for alias in aliases:
            alias = alias.replace(",", "")
            keywords.append(alias)
            directory_names[alias + " " + keyword_suffix] = alias

    keywords_joined = ",".join(keywords)

    arguments = {
        "keywords": keywords_joined,
        "limit": images_per_keyword,
        "print_urls": True,
        "output_directory": output_directory,
        "color_type": "full-color",
        "format": "jpg",
        "suffix_keywords": keyword_suffix,
    }  # creating list of arguments

    response.download(arguments)  # passing the arguments to the function

    time.sleep(3)

    for directory in os.listdir(output_directory):
        if not os.path.isdir(os.path.join(output_directory, directory)):
            continue
        print(
            f"Renaming: {output_directory}/{directory} ==> {output_directory}/{directory_names[directory]}"
        )
        os.rename(
            os.path.join(output_directory, directory),
            os.path.join(output_directory, directory_names[directory]),
        )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Downloads images from google.")
    parser.add_argument(
        "--aliases_path",
        type=str,
        default="data/data_organs/organ2alias.json",
        help="Path to the aliases json file.",
    )
    parser.add_argument(
        "--keyword_suffix",
        type=str,
        default="laparoscopic",
        help="The suffix to be added to the keywords",
    )
    parser.add_argument(
        "--images_per_keyword",
        type=int,
        default=5,
        help="The number of images per keyword.",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="data/scraped_images",
        help="The name or path to a pretrained bert model.",
    )

    return parser.parse_args()


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    download_images(
        args.aliases_path,
        args.keyword_suffix,
        args.images_per_keyword,
        args.output_directory,
    )


if __name__ == "__main__":
    main()
