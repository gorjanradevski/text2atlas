import argparse
import os
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm


def delete_images(directory_path: str, threshold: float):
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        if not os.path.isdir(file_path):
            continue
        frames_path = os.path.join(file_path, "frames")
        frame_variances = []
        for frame in tqdm(os.listdir(frames_path)):
            if not frame.endswith(".jpg"):
                print(f"{frame} not an image")
            frame_path = os.path.join(frames_path, frame)
            image = np.array(Image.open(frame_path).convert("LA"))
            variance = cv2.Laplacian(image, cv2.CV_64F).var()
            if variance <= threshold:
                frame_variances.append((frame_path, variance))
        print(f"Will remove {len(frame_variances)} frames...")
        for frame_path, variance in frame_variances:
            os.remove(frame_path)


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    delete_images(args.directory_path, args.threshold)


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Deletes low variance images.")
    parser.add_argument(
        "--directory_path",
        type=str,
        default=".",
        help="Path to the directory with the organ files.",
    )
    parser.add_argument(
        "--threshold", type=float, default=1.5, help="The variance threshold."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
