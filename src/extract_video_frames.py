import cv2
import os
from tqdm import tqdm
import argparse


def extract_single_video_frames(
    video_path: str, output_dir: str, every_n_frame: int = 15
):
    if os.path.exists(video_path):
        video_name = video_path.split(".")[0].split("/")[-1]
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        while success:
            if count % every_n_frame == 0:
                frame_save_path = os.path.join(
                    output_dir, f"{video_name}_frame_{count}.jpg"
                )
                cv2.imwrite(frame_save_path, image)  # save frame as JPEG file
            success, image = vidcap.read()
            count += 1
    else:
        raise ValueError("The path does not exists")


def extract_frames_for_organ(videos_folder: str, every_n_frame: int = 15):
    frames_dir = os.path.join(videos_folder, "frames")
    if not os.path.exists(frames_dir):
        os.mkdir(frames_dir)
    else:
        print(f"Frames directory exists for {videos_folder}! Skipping...")
        return
    for file in tqdm(os.listdir(videos_folder)):
        if file.endswith(".mp4"):
            video_file_path = os.path.join(videos_folder, file)
            extract_single_video_frames(video_file_path, frames_dir, every_n_frame)


def extract_frames_for_all_organs(organs_dir: str, every_n_frame: int = 15):
    for organ_dir in os.listdir(organs_dir):
        organ_dir_path = os.path.join(organs_dir, organ_dir)
        if os.path.isdir(organ_dir_path):
            print(f"Extracting for: {organ_dir_path}...")
            extract_frames_for_organ(organ_dir_path, every_n_frame)


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    extract_frames_for_all_organs(args.organs_dir, args.every_n_frame)


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Trains an image-text matching model.")
    parser.add_argument(
        "--organs_dir",
        type=str,
        default=".",
        help="Path to the directory with the organ files.",
    )
    parser.add_argument(
        "--every_n_frame",
        type=int,
        default=15,
        help="The number of frames per second (Assuming 30fps).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
