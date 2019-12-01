import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from copy import deepcopy

from voxel_mapping.datasets import VoxelImageMappingTestDataset, collate_pad_image_batch
from voxel_mapping.models import ImageMappingsProducer
from voxel_mapping.evaluator import bbox_inside


def inference(
    test_json_path: str,
    ind2organ_path: str,
    organ2center_path: str,
    organ2bbox_path: str,
    batch_size: int,
    checkpoint_path: str,
    joint_space: int,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = VoxelImageMappingTestDataset(
        test_json_path, ind2organ_path, organ2center_path, organ2bbox_path
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_image_batch,
    )
    model = nn.DataParallel(ImageMappingsProducer(joint_space, finetune=False)).to(
        device
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # Set model in evaluation mode
    model.train(False)
    with torch.no_grad():
        # Restart counters
        total = 0
        correct = 0
        for images, _, _, bounding_boxes in tqdm(test_loader):
            images = images.to(device)
            output_mappings = model(images).cpu().numpy()
            # https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
            bounding_boxes_copy = deepcopy(bounding_boxes)
            del bounding_boxes
            del images
            for output_mapping, bounding_box in zip(
                output_mappings, bounding_boxes_copy
            ):
                total += 1
                correct += bbox_inside(output_mapping, bounding_box.numpy())

        print(f"The accuracy on the test set is: {(correct / total)}")


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    inference(
        args.test_json_path,
        args.ind2organ_path,
        args.organ2center_path,
        args.organ2bbox_path,
        args.batch_size,
        args.checkpoint_path,
        args.joint_space,
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Trains an image voxel mapping model.")
    parser.add_argument(
        "--test_json_path",
        type=str,
        default="data/test_dataset.json",
        help="Path to the test set",
    )
    parser.add_argument(
        "--ind2organ_path",
        type=str,
        default="data/data_organs/ind2organ.json",
        help="Path to the ind2organ json file.",
    )
    parser.add_argument(
        "--organ2center_path",
        type=str,
        default="data/data_organs/organ2center.json",
        help="Path to the organ2center json file.",
    )
    parser.add_argument(
        "--organ2bbox_path",
        type=str,
        default="data/data_organs/organ2bbox.json",
        help="Path to the organ2bbow json file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models/image_finetuned.pt",
        help="Path to a checkpoint.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The size of the batch."
    )
    parser.add_argument(
        "--joint_space",
        type=int,
        default=512,
        help="The joint space where the encodings will be projected.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
