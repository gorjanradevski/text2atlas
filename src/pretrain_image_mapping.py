import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from copy import deepcopy

from voxel_mapping.datasets import (
    VoxelImageMappingTrainDataset,
    VoxelImageMappingTestDataset,
    collate_pad_image_batch,
)
from voxel_mapping.models import ImageMappingsProducer
from voxel_mapping.losses import MinDistanceLoss
from voxel_mapping.evaluator import bbox_inside


def pretrain(
    images_location: str,
    train_json_path: str,
    val_json_path: str,
    epochs: int,
    batch_size: int,
    save_model_path: str,
    learning_rate: float,
    weight_decay: float,
    clip_val: float,
    joint_space: int,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = VoxelImageMappingTrainDataset(train_json_path, images_location)
    val_dataset = VoxelImageMappingTestDataset(val_json_path, images_location)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_pad_image_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_image_batch,
    )
    model = nn.DataParallel(ImageMappingsProducer(joint_space, finetune=False)).to(
        device
    )
    criterion = MinDistanceLoss()
    # noinspection PyUnresolvedReferences
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_accuracy = -1
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}...")
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for images, true_mappings, num_organs, _ in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                images, true_mappings, num_organs = (
                    images.to(device),
                    true_mappings.to(device),
                    num_organs.to(device),
                )
                output_mappings = model(images)
                loss = criterion(output_mappings, true_mappings, num_organs, device)
                # backward
                loss.backward()
                # clip the gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                # update weights
                optimizer.step()
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss.item()})

        # Set model in evaluation mode
        model.train(False)
        with torch.no_grad():
            # Restart counters
            total = 0
            correct = 0
            for images, _, _, bounding_boxes in tqdm(val_loader):
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

            if (correct / total) > best_accuracy:
                best_accuracy = correct / total
                print("======================")
                print(
                    f"Found new best with accuracy {best_accuracy} on epoch "
                    f"{epoch+1}. Saving model!!!"
                )
                torch.save(model.state_dict(), save_model_path)
                print("======================")
            else:
                print(f"Accuracy on epoch {epoch+1} is: {(correct / total)}")


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    pretrain(
        args.images_location,
        args.train_json_path,
        args.val_json_path,
        args.epochs,
        args.batch_size,
        args.save_model_path,
        args.learning_rate,
        args.weight_decay,
        args.clip_val,
        args.joint_space,
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Trains an image voxel mapping model.")
    parser.add_argument(
        "--images_location",
        type=str,
        default="data/videos",
        help="Path where all images are.",
    )
    parser.add_argument(
        "--train_json_path",
        type=str,
        default="data/train_dataset.json",
        help="Path to the training set",
    )
    parser.add_argument(
        "--val_json_path",
        type=str,
        default="data/val_dataset.json",
        help="Path to the validation set",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="models/image_pretrained.pt",
        help="Where to save the model.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="The number of epochs to train the model."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The size of the batch."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0002, help="The learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="The weight decay."
    )
    parser.add_argument(
        "--joint_space",
        type=int,
        default=512,
        help="The joint space where the encodings will be projected.",
    )
    parser.add_argument(
        "--clip_val", type=float, default=2.0, help="The clipping threshold."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
