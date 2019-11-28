import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from voxel_mapping.datasets import VoxelMappingDataset, collate_pad_batch
from voxel_mapping.models import MappingsProducer
from voxel_mapping.losses import MinDistanceLoss


def pretrain(
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
    train_dataset = VoxelMappingDataset(train_json_path)
    val_dataset = VoxelMappingDataset(val_json_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_pad_batch,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4, collate_fn=collate_pad_batch
    )
    model = nn.DataParallel(MappingsProducer(joint_space, finetune=False)).to(device)
    criterion = MinDistanceLoss()
    # noinspection PyUnresolvedReferences
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}...")
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for sentences, true_mappings, num_organs, _ in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                sentences, true_mappings = (
                    sentences.to(device),
                    true_mappings.to(device),
                )
                output_mappings = model(sentences)
                loss = criterion(output_mappings, true_mappings, num_organs)
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
            for sentences, true_mappings, num_organs, bounding_boxes in tqdm(
                val_loader
            ):
                sentences, true_mappings = (
                    sentences.to(device),
                    true_mappings.to(device),
                )
                output_mappings = model(sentences).cpu().numpy()
                # TODO: Model validation and saving


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    pretrain(
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
    parser = argparse.ArgumentParser(description="Trains an image-text matching model.")
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
        default="models/pretrained.pt",
        help="Where to save the model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="The number of epochs to train the model excluding the vgg.",
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
