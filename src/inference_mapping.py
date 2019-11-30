import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from copy import deepcopy

from voxel_mapping.datasets import (
    VoxelSentenceMappingTestDataset,
    VoxelSentenceMappingTestMaskedDataset,
    collate_pad_batch,
)
from voxel_mapping.models import SentenceMappingsProducer
from voxel_mapping.evaluator import bbox_inside


def inference(
    val_json_path: str,
    batch_size: int,
    bert_path_or_name: str,
    checkpoint_path: str,
    joint_space: int,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = VoxelSentenceMappingTestDataset(val_json_path, bert_path_or_name)
    val_masked_dataset = VoxelSentenceMappingTestMaskedDataset(
        val_json_path, bert_path_or_name
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4, collate_fn=collate_pad_batch
    )
    val_masked_loader = DataLoader(
        val_masked_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_batch,
    )
    model = nn.DataParallel(
        SentenceMappingsProducer(bert_path_or_name, joint_space, finetune=True)
    ).to(device)
    # Load model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # Set model in evaluation mode
    model.train(False)
    with torch.no_grad():
        # Restart counters
        total = 0
        correct = 0
        for sentences, _, _, bounding_boxes in tqdm(val_loader):
            sentences = sentences.to(device)
            output_mappings = model(sentences).cpu().numpy()
            # https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
            bounding_boxes_copy = deepcopy(bounding_boxes)
            del bounding_boxes
            del sentences
            for output_mapping, bounding_box in zip(
                output_mappings, bounding_boxes_copy
            ):
                total += 1
                correct += bbox_inside(output_mapping, bounding_box.numpy())

        print(f"The accuracy on the non masked validation set is {correct/total}")
        # Restart counters
        total = 0
        correct = 0
        for sentences, _, _, bounding_boxes in tqdm(val_masked_loader):
            sentences = sentences.to(device)
            output_mappings = model(sentences).cpu().numpy()
            # https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
            bounding_boxes_copy = deepcopy(bounding_boxes)
            del bounding_boxes
            del sentences
            for output_mapping, bounding_box in zip(
                output_mappings, bounding_boxes_copy
            ):
                total += 1
                correct += bbox_inside(output_mapping, bounding_box.numpy())

        print(f"The accuracy on the masked validation set is {correct/total}")


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    inference(
        args.val_json_path,
        args.batch_size,
        args.bert_path_or_name,
        args.checkpoint_path,
        args.joint_space,
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Performs mapping inference.")
    parser.add_argument(
        "--val_json_path",
        type=str,
        default="data/val_dataset.json",
        help="Path to the validation set",
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
    parser.add_argument(
        "--bert_path_or_name",
        type=str,
        default="bert-base-uncased",
        help="The name or path to a pretrained bert model.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="pretrained.pt",
        help="Path to a pretrained checkpoint.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
