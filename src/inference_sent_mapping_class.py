import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import json

from voxel_mapping.datasets import (
    VoxelSentenceMappingTestMaskedClassDataset,
    collate_pad_sentence_class_batch,
)
from voxel_mapping.models import SentenceMappingsProducer


def inference(
    ind2organ_path: str,
    test_json_path: str,
    batch_size: int,
    bert_path_or_name: str,
    checkpoint_path: str,
    joint_space: int,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load organ to indices to obtain the number of classes
    num_classes = len([index for index in json.load(open(ind2organ_path)).keys()])
    test_dataset = VoxelSentenceMappingTestMaskedClassDataset(
        test_json_path, bert_path_or_name, num_classes
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_sentence_class_batch,
    )
    model = nn.DataParallel(
        SentenceMappingsProducer(
            bert_path_or_name,
            joint_space,
            finetune=True,
            reg_or_class="class",
            num_classes=num_classes,
        )
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # Set model in evaluation mode
    model.train(False)
    with torch.no_grad():
        corrects = 0
        totals = 0
        cur_unmasked_acc = 0
        for sentences, organ_indices in tqdm(test_loader):
            sentences = sentences.to(device)
            output_mappings = model(sentences)
            y_pred = torch.argmax(output_mappings, dim=1)
            y_one_hot = torch.zeros(organ_indices.size()[0], num_classes)
            y_one_hot[torch.arange(organ_indices.size()[0]), y_pred] = 1
            y_one_hot[torch.where(y_one_hot == 0)] = -100
            corrects += (y_one_hot == organ_indices).sum(dim=1).sum().item()
            totals += organ_indices.size()[0]

        cur_unmasked_acc = corrects * 100 / totals
        print(f"The accuracy on the non masked validation set is {cur_unmasked_acc}")
        corrects = 0
        totals = 0
        cur_masked_acc = 0
        for sentences, organ_indices in tqdm(test_loader):
            sentences = sentences.to(device)
            output_mappings = model(sentences)
            y_pred = torch.argmax(output_mappings, dim=1)
            y_one_hot = torch.zeros(organ_indices.size()[0], num_classes)
            y_one_hot[torch.arange(organ_indices.size()[0]), y_pred] = 1
            y_one_hot[torch.where(y_one_hot == 0)] = -100
            corrects += (y_one_hot == organ_indices).sum(dim=1).sum().item()
            totals += organ_indices.size()[0]

        cur_masked_acc = corrects * 100 / totals

        print(f"The accuracy on the masked validation set is {cur_masked_acc}")


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    inference(
        args.ind2organ_path,
        args.test_json_path,
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
    parser = argparse.ArgumentParser(
        description="Inference with a sentence voxel mapping model."
    )
    parser.add_argument(
        "--ind2organ_path",
        type=str,
        default="data/data_organs/ind2organ.json",
        help="Path to the ind2organ path.",
    )
    parser.add_argument(
        "--test_json_path",
        type=str,
        default="data/test_dataset.json",
        help="Path to the test set",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="The size of the batch."
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
