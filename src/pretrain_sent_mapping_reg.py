import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from typing import Dict
import json
import numpy as np
import random

from voxel_mapping.datasets import (
    VoxelSentenceMappingTrainRegDataset,
    VoxelSentenceMappingTestRegDataset,
    VoxelSentenceMappingTestMaskedRegDataset,
    collate_pad_sentence_reg_batch,
)
from voxel_mapping.models import SentenceMappingsProducer
from voxel_mapping.losses import MinDistanceLoss, OrganDistanceLoss
from voxel_mapping.evaluator import TrainingRegEvaluator

torch.multiprocessing.set_sharing_strategy("file_system")
# https://github.com/pytorch/pytorch/issues/973#issuecomment-426559250


def create_ind2anchors(
    organ2ind_path: str, organ2voxels_path: str, num_anchors: int = 1000
) -> Dict:
    """CREATING A MAPPING FROM INDEX TO A SET OF ORGAN POINTS"""
    """RANDOM SAMPLING ORGAN VOXELS - DONE IN THE BEGINING OF EVERY EPOCH"""

    organ2ind = json.load(open(organ2ind_path))
    organ2voxels = json.load(open(organ2voxels_path))
    ind2voxels = {}

    for organ, ind in organ2ind.items():
        if len(organ2voxels[organ]) > num_anchors:
            ind2voxels[ind] = random.sample(organ2voxels[organ], num_anchors)
        else:
            ind2voxels[ind] = np.array(organ2voxels[organ])[
                np.random.choice(range(len(organ2voxels[organ])), num_anchors)
            ].tolist()

    ind2voxels[-1] = np.zeros(shape=(num_anchors, 3), dtype=np.float)

    return ind2voxels


def pretrain(
    organ2ind_path: str,
    organ2voxels_path: str,
    ind2organ_path: str,
    organ2label_path: str,
    voxelman_images_path: str,
    train_json_path: str,
    val_json_path: str,
    use_all_voxels: bool,
    epochs: int,
    batch_size: int,
    bert_path_or_name: str,
    mask_probability: float,
    save_model_path: str,
    learning_rate: float,
    weight_decay: float,
    clip_val: float,
    joint_space: int,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = VoxelSentenceMappingTrainRegDataset(
        train_json_path, bert_path_or_name, mask_probability
    )
    val_dataset = VoxelSentenceMappingTestRegDataset(val_json_path, bert_path_or_name)
    val_masked_dataset = VoxelSentenceMappingTestMaskedRegDataset(
        val_json_path, bert_path_or_name
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_pad_sentence_reg_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_sentence_reg_batch,
    )
    val_masked_loader = DataLoader(
        val_masked_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_sentence_reg_batch,
    )
    model = nn.DataParallel(
        SentenceMappingsProducer(bert_path_or_name, joint_space, finetune=False)
    ).to(device)
    if use_all_voxels:
        ind2anchors = create_ind2anchors(organ2ind_path, organ2voxels_path, 1000)
        criterion = OrganDistanceLoss()
        print("Using all organ points!")
    else:
        print("Using only one organ center!")
        criterion = MinDistanceLoss()
    # noinspection PyUnresolvedReferences
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # Create evaluator
    evaluator = TrainingRegEvaluator(
        ind2organ_path, organ2label_path, voxelman_images_path, len(val_dataset)
    )
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}...")
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for sentences, true_mappings, num_organs, true_labels in train_loader:
                if use_all_voxels:
                    anchors_list = [
                        ind2anchors[item.item()]
                        for true_label in true_labels
                        for item in true_label
                    ]
                    anchors = (
                        torch.tensor(anchors_list)
                        .view(true_labels.shape[0], true_labels.shape[1], 1000, 3)
                        .to(device)
                    )
                # remove past gradients
                optimizer.zero_grad()
                # forward
                sentences, true_mappings, num_organs = (
                    sentences.to(device),
                    true_mappings.to(device),
                    num_organs.to(device),
                )
                output_mappings = model(sentences)
                if use_all_voxels:
                    loss = criterion(output_mappings, anchors, num_organs, device)
                else:
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
        # Reset current average accuracy
        evaluator.reset_current_average_accuracy()
        with torch.no_grad():
            # Restart counters
            evaluator.reset_counters()
            for sentences, _, _, organs_indices in tqdm(val_loader):
                sentences = sentences.to(device)
                output_mappings = model(sentences).cpu().numpy()
                for output_mapping, organ_indices in zip(
                    output_mappings, organs_indices
                ):
                    evaluator.update_counters(output_mapping, organ_indices.numpy())

            print(
                f"The accuracy on the non masked validation set is {evaluator.get_current_accuracy()}"
            )
            evaluator.update_current_average_accuracy()
            # Restart counters
            evaluator.reset_counters()
            for sentences, _, _, organs_indices in tqdm(val_masked_loader):
                sentences = sentences.to(device)
                output_mappings = model(sentences).cpu().numpy()
                for output_mapping, organ_indices in zip(
                    output_mappings, organs_indices
                ):
                    evaluator.update_counters(output_mapping, organ_indices.numpy())

            print(
                f"The accuracy on the masked validation set is {evaluator.get_current_accuracy()}"
            )
            evaluator.update_current_average_accuracy()
            evaluator.finalize_current_average_accuracy()

            if evaluator.is_best_avg_accuracy():
                evaluator.update_best_avg_accuracy()
                print("======================")
                print(
                    f"Found new best with avg accuracy {evaluator.best_avg_accuracy} on epoch "
                    f"{epoch+1}. Saving model!!!"
                )
                torch.save(model.state_dict(), save_model_path)
                print("======================")
            else:
                print(
                    f"Avg accuracy on epoch {epoch+1} is: {evaluator.current_average_accuracy}"
                )


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    pretrain(
        args.organ2ind_path,
        args.organ2voxels_path,
        args.ind2organ_path,
        args.organ2label_path,
        args.voxelman_images_path,
        args.train_json_path,
        args.val_json_path,
        args.use_all_voxels,
        args.epochs,
        args.batch_size,
        args.bert_path_or_name,
        args.mask_probability,
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
    parser = argparse.ArgumentParser(
        description="Trains a sentence voxel mapping model."
    )
    parser.add_argument(
        "--organ2voxels_path",
        type=str,
        default="data/data_organs_new/organ2voxels_new.json",
        help="Path to the ind2organ path.",
    )
    parser.add_argument(
        "--organ2ind_path",
        type=str,
        default="data/data_organs_new/organ2ind_new.json",
        help="Path to the ind2organ path.",
    )
    parser.add_argument(
        "--use_all_voxels", action="store_true", help="Whether to use the all voxels."
    )
    parser.add_argument(
        "--ind2organ_path",
        type=str,
        default="data/data_organs_new/ind2organ_new.json",
        help="Path to the ind2organ path.",
    )
    parser.add_argument(
        "--organ2label_path",
        type=str,
        default="data/data_organs_new/organ2label_new.json",
        help="Path to the organ2label path.",
    )
    parser.add_argument(
        "--voxelman_images_path",
        type=str,
        default="data/data_organs_new/voxelman_images",
        help="Path to the voxel-man images",
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
        default="models/sentence_pretrained.pt",
        help="Where to save the model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="The number of epochs to train the model.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="The size of the batch."
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
    parser.add_argument(
        "--bert_path_or_name",
        type=str,
        default="bert-base-uncased",
        help="The name or path to a pretrained bert model.",
    )
    parser.add_argument(
        "--mask_probability", type=float, default=0.5, help="The mask probability."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
