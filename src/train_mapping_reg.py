import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import json
from typing import Dict
import numpy as np
import random
import os
import sys
import logging
from transformers import BertConfig, BertTokenizer

from voxel_mapping.datasets import (
    VoxelSentenceMappingTrainRegDataset,
    VoxelSentenceMappingTestRegDataset,
    VoxelSentenceMappingTestMaskedRegDataset,
    collate_pad_sentence_reg_train_batch,
    collate_pad_sentence_reg_test_batch,
)
from voxel_mapping.models import SentenceMappingsProducer
from voxel_mapping.losses import MinDistanceLoss, OrganDistanceLoss, BaselineRegLoss
from voxel_mapping.evaluator import TrainingRegEvaluator
from utils.constants import bert_variants


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


def train(
    organs_dir_path: str,
    voxelman_images_path: str,
    train_json_path: str,
    val_json_path: str,
    loss_type: str,
    epochs: int,
    batch_size: int,
    bert_name: str,
    weight_decay: float,
    checkpoint_path: str,
    save_model_path: str,
    save_intermediate_model_path: str,
    log_filepath: str,
    learning_rate: float,
    clip_val: float,
):
    # Set up logging
    if log_filepath:
        logging.basicConfig(level=logging.INFO, filename=log_filepath, filemode="w")
    else:
        logging.basicConfig(level=logging.INFO)
    # Check whether bert_name is valid
    assert bert_name in bert_variants
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare paths
    organ2voxels_path = os.path.join(organs_dir_path, "organ2voxels_eroded.json")
    organ2ind_path = os.path.join(organs_dir_path, "organ2ind.json")
    ind2organ_path = os.path.join(organs_dir_path, "ind2organ.json")
    organ2label_path = os.path.join(organs_dir_path, "organ2label.json")
    organ2summary_path = os.path.join(organs_dir_path, "organ2summary.json")
    # Check for the type of loss
    ind2anchors = None
    assert loss_type in ["one_voxel", "all_voxels", "baseline"]
    if loss_type == "all_voxels":
        ind2anchors = create_ind2anchors(organ2ind_path, organ2voxels_path, 1000)
        criterion = OrganDistanceLoss()
        logging.warning("Using all organ points!")
    elif loss_type == "one_voxel":
        logging.warning("Using only one organ center!")
        criterion = MinDistanceLoss()
    else:
        ind2anchors = create_ind2anchors(organ2ind_path, organ2voxels_path, 1000)
        criterion = BaselineRegLoss()
        logging.warning("Training using the baseline regression loss!")

    tokenizer = BertTokenizer.from_pretrained(bert_name)
    organ_names = [organ_name for organ_name in json.load(open(organ2ind_path)).keys()]
    train_dataset = VoxelSentenceMappingTrainRegDataset(
        train_json_path, tokenizer, organ_names, ind2anchors
    )
    val_dataset = VoxelSentenceMappingTestRegDataset(
        val_json_path, tokenizer, ind2anchors
    )

    val_masked_dataset = VoxelSentenceMappingTestMaskedRegDataset(
        val_json_path, tokenizer, ind2anchors
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_pad_sentence_reg_train_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_sentence_reg_test_batch,
    )
    val_masked_loader = DataLoader(
        val_masked_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_sentence_reg_test_batch,
    )
    config = BertConfig.from_pretrained(bert_name)
    model = nn.DataParallel(
        SentenceMappingsProducer(bert_name, config, final_project_size=3)
    ).to(device)
    # noinspection PyUnresolvedReferences
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Load model
    cur_epoch = 0
    best_avg_distance = sys.maxsize
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        cur_epoch = checkpoint["epoch"]
        best_avg_distance = checkpoint["best_distance"]
        # https://discuss.pytorch.org/t/cuda-out-of-memory-after-loading-model/50681
        del checkpoint
        logging.warning(
            f"Starting training from checkpoint {checkpoint_path} with starting epoch {cur_epoch}!"
        )
        logging.warning(f"The previous best distance was: {best_avg_distance}!")

    # Create evaluator
    evaluator = TrainingRegEvaluator(
        ind2organ_path,
        organ2label_path,
        organ2summary_path,
        voxelman_images_path,
        len(val_dataset),
        best_avg_distance,
    )

    for epoch in range(cur_epoch, cur_epoch + epochs):
        logging.info(f"Starting epoch {epoch + 1}...")
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for sentences, attn_mask, true_mappings, num_organs in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                sentences, attn_mask, true_mappings, num_organs = (
                    sentences.to(device),
                    attn_mask.to(device),
                    true_mappings.to(device),
                    num_organs.to(device),
                )
                output_mappings = model(input_ids=sentences, attention_mask=attn_mask)
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
        # Reset current average distance
        evaluator.reset_current_average_distance()
        with torch.no_grad():
            # Restart counters
            evaluator.reset_counters()
            for sentences, attn_mask, organs_indices in tqdm(val_loader):
                sentences, attn_mask = sentences.to(device), attn_mask.to(device)
                output_mappings = (
                    model(input_ids=sentences, attention_mask=attn_mask).cpu().numpy()
                )
                for output_mapping, organ_indices in zip(
                    output_mappings, organs_indices
                ):
                    evaluator.update_counters(output_mapping, organ_indices.numpy())

            logging.info(
                f"The IOR on the non-masked validation set is {evaluator.get_current_ior()}"
            )
            logging.info(
                f"The distance on the non-masked validation set is {evaluator.get_current_distance()}"
            )
            logging.info(
                f"The miss distance on the non-masked validation set is {evaluator.get_current_miss_distance()}"
            )

            evaluator.update_current_average_distance()
            # Restart counters
            evaluator.reset_counters()
            for sentences, attn_mask, organs_indices in tqdm(val_masked_loader):
                sentences, attn_mask = sentences.to(device), attn_mask.to(device)
                output_mappings = (
                    model(input_ids=sentences, attention_mask=attn_mask).cpu().numpy()
                )
                for output_mapping, organ_indices in zip(
                    output_mappings, organs_indices
                ):
                    evaluator.update_counters(output_mapping, organ_indices.numpy())

            logging.info(
                f"The IOR on the masked validation set is {evaluator.get_current_ior()}"
            )
            logging.info(
                f"The distance on the masked validation set is {evaluator.get_current_distance()}"
            )
            logging.info(
                f"The miss distance on the masked validation set is {evaluator.get_current_miss_distance()}"
            )

            evaluator.update_current_average_distance()
            evaluator.finalize_current_average_distance()

            if evaluator.is_best_avg_distance():
                evaluator.update_best_avg_distance()
                logging.info("======================")
                logging.info(
                    f"Found new best with avg distance: "
                    f"{evaluator.best_avg_distance} on epoch "
                    f"{epoch+1}. Saving model!!!"
                )
                logging.info("======================")
                # torch.save(model.state_dict(), save_model_path)
            else:
                logging.info(
                    f"Avg distance on epoch {epoch+1} is: "
                    f"{evaluator.current_average_distance}"
                )
            logging.info("Saving intermediate checkpoint...")
            # torch.save(
            #     {
            #         "epoch": epoch + 1,
            #         "model_state_dict": model.state_dict(),
            #         "optimizer_state_dict": optimizer.state_dict(),
            #         "best_distance": evaluator.best_avg_distance,
            #     },
            #     save_intermediate_model_path,
            # )


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    train(
        args.organs_dir_path,
        args.voxelman_images_path,
        args.train_json_path,
        args.val_json_path,
        args.loss_type,
        args.epochs,
        args.batch_size,
        args.bert_name,
        args.weight_decay,
        args.checkpoint_path,
        args.save_model_path,
        args.save_intermediate_model_path,
        args.log_filepath,
        args.learning_rate,
        args.clip_val,
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Trains atlas reg mapping model.")
    parser.add_argument(
        "--organs_dir_path",
        type=str,
        default="data/data_organs",
        help="Path to the data organs directory path.",
    )
    parser.add_argument(
        "--voxelman_images_path",
        type=str,
        default="data/voxelman_images",
        help="Path to the voxel-man images",
    )
    parser.add_argument(
        "--train_json_path",
        type=str,
        default="data/cord_dataset_train.json",
        help="Path to the training set",
    )
    parser.add_argument(
        "--val_json_path",
        type=str,
        default="data/cord_dataset_val.json",
        help="Path to the validation set",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="models/sentence_mapping_regressor.pt",
        help="Where to save the model.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="all_voxels",
        help="The type of loss to use to train the model",
    )
    parser.add_argument(
        "--save_intermediate_model_path",
        type=str,
        default="models/intermediate_sentence_mapping_regressor.pt",
        help="Where to save the intermediate checkpoint model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="The number of epochs to train the model.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="The weight decay - default as per BERT.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="The size of the batch."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="The learning rate."
    )
    parser.add_argument(
        "--clip_val", type=float, default=2.0, help="The clipping threshold."
    )
    parser.add_argument(
        "--bert_name",
        type=str,
        default="bert-base-uncased",
        help="Should be one of [bert-base-uncased, allenai/scibert_scivocab_uncased,"
        "monologg/biobert_v1.1_pubmed, emilyalsentzer/Bio_ClinicalBERT,"
        "google/bert_uncased_L-4_H-512_A-8]",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="If resuming training, start from here.",
    )
    parser.add_argument(
        "--log_filepath", type=str, default=None, help="The logging file."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
