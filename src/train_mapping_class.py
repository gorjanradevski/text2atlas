import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import json
import sys
import os
import logging
import numpy as np
from transformers import BertConfig, BertTokenizer

from voxel_mapping.datasets import (
    VoxelSentenceMappingTrainClassDataset,
    VoxelSentenceMappingTestClassDataset,
    collate_pad_sentence_class_batch,
)
from voxel_mapping.models import ClassModel
from voxel_mapping.evaluator import TrainingEvaluator


def train(
    organs_dir_path: str,
    voxelman_images_path: str,
    train_json_path: str,
    val_json_path: str,
    epochs: int,
    batch_size: int,
    bert_name: str,
    checkpoint_path: str,
    save_model_path: str,
    save_intermediate_model_path: str,
    log_filepath: str,
    learning_rate: float,
    weight_decay: float,
    masking: bool,
    clip_val: float,
):
    # Set up logging
    if log_filepath:
        logging.basicConfig(level=logging.INFO, filename=log_filepath, filemode="w")
    else:
        logging.basicConfig(level=logging.INFO)
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare jsons
    ind2organ = json.load(open(os.path.join(organs_dir_path, "ind2organ.json")))
    organ2center = json.load(open(os.path.join(organs_dir_path, "organ2center.json")))
    organ2label = json.load(open(os.path.join(organs_dir_path, "organ2label.json")))
    organ2summary = json.load(open(os.path.join(organs_dir_path, "organ2summary.json")))
    num_classes = max([int(index) for index in ind2organ.keys()]) + 1
    # Prepare datasets
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    logging.warning(f"The masking is set to: ---{masking}---")
    train_dataset = VoxelSentenceMappingTrainClassDataset(
        train_json_path, tokenizer, num_classes, masking
    )
    val_dataset = VoxelSentenceMappingTestClassDataset(
        val_json_path, tokenizer, num_classes
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_pad_sentence_class_batch,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_pad_sentence_class_batch,
    )
    config = BertConfig.from_pretrained(bert_name)
    # Prepare model
    model = nn.DataParallel(
        ClassModel(bert_name, config, final_project_size=num_classes)
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    # noinspection PyUnresolvedReferences
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_distance = sys.maxsize
    cur_epoch = 0
    # Load model
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        cur_epoch = checkpoint["epoch"]
        best_distance = checkpoint["best_distance"]
        # https://discuss.pytorch.org/t/cuda-out-of-memory-after-loading-model/50681
        del checkpoint
        logging.warning(
            f"Starting training from checkpoint {checkpoint_path} with starting epoch {cur_epoch}!"
        )
        logging.warning(f"The previous best distance was: {best_distance}!")

    # Prepare evaluator
    evaluator = TrainingEvaluator(
        ind2organ,
        organ2label,
        organ2summary,
        voxelman_images_path,
        len(val_dataset),
        best_distance,
    )
    for epoch in range(cur_epoch, cur_epoch + epochs):
        logging.info(f"Starting epoch {epoch + 1}...")
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for sentences, attn_mask, organ_indices in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                sentences, attn_mask, organ_indices = (
                    sentences.to(device),
                    attn_mask.to(device),
                    organ_indices.to(device),
                )
                output_mappings = model(input_ids=sentences, attention_mask=attn_mask)
                loss = criterion(output_mappings, organ_indices)
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
        evaluator.reset_current_distance()
        with torch.no_grad():
            evaluator.reset_counters()
            for sentences, attn_mask, organs_indices in tqdm(val_loader):
                sentences, attn_mask = sentences.to(device), attn_mask.to(device)
                output_mappings = model(input_ids=sentences, attention_mask=attn_mask)
                y_pred = torch.argmax(output_mappings, dim=-1)
                pred_organ_names = [ind2organ[str(ind.item())] for ind in y_pred]
                pred_centers = [
                    organ2center[organ_name] for organ_name in pred_organ_names
                ]
                for pred_center, organ_indices in zip(pred_centers, organs_indices):
                    evaluator.update_counters(
                        np.array(pred_center), np.where(organ_indices == 1)[0]
                    )

            logging.info(
                f"The IOR on the validation set is {evaluator.get_current_ior()}"
            )
            logging.info(
                f"The distance on the validation set is {evaluator.get_current_distance()}"
            )
            logging.info(
                f"The miss distance on the validation set is {evaluator.get_current_miss_distance()}"
            )

            evaluator.update_current_distance()

            if evaluator.is_best_distance():
                evaluator.update_best_distance()
                logging.info("======================")
                logging.info(
                    f"Found new best with distance: "
                    f"{evaluator.best_distance} on epoch "
                    f"{epoch+1}. Saving model!!!"
                )
                logging.info("======================")
                torch.save(model.state_dict(), save_model_path)
            else:
                logging.info(
                    f"Avg distance on epoch {epoch+1} is: "
                    f"{evaluator.current_distance}"
                )
            logging.info("Saving intermediate checkpoint...")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_distance": evaluator.best_distance,
                },
                save_intermediate_model_path,
            )


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    train(
        args.organs_dir_path,
        args.voxelman_images_path,
        args.train_json_path,
        args.val_json_path,
        args.epochs,
        args.batch_size,
        args.bert_name,
        args.checkpoint_path,
        args.save_model_path,
        args.save_intermediate_model_path,
        args.log_filepath,
        args.learning_rate,
        args.weight_decay,
        args.masking,
        args.clip_val,
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Trains atlas class mapping model.")
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
        default="models/sentence_mapping_classifier.pt",
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
    parser.add_argument("--masking", action="store_true", help="Whether to use masking")
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
        "--weight_decay", type=float, default=0.01, help="The (default) weight decay."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a pretrained checkpoint.",
    )
    parser.add_argument(
        "--save_intermediate_model_path",
        type=str,
        default="models/intermediate_sentence_mapping_classifier.pt",
        help="Where to save the intermediate checkpoint model.",
    )
    parser.add_argument(
        "--log_filepath", type=str, default=None, help="The logging file."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
