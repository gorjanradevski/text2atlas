import argparse
import json
import logging
import os
import sys

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from utils.constants import VOXELMAN_CENTER
from voxel_mapping.datasets import (
    VoxelSentenceMappingTestRegDataset,
    VoxelSentenceMappingTrainRegDataset,
    collate_pad_sentence_reg_test_batch,
    collate_pad_sentence_reg_train_batch,
)
from voxel_mapping.evaluator import TrainingEvaluator
from voxel_mapping.losses import BaselineRegLoss, OrganDistanceLoss
from voxel_mapping.models import RegModel


def train(args):
    # Set up logging
    if args.log_filepath:
        logging.basicConfig(
            level=logging.INFO, filename=args.log_filepath, filemode="w"
        )
    else:
        logging.basicConfig(level=logging.INFO)
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare paths
    organ2voxels = json.load(
        open(os.path.join(args.organs_dir_path, "organ2voxels.json"))
    )
    ind2organ = json.load(open(os.path.join(args.organs_dir_path, "ind2organ.json")))
    organ2label = json.load(
        open(os.path.join(args.organs_dir_path, "organ2label.json"))
    )
    organ2summary = json.load(
        open(os.path.join(args.organs_dir_path, "organ2summary.json"))
    )
    # Prepare datasets
    tokenizer = BertTokenizer.from_pretrained(args.bert_name)
    logging.warning(f"Usage of masking is set to: ---{args.masking}---")
    logging.warning(f"Usage of occurences is set to: ---{args.use_occurrences}---")
    train_dataset = VoxelSentenceMappingTrainRegDataset(
        args.train_json_path,
        tokenizer,
        ind2organ,
        organ2voxels,
        num_anchors=args.num_anchors,
        masking=args.masking,
        use_occurences=args.use_occurrences,
    )
    val_dataset = VoxelSentenceMappingTestRegDataset(args.val_json_path, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pad_sentence_reg_train_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_pad_sentence_reg_test_batch,
    )
    config = BertConfig.from_pretrained(args.bert_name)
    # Prepare model
    model = nn.DataParallel(RegModel(args.bert_name, config, final_project_size=3)).to(
        device
    )
    logging.warning(f"Using {args.num_anchors} voxel points!")
    # Check for the type of loss
    if args.loss_type == "organ_loss":
        criterion = OrganDistanceLoss(
            device=device,
            voxel_temperature=args.voxel_temperature,
            organ_temperature=args.organ_temperature,
        )
        logging.warning("Using SSL loss!")
    elif args.loss_type == "baseline_loss":
        criterion = BaselineRegLoss()
        logging.warning("Using baseline REG loss!")
    else:
        raise ValueError(f"Invalid loss type {args.loss_type}")
    # noinspection PyUnresolvedReference
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # Load model
    cur_epoch = 0
    best_distance = sys.maxsize
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        cur_epoch = checkpoint["epoch"]
        best_distance = checkpoint["best_distance"]
        # https://discuss.pytorch.org/t/cuda-out-of-memory-after-loading-model/50681
        del checkpoint
        logging.warning(
            f"Starting training from checkpoint {args.checkpoint_path} with starting epoch {cur_epoch}!"
        )
        logging.warning(f"The previous best distance was: {best_distance}!")
    # Prepare evaluator
    evaluator = TrainingEvaluator(
        ind2organ,
        organ2label,
        organ2summary,
        args.voxelman_images_path,
        len(val_dataset),
        best_distance,
    )
    center = torch.from_numpy(VOXELMAN_CENTER)
    for epoch in range(cur_epoch, cur_epoch + args.epochs):
        logging.info(f"Starting epoch {epoch + 1}...")
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                batch = {key: val.to(device) for key, val in batch.items()}
                output_mappings = model(
                    input_ids=batch["sentences"], attention_mask=batch["attn_mask"]
                )
                if args.loss_type == "organ_loss":
                    loss = criterion(
                        output_mappings, batch["mappings"], batch["num_organs"]
                    )
                elif args.loss_type == "baseline_loss":
                    loss = criterion(output_mappings, batch["true_mappings"])
                else:
                    raise ValueError(f"Invalid loss type: {args.loss_type}")
                # backward
                loss.backward()
                # clip the gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                # update weights
                optimizer.step()
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss.item()})

        # Set model in evaluation mode
        model.train(False)
        # Reset current distance
        evaluator.reset_current_distance()
        with torch.no_grad():
            # Restart counters
            evaluator.reset_counters()
            for input_batch, organs_indices, _ in tqdm(val_loader):
                input_batch = {key: val.to(device) for key, val in input_batch.items()}
                output_mappings = model(
                    input_ids=input_batch["sentences"],
                    attention_mask=input_batch["attn_mask"],
                )
                output_mappings = output_mappings.cpu() * center
                for output_mapping, organ_indices in zip(
                    output_mappings.numpy(), organs_indices
                ):
                    evaluator.update_counters(output_mapping, organ_indices.numpy())

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
                    f"Found new best with avg distance: "
                    f"{evaluator.best_distance} on epoch "
                    f"{epoch+1}. Saving model!!!"
                )
                logging.info("======================")
                torch.save(model.state_dict(), args.save_model_path)
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
                args.save_intermediate_model_path,
            )


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    train(args)


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
        "--num_anchors",
        type=int,
        default=100,
        help="The number of anchor points to use.",
    )
    parser.add_argument(
        "--loss_type", type=str, default="organ_loss", help="The loss type",
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
        default=15,
        help="The number of epochs to train the model.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="The size of the batch."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="The learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="The (default) weight decay."
    )
    parser.add_argument(
        "--voxel_temperature", type=float, default=1.0, help="The voxel temperature."
    )
    parser.add_argument(
        "--organ_temperature", type=float, default=1.0, help="The organ temperature."
    )
    parser.add_argument(
        "--clip_val", type=float, default=2.0, help="The clipping threshold."
    )
    parser.add_argument("--masking", action="store_true", help="Whether to use masking")
    parser.add_argument(
        "--use_occurrences",
        action="store_true",
        help="Whether to use organ occurences as training.",
    )
    parser.add_argument(
        "--bert_name",
        type=str,
        default="bert-base-uncased",
        help="The pre-trained Bert model.",
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
