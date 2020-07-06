import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import os
import sys
import logging
from transformers import BertConfig, BertTokenizer
import json
from utils.constants import VOXELMAN_CENTER

from voxel_mapping.datasets import (
    VoxelSentenceMappingTrainRegDataset,
    VoxelSentenceMappingTestRegDataset,
    collate_pad_sentence_reg_train_batch,
    collate_pad_sentence_reg_test_batch,
)
from voxel_mapping.models import RegModel
from voxel_mapping.losses import OrganDistanceLoss, BaselineRegLoss
from voxel_mapping.evaluator import TrainingEvaluator


def train(
    organs_dir_path: str,
    voxelman_images_path: str,
    train_json_path: str,
    val_json_path: str,
    num_anchors: str,
    loss_type: str,
    masking: bool,
    epochs: int,
    batch_size: int,
    bert_name: str,
    checkpoint_path: str,
    save_model_path: str,
    save_intermediate_model_path: str,
    log_filepath: str,
    learning_rate: float,
    weight_decay: float,
    voxel_temperature: float,
    organ_temperature: float,
    clip_val: float,
):
    # Set up logging
    if log_filepath:
        logging.basicConfig(level=logging.INFO, filename=log_filepath, filemode="w")
    else:
        logging.basicConfig(level=logging.INFO)
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare paths
    organ2voxels = json.load(open(os.path.join(organs_dir_path, "organ2voxels.json")))
    ind2organ = json.load(open(os.path.join(organs_dir_path, "ind2organ.json")))
    organ2label = json.load(open(os.path.join(organs_dir_path, "organ2label.json")))
    organ2summary = json.load(open(os.path.join(organs_dir_path, "organ2summary.json")))
    # Prepare datasets
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    logging.warning(f"Usage of masking is set to: ---{masking}---")
    train_dataset = VoxelSentenceMappingTrainRegDataset(
        train_json_path,
        tokenizer,
        ind2organ,
        organ2voxels,
        num_anchors=num_anchors,
        masking=masking,
    )
    val_dataset = VoxelSentenceMappingTestRegDataset(val_json_path, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_pad_sentence_reg_train_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_pad_sentence_reg_test_batch,
    )
    config = BertConfig.from_pretrained(bert_name)
    # Prepare model
    model = nn.DataParallel(RegModel(bert_name, config, final_project_size=3)).to(
        device
    )
    logging.warning(f"Using {num_anchors} voxel points!")
    # Check for the type of loss
    if loss_type == "organ_loss":
        criterion = OrganDistanceLoss(
            device=device,
            voxel_temperature=voxel_temperature,
            organ_temperature=organ_temperature,
        )
        logging.warning("Using SSL loss!")
    elif loss_type == "baseline_loss":
        criterion = BaselineRegLoss()
        logging.warning("Using baseline REG loss!")
    else:
        raise ValueError(f"Invalid loss type {loss_type}")
    # noinspection PyUnresolvedReference
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # Load model
    cur_epoch = 0
    best_distance = sys.maxsize
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
    center = torch.from_numpy(VOXELMAN_CENTER)
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
                if loss_type == "organ_loss":
                    loss = criterion(output_mappings, true_mappings, num_organs)
                elif loss_type == "baseline_loss":
                    loss = criterion(output_mappings, true_mappings)
                else:
                    raise ValueError(f"Invalid loss type: {loss_type}")
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
        # Reset current distance
        evaluator.reset_current_distance()
        with torch.no_grad():
            # Restart counters
            evaluator.reset_counters()
            for sentences, attn_mask, organs_indices, _ in tqdm(val_loader):
                sentences, attn_mask = (
                    sentences.to(device),
                    attn_mask.to(device),
                )
                output_mappings = model(input_ids=sentences, attention_mask=attn_mask)
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
        args.num_anchors,
        args.loss_type,
        args.masking,
        args.epochs,
        args.batch_size,
        args.bert_name,
        args.checkpoint_path,
        args.save_model_path,
        args.save_intermediate_model_path,
        args.log_filepath,
        args.learning_rate,
        args.weight_decay,
        args.voxel_temperature,
        args.organ_temperature,
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
        "--learning_rate", type=float, default=2e-5, help="The learning rate."
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
