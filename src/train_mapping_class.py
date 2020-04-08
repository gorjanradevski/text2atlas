import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import json
from transformers import BertConfig

from voxel_mapping.datasets import (
    VoxelSentenceMappingTrainClassDataset,
    VoxelSentenceMappingTestClassDataset,
    VoxelSentenceMappingTestMaskedClassDataset,
    collate_pad_sentence_class_batch,
)
from voxel_mapping.models import SentenceMappingsProducer


def train(
    ind2organ_path: str,
    train_json_path: str,
    val_json_path: str,
    epochs: int,
    batch_size: int,
    bert_path_or_name: str,
    mask_probability: float,
    checkpoint_path: str,
    save_model_path: str,
    save_intermediate_model_path: str,
    learning_rate: float,
    weight_decay: float,
    clip_val: float,
    joint_space: int,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load organ to indices to obtain the number of classes
    num_classes = len([index for index in json.load(open(ind2organ_path)).keys()])
    train_dataset = VoxelSentenceMappingTrainClassDataset(
        train_json_path, bert_path_or_name, mask_probability, num_classes
    )
    val_dataset = VoxelSentenceMappingTestClassDataset(
        val_json_path, bert_path_or_name, num_classes
    )
    val_masked_dataset = VoxelSentenceMappingTestMaskedClassDataset(
        val_json_path, bert_path_or_name, num_classes
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_pad_sentence_class_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_sentence_class_batch,
    )
    val_masked_loader = DataLoader(
        val_masked_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_sentence_class_batch,
    )
    config = BertConfig.from_pretrained(bert_path_or_name)
    model = nn.DataParallel(
        SentenceMappingsProducer(bert_path_or_name, config, reg_or_class="class")
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    # noinspection PyUnresolvedReferences
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_avg_acc = -1
    cur_epoch = 0
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        cur_epoch = checkpoint["epoch"]
        best_avg_acc = checkpoint["best_avg_acc"]
        # https://discuss.pytorch.org/t/cuda-out-of-memory-after-loading-model/50681
        del checkpoint
        print(
            f"Starting training from checkpoint {checkpoint_path} with starting epoch {cur_epoch}!"
        )
        print(f"The previous best accuracy was: {best_avg_acc}!")
    for epoch in range(cur_epoch, cur_epoch + epochs):
        print(f"Starting epoch {epoch + 1}...")
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for sentences, organ_indices in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                sentences, organ_indices = (
                    sentences.to(device),
                    organ_indices.to(device),
                )
                output_mappings = model(sentences)
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
        with torch.no_grad():
            corrects = 0
            totals = 0
            cur_unmasked_acc = 0
            for sentences, organ_indices in tqdm(val_loader):
                sentences = sentences.to(device)
                output_mappings = model(sentences)
                y_pred = torch.argmax(output_mappings, dim=1)
                y_one_hot = torch.zeros(organ_indices.size()[0], num_classes)
                y_one_hot[torch.arange(organ_indices.size()[0]), y_pred] = 1
                y_one_hot[torch.where(y_one_hot == 0)] = -100
                corrects += (y_one_hot == organ_indices).sum(dim=1).sum().item()
                totals += organ_indices.size()[0]

            cur_unmasked_acc = corrects * 100 / totals
            print(
                f"The accuracy on the non masked validation set is {cur_unmasked_acc}"
            )
            corrects = 0
            totals = 0
            cur_masked_acc = 0
            for sentences, organ_indices in tqdm(val_masked_loader):
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
            if (cur_unmasked_acc + cur_masked_acc) / 2 > best_avg_acc:
                best_avg_acc = (cur_unmasked_acc + cur_masked_acc) / 2
                print("======================")
                print(
                    f"Found new best with avg accuracy {best_avg_acc} on epoch "
                    f"{epoch+1}. Saving model!!!"
                )
                torch.save(model.state_dict(), save_model_path)
                print("======================")
            else:
                print(
                    f"Avg accuracy on epoch {epoch+1} is: {(cur_unmasked_acc + cur_masked_acc) / 2}"
                )
            print("Saving intermediate checkpoint...")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_avg_acc": best_avg_acc,
                },
                save_intermediate_model_path,
            )


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    train(
        args.ind2organ_path,
        args.train_json_path,
        args.val_json_path,
        args.epochs,
        args.batch_size,
        args.bert_path_or_name,
        args.mask_probability,
        args.checkpoint_path,
        args.save_model_path,
        args.save_intermediate_model_path,
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
        "--ind2organ_path",
        type=str,
        default="data/data_organs_new/ind2organ_new.json",
        help="Path to the ind2organ path.",
    )
    parser.add_argument(
        "--train_json_path",
        type=str,
        default="data/dataset_text_atlas_mapping_train_new.json",
        help="Path to the training set",
    )
    parser.add_argument(
        "--val_json_path",
        type=str,
        default="data/dataset_text_atlas_mapping_val_new.json",
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
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="The learning rate."
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
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a pretrained checkpoint.",
    )
    parser.add_argument(
        "--save_intermediate_model_path",
        type=str,
        default="models/intermediate_sentence_mapping_regressor.pt",
        help="Where to save the intermediate checkpoint model.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
