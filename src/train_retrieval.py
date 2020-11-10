import argparse
import json
import os

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from voxel_mapping.datasets import (
    VoxelSentenceMappingTestClassDataset,
    VoxelSentenceMappingTrainClassDataset,
    collate_pad_sentence_class_test_batch,
    collate_pad_sentence_class_train_batch,
)
from voxel_mapping.models import SiameseModel
from voxel_mapping.retrieval_utils import EmbeddedDoc, batch_all_triplet_loss


def train(args):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare jsons
    ind2organ = json.load(open(os.path.join(args.organs_dir_path, "ind2organ.json")))
    num_classes = max([int(index) for index in ind2organ.keys()]) + 1
    # Prepare datasets
    tokenizer = BertTokenizer.from_pretrained(args.bert_name)
    train_dataset = VoxelSentenceMappingTrainClassDataset(
        args.train_json_path, tokenizer, num_classes, masking=False
    )
    val_dataset = VoxelSentenceMappingTestClassDataset(
        args.val_json_path, tokenizer, num_classes
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pad_sentence_class_train_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_pad_sentence_class_test_batch,
    )
    config = BertConfig.from_pretrained(args.bert_name)
    # Prepare model
    model = nn.DataParallel(
        SiameseModel(args.bert_name, config, final_project_size=args.project_size)
    ).to(device)
    # noinspection PyUnresolvedReferences
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_recall = -1
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch + 1}...")
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
                loss, _ = batch_all_triplet_loss(
                    batch["organ_indices"], output_mappings, args.margin, device
                )
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
        embedded_docs = []
        with torch.no_grad():
            for input_batch, organs_indices, docs_ids in tqdm(val_loader):
                input_batch = {key: val.to(device) for key, val in input_batch.items()}
                output_mappings = model(
                    input_ids=input_batch["sentences"],
                    attention_mask=input_batch["attn_mask"],
                ).cpu()
                for output_mapping, organ_indices, doc_id in zip(
                    output_mappings, organs_indices, docs_ids
                ):
                    # Get only non -1 indices
                    organ_indices = organ_indices[: (organ_indices >= 0).sum()]
                    embedded_docs.append(
                        EmbeddedDoc(
                            doc_id, organ_indices.numpy(), output_mapping.numpy()
                        )
                    )

        recalls = {"1": 0, "5": 0, "10": 0}
        total_recall = 0
        for document1 in tqdm(embedded_docs):
            cur_doc_distances = []
            for document2 in embedded_docs:
                if document1.doc_id == document2.doc_id:
                    continue
                cur_doc_distances.append(
                    (document2.organ_indices, document1.docs_distance(document2))
                )
            cur_doc_distances_sorted = sorted(cur_doc_distances, key=lambda tup: tup[1])
            for k in recalls.keys():
                for cur_doc in cur_doc_distances_sorted[: int(k)]:
                    if cur_doc[0].shape == document1.organ_indices.shape:
                        if (cur_doc[0] == document1.organ_indices).all():
                            recalls[k] += 1
                            break

        for k, recall in recalls.items():
            total_recall += round(recall / len(embedded_docs) * 100, 1)
            print(
                f"The recall at {k} is: {round(recall / len(embedded_docs) * 100, 1)}"
            )

        if total_recall >= best_recall:
            print("===================================")
            print(f"Found new best on epoch {epoch+1}. Saving model!!!")
            print("===================================")
            best_recall = total_recall
            torch.save(model.state_dict(), args.save_model_path)


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
    parser = argparse.ArgumentParser(description="Trains retrieval model.")
    parser.add_argument(
        "--organs_dir_path",
        type=str,
        default="data/data_organs_mesh",
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
        default="data/mesh_dataset_train.json",
        help="Path to the training set",
    )
    parser.add_argument(
        "--val_json_path",
        type=str,
        default="data/mesh_dataset_val.json",
        help="Path to the validation set",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="models/retrieval_3.pt",
        help="Where to save the model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The number of epochs to train the model.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="The size of the batch."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="The learning rate."
    )
    parser.add_argument(
        "--clip_val", type=float, default=2.0, help="The clipping threshold."
    )
    parser.add_argument(
        "--project_size", type=int, default=3, help="The projection size."
    )
    parser.add_argument(
        "--bert_name",
        type=str,
        default="bert-base-uncased",
        help="The pre-trained Bert model.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="The (default) weight decay."
    )
    parser.add_argument(
        "--margin", type=float, default=2.0, help="The (default) margin."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
