import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import json
import os
import logging
from transformers import BertConfig, BertTokenizer

from voxel_mapping.datasets import (
    VoxelSentenceMappingTrainClassDataset,
    VoxelSentenceMappingTestClassDataset,
    collate_pad_sentence_class_batch,
)
from voxel_retrieval.models import SiameseModel
from voxel_retrieval.losses import batch_all_triplet_loss
from voxel_retrieval.embedded_doc import EmbeddedDoc


def train(
    organs_dir_path: str,
    voxelman_images_path: str,
    train_json_path: str,
    val_json_path: str,
    project_size: int,
    epochs: int,
    batch_size: int,
    bert_name: str,
    save_model_path: str,
    learning_rate: float,
    weight_decay: float,
    clip_val: float,
    margin: float,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare jsons
    ind2organ = json.load(open(os.path.join(organs_dir_path, "ind2organ.json")))
    num_classes = max([int(index) for index in ind2organ.keys()]) + 1
    # Prepare datasets
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    train_dataset = VoxelSentenceMappingTrainClassDataset(
        train_json_path, tokenizer, num_classes, masking=False
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
        SiameseModel(bert_name, config, final_project_size=project_size)
    ).to(device)
    # noinspection PyUnresolvedReferences
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch + 1}...")
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for sentences, attn_mask, organ_indices, _ in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                sentences, attn_mask, organ_indices = (
                    sentences.to(device),
                    attn_mask.to(device),
                    organ_indices.to(device),
                )
                output_mappings = model(input_ids=sentences, attention_mask=attn_mask)
                loss, _ = batch_all_triplet_loss(organ_indices, output_mappings, margin)
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
        embedded_docs = []
        with torch.no_grad():
            for sentences, attn_mask, organs_indices, docs_ids in tqdm(val_loader):
                sentences, attn_mask = sentences.to(device), attn_mask.to(device)
                output_mappings = model(
                    input_ids=sentences, attention_mask=attn_mask
                ).cpu()
                for output_mapping, organ_indices, doc_id in zip(
                    output_mappings, organs_indices, docs_ids
                ):
                    embedded_docs.append(
                        EmbeddedDoc(
                            doc_id, organ_indices.numpy(), output_mapping.numpy()
                        )
                    )

        recalls = {"1": 0, "5": 0, "10": 0}
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
            print(f"The recall at {k} is: {round(recall/len(embedded_docs) * 100, 1)}")


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    train(
        args.organs_dir_path,
        args.voxelman_images_path,
        args.train_json_path,
        args.val_json_path,
        args.project_size,
        args.epochs,
        args.batch_size,
        args.bert_name,
        args.save_model_path,
        args.learning_rate,
        args.weight_decay,
        args.clip_val,
        args.margin,
    )


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
        default="models/retrieval.pt",
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
        "--clip_val", type=float, default=2.0, help="The clipping threshold."
    )
    parser.add_argument(
        "--project_size", type=int, default=3, help="The projection size."
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
        "--margin", type=float, default=10.0, help="The (default) margin."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
