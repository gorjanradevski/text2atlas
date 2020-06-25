import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np
from torch import nn
import os
from transformers import BertConfig, BertTokenizer

from voxel_mapping.datasets import (
    VoxelSentenceMappingTestClassDataset,
    collate_pad_sentence_class_batch,
)
from voxel_mapping.models import ClassModel
from voxel_mapping.evaluator import InferenceEvaluator


def inference(
    test_json_path: str,
    organs_dir_path: str,
    voxelman_images_path: str,
    batch_size: int,
    bert_name: str,
    checkpoint_path: str,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare paths
    organ2mass_path = os.path.join(organs_dir_path, "organ2center.json")
    ind2organ_path = os.path.join(organs_dir_path, "ind2organ.json")
    organ2label_path = os.path.join(organs_dir_path, "organ2label.json")
    organ2summary_path = os.path.join(organs_dir_path, "organ2summary.json")
    # Load organ to indices to obtain the number of classes
    ind2organ = json.load(open(ind2organ_path))
    organ2center = json.load(open(organ2mass_path))
    num_classes = max([int(index) for index in ind2organ.keys()]) + 1
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    test_dataset = VoxelSentenceMappingTestClassDataset(
        test_json_path, tokenizer, num_classes
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_pad_sentence_class_batch,
    )
    config = BertConfig.from_pretrained(bert_name)
    model = nn.DataParallel(
        ClassModel(bert_name, config, final_project_size=num_classes)
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # Set model in evaluation mode
    model.train(False)
    # Create evaluator
    evaluator = InferenceEvaluator(
        ind2organ_path,
        organ2label_path,
        organ2summary_path,
        voxelman_images_path,
        len(test_dataset),
    )
    with torch.no_grad():
        evaluator.reset_counters()
        for sentences, attn_mask, organs_indices in tqdm(test_loader):
            sentences, attn_mask = sentences.to(device), attn_mask.to(device)
            output_mappings = model(input_ids=sentences, attention_mask=attn_mask)
            y_pred = torch.argmax(output_mappings, dim=-1)
            pred_organ_names = [ind2organ[str(ind.item())] for ind in y_pred]
            pred_centers = [organ2center[organ_name] for organ_name in pred_organ_names]
            for pred_center, organ_indices in zip(pred_centers, organs_indices):
                evaluator.update_counters(
                    np.array(pred_center), np.where(organ_indices == 1)[0]
                )

        print(
            "The IOR on the test set is: "
            f"{evaluator.get_current_ior()} +/- {evaluator.get_ior_error_bar()}"
        )
        print(
            "The avg distance on the test set is: "
            f"{evaluator.get_current_distance()} +/- {evaluator.get_distance_error_bar()}"
        )
        print(
            "The avg miss distance on the test set is: "
            f"{evaluator.get_current_miss_distance()} +/- {evaluator.get_miss_distance_error_bar()}"
        )


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    inference(
        args.test_json_path,
        args.organs_dir_path,
        args.voxelman_images_path,
        args.batch_size,
        args.bert_name,
        args.checkpoint_path,
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
        "--organs_dir_path",
        type=str,
        default="data/data_organs",
        help="Path to the data organs directory path.",
    )
    parser.add_argument(
        "--voxelman_images_path",
        type=str,
        default="data/voxelman_images",
        help="Path to the voxelman images.",
    )
    parser.add_argument(
        "--test_json_path",
        type=str,
        default="data/dataset_text_atlas_mapping_test_fixd.json",
        help="Path to the test set",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="The size of the batch."
    )
    parser.add_argument(
        "--bert_name",
        type=str,
        default="bert-base-uncased",
        help="Should be one of [bert-base-uncased, allenai/scibert_scivocab_uncased,"
        "monologg/biobert_v1.1_pubmed, emilyalsentzer/Bio_ClinicalBERT,"
        "google/bert_uncased_L-8_H-512_A-8]",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a pretrained checkpoint.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
