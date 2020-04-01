import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np
from torch import nn
from transformers import BertConfig

from voxel_mapping.datasets import (
    VoxelSentenceMappingTestMaskedClassDataset,
    VoxelSentenceMappingTestClassDataset,
    collate_pad_sentence_class_batch,
)
from voxel_mapping.models import SentenceMappingsProducer
from voxel_mapping.evaluator import InferenceEvaluator


def inference(
    test_json_path: str,
    organ2mass_path: str,
    ind2organ_path: str,
    organ2label_path: str,
    voxelman_images_path: str,
    organ2summary_path: str,
    batch_size: int,
    bert_path_or_name: str,
    checkpoint_path: str,
    joint_space: int,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load organ to indices to obtain the number of classes
    ind2organ = json.load(open(ind2organ_path))
    organ2center = json.load(open(organ2mass_path))
    num_classes = len([index for index in ind2organ.keys()])
    test_dataset = VoxelSentenceMappingTestClassDataset(
        test_json_path, bert_path_or_name, num_classes
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_sentence_class_batch,
    )
    test_dataset_masked = VoxelSentenceMappingTestMaskedClassDataset(
        test_json_path, bert_path_or_name, num_classes
    )

    test_masked_loader = DataLoader(
        test_dataset_masked,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_sentence_class_batch,
    )
    config = BertConfig.from_pretrained(bert_path_or_name)
    model = nn.DataParallel(
        SentenceMappingsProducer(
            bert_path_or_name,
            joint_space,
            config,
            reg_or_class="class",
            num_classes=num_classes,
        )
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # Set model in evaluation mode
    model.train(False)
    # Create evaluator
    evaluator = InferenceEvaluator(
        ind2organ_path,
        organ2label_path,
        voxelman_images_path,
        organ2summary_path,
        len(test_dataset),
    )
    with torch.no_grad():
        # UNMASKED SETTING
        evaluator.reset_counters()
        for sentences, organs_indices in tqdm(test_loader):
            sentences = sentences.to(device)
            output_mappings = model(sentences)
            y_pred = torch.argmax(output_mappings, dim=-1)
            pred_organ_names = [ind2organ[str(ind.item())] for ind in y_pred]
            pred_centers = [organ2center[organ_name] for organ_name in pred_organ_names]
            for pred_center, organ_indices in zip(pred_centers, organs_indices):
                evaluator.update_counters(
                    np.array(pred_center), np.where(organ_indices == 1)[0]
                )

        print(
            f"The accuracy on the non-masked test set is {evaluator.get_current_accuracy()}"
        )
        print(
            f"The avg distance on the non-masked test set is {evaluator.get_current_distance()}"
        )
        print(
            f"The error bar of the accuracy on the non-masked test set is {evaluator.get_accuracy_error_bar()}"
        )
        print(
            f"The error bar of the distance on the non-masked test set is {evaluator.get_distance_error_bar()}"
        )
        # MASKED SETTING
        evaluator.reset_counters()
        for sentences, organs_indices in tqdm(test_masked_loader):
            sentences = sentences.to(device)
            output_mappings = model(sentences)
            y_pred = torch.argmax(output_mappings, dim=1)
            # Measure distance
            pred_organ_names = [ind2organ[str(ind.item())] for ind in y_pred]
            pred_centers = [organ2center[organ_name] for organ_name in pred_organ_names]
            for pred_center, organ_indices in zip(pred_centers, organs_indices):
                evaluator.update_counters(
                    np.array(pred_center), np.where(organ_indices == 1)[0]
                )

        print(
            f"The accuracy on the masked test set is {evaluator.get_current_accuracy()}"
        )
        print(
            f"The avg distance on the masked test set is {evaluator.get_current_distance()}"
        )
        print(
            f"The error bar of the accuracy on the masked test set is {evaluator.get_accuracy_error_bar()}"
        )
        print(
            f"The error bar of the distance on the masked test set is {evaluator.get_distance_error_bar()}"
        )


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    inference(
        args.test_json_path,
        args.organ2mass_path,
        args.ind2organ_path,
        args.organ2label_path,
        args.voxelman_images_path,
        args.organ2summary_path,
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
        "--organ2summary_path",
        type=str,
        default="data/data_organs_new/organ2voxels_new.json",
        help="Path to the organ2label file.",
    )
    parser.add_argument(
        "--voxelman_images_path",
        type=str,
        default="data/data_organs_new/voxelman_images",
        help="Path to the voxelman images.",
    )
    parser.add_argument(
        "--organ2label_path",
        type=str,
        default="data/data_organs_new/organ2label_new.json",
        help="Path to the organ2label file.",
    )
    parser.add_argument(
        "--organ2mass_path",
        type=str,
        default="data/data_organs_new/organ2mass_new.json",
        help="Path to the organ2center path.",
    )
    parser.add_argument(
        "--ind2organ_path",
        type=str,
        default="data/data_organs_new/ind2organ_new.json",
        help="Path to the ind2organ path.",
    )
    parser.add_argument(
        "--test_json_path",
        type=str,
        default="data/dataset_text_atlas_mapping_test_new.json",
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
        default=None,
        help="Path to a pretrained checkpoint.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
