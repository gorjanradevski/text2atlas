import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from voxel_mapping.datasets import (
    VoxelSentenceMappingTestRegDataset,
    VoxelSentenceMappingTestMaskedRegDataset,
    collate_pad_sentence_reg_test_batch,
)
from voxel_mapping.models import SentenceMappingsProducer
from voxel_mapping.evaluator import InferenceEvaluator


def inference(
    ind2organ_path: str,
    organ2label_path: str,
    voxelman_images_path: str,
    organ2summary_path: str,
    test_json_path: str,
    batch_size: int,
    bert_path_or_name: str,
    checkpoint_path: str,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_path_or_name)
    test_dataset = VoxelSentenceMappingTestRegDataset(test_json_path, tokenizer)
    test_masked_dataset = VoxelSentenceMappingTestMaskedRegDataset(
        test_json_path, tokenizer
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_sentence_reg_test_batch,
    )
    test_masked_loader = DataLoader(
        test_masked_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_sentence_reg_test_batch,
    )
    # Create model
    config = BertConfig.from_pretrained(bert_path_or_name)
    model = nn.DataParallel(SentenceMappingsProducer(bert_path_or_name, config)).to(
        device
    )
    # Load model
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
        # Restart counters
        evaluator.reset_counters()
        for sentences, organs_indices in tqdm(test_loader):
            sentences = sentences.to(device)
            output_mappings = model(sentences).cpu().numpy()
            for output_mapping, organ_indices in zip(output_mappings, organs_indices):
                evaluator.update_counters(output_mapping, organ_indices.numpy())

        print(f"The IOR on the non-masked test set is {evaluator.get_current_ior()}")
        print(
            f"The avg distance on the non-masked test set is {evaluator.get_current_distance()}"
        )
        print(
            f"The error bar of the IOR on the non-masked test set is {evaluator.get_ior_error_bar()}"
        )
        print(
            f"The error bar of the distance on the non-masked test set is {evaluator.get_distance_error_bar()}"
        )
        # Restart counters
        evaluator.reset_counters()
        for sentences, organs_indices in tqdm(test_masked_loader):
            sentences = sentences.to(device)
            output_mappings = model(sentences).cpu().numpy()
            for output_mapping, organ_indices in zip(output_mappings, organs_indices):
                evaluator.update_counters(output_mapping, organ_indices.numpy())

        print(f"The IOR on the masked test set is {evaluator.get_current_ior()}")
        print(
            f"The avg distance on the masked test set is {evaluator.get_current_distance()}"
        )
        print(
            f"The error bar of the IOR on the masked test set is {evaluator.get_ior_error_bar()}"
        )
        print(
            f"The error bar of the distance on the masked test set is {evaluator.get_distance_error_bar()}"
        )


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    inference(
        args.ind2organ_path,
        args.organ2label_path,
        args.voxelman_images_path,
        args.organ2summary_path,
        args.test_json_path,
        args.batch_size,
        args.bert_path_or_name,
        args.checkpoint_path,
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Performs class mapping inference.")
    parser.add_argument(
        "--organ2summary_path",
        type=str,
        default="data/data_organs_sages/organ2summary.json",
        help="Path to the organ2voxels path.",
    )
    parser.add_argument(
        "--ind2organ_path",
        type=str,
        default="data/data_organs_sages/ind2organ.json",
        help="Path to the ind2organ path.",
    )
    parser.add_argument(
        "--organ2label_path",
        type=str,
        default="data/data_organs_sages/organ2label.json",
        help="Path to the organ2label path.",
    )
    parser.add_argument(
        "--voxelman_images_path",
        type=str,
        default="data/voxelman_images",
        help="Path to the voxel-man images",
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
