import argparse
import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from voxel_mapping.constants import VOXELMAN_CENTER
from voxel_mapping.datasets import (
    VoxelSentenceMappingTestRegDataset,
    collate_pad_sentence_reg_test_batch,
)
from voxel_mapping.evaluator import InferenceEvaluatorPerOrgan
from voxel_mapping.models import RegModel


@torch.no_grad()
def inference(args):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.bert_name)
    # Prepare jsons
    ind2organ = json.load(open(os.path.join(args.organs_dir_path, "ind2organ.json")))
    organ2label = json.load(
        open(os.path.join(args.organs_dir_path, "organ2label.json"))
    )
    organ2voxels = json.load(
        open(os.path.join(args.organs_dir_path, "organ2voxels.json"))
    )
    test_dataset = VoxelSentenceMappingTestRegDataset(
        args.test_json_path, tokenizer, ind2organ
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_pad_sentence_reg_test_batch,
    )
    # Create model
    config = BertConfig.from_pretrained(args.bert_name)
    model = nn.DataParallel(RegModel(args.bert_name, config, final_project_size=3)).to(
        device
    )
    # Load model
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    # Set model in evaluation mode
    model.train(False)
    # Create evaluator
    evaluator = InferenceEvaluatorPerOrgan(
        ind2organ,
        organ2label,
        organ2voxels,
        args.voxelman_images_path,
        test_dataset.organ2count,
        len(test_dataset),
    )
    center = torch.from_numpy(VOXELMAN_CENTER)
    # Restart counters
    evaluator.reset_counters()
    for input_batch, organs_indices, _ in tqdm(test_loader):
        input_batch = {key: val.to(device) for key, val in input_batch.items()}
        output_mappings = (
            model(
                input_ids=input["sentences"], attention_mask=input_batch["attn_mask"],
            ).cpu()
            * center
        )
        for output_mapping, organ_indices in zip(output_mappings, organs_indices):
            evaluator.update_counters(output_mapping.numpy(), organ_indices.numpy())

        print(
            "The avg IOR on the test set is: "
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
        print("============================================")
        for organ_name in evaluator.organ2count.keys():
            if evaluator.get_current_ior_for_organ(organ_name) > -1:
                print(
                    f"The avg IOR for {organ_name} is: "
                    f"{evaluator.get_current_ior_for_organ(organ_name)} +/- "
                    f"{evaluator.get_ior_error_bar_for_organ(organ_name)}"
                )
                print(
                    f"The avg NVD {organ_name} is: "
                    f"{evaluator.get_current_distance_for_organ(organ_name)} +/- "
                    f"{evaluator.get_distance_error_bar_for_organ(organ_name)}"
                )
                print(
                    f"The avg NVD-O {organ_name} is: "
                    f"{evaluator.get_current_miss_distance_for_organ(organ_name)} +/- "
                    f"{evaluator.get_miss_distance_error_bar_for_organ(organ_name)}"
                )
                print("============================================")


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    inference(args)


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Performs class mapping inference.")
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
        help="Path to the voxelman images",
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
        help="The pre-trained Bert model.",
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
