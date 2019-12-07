import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from voxel_mapping.datasets import (
    VoxelSentenceMappingTrainDataset,
    VoxelSentenceMappingTestDataset,
    VoxelSentenceMappingTestMaskedDataset,
    collate_pad_sentence_batch,
)
from voxel_mapping.models import SentenceMappingsProducer
from voxel_mapping.losses import MinDistanceLoss
from voxel_mapping.evaluator import Evaluator


torch.multiprocessing.set_sharing_strategy("file_system")
# https://github.com/pytorch/pytorch/issues/973#issuecomment-426559250


def finetune(
    ind2organ_path: str,
    organ2label_path: str,
    voxelman_images_path: str,
    train_json_path: str,
    val_json_path: str,
    epochs: int,
    batch_size: int,
    bert_path_or_name: str,
    mask_probability: float,
    checkpoint_path: str,
    save_model_path: str,
    learning_rate: float,
    weight_decay: float,
    clip_val: float,
    joint_space: int,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = VoxelSentenceMappingTrainDataset(
        train_json_path, bert_path_or_name, mask_probability
    )
    val_dataset = VoxelSentenceMappingTestDataset(val_json_path, bert_path_or_name)
    val_masked_dataset = VoxelSentenceMappingTestMaskedDataset(
        val_json_path, bert_path_or_name
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_pad_sentence_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_sentence_batch,
    )
    val_masked_loader = DataLoader(
        val_masked_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_sentence_batch,
    )
    model = nn.DataParallel(
        SentenceMappingsProducer(bert_path_or_name, joint_space, finetune=True)
    ).to(device)
    # Load model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    criterion = MinDistanceLoss()
    # noinspection PyUnresolvedReferences
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Create evaluator
    evaluator = Evaluator(ind2organ_path, organ2label_path, voxelman_images_path)
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}...")
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for sentences, true_mappings, num_organs, _ in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                sentences, true_mappings, num_organs = (
                    sentences.to(device),
                    true_mappings.to(device),
                    num_organs.to(device),
                )
                output_mappings = model(sentences)
                loss = criterion(output_mappings, true_mappings, num_organs, device)
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
        # Reset current average accuracy
        evaluator.reset_current_average_accuracy()
        with torch.no_grad():
            # Restart counters
            evaluator.reset_counters()
            for sentences, _, _, organs_indices in tqdm(val_loader):
                sentences = sentences.to(device)
                output_mappings = model(sentences).cpu().numpy()
                for output_mapping, organ_indices in zip(
                    output_mappings, organs_indices
                ):
                    evaluator.update_counters(output_mapping, organ_indices.numpy())

            print(
                f"The accuracy on the non masked validation set is {evaluator.get_current_accuracy()}"
            )
            evaluator.update_current_average_accuracy()
            # Restart counters
            evaluator.reset_counters()
            for sentences, _, _, organs_indices in tqdm(val_masked_loader):
                sentences = sentences.to(device)
                output_mappings = model(sentences).cpu().numpy()
                for output_mapping, organ_indices in zip(
                    output_mappings, organs_indices
                ):
                    evaluator.update_counters(output_mapping, organ_indices.numpy())

            print(
                f"The accuracy on the masked validation set is {evaluator.get_current_accuracy()}"
            )
            evaluator.update_current_average_accuracy()
            evaluator.finalize_current_average_accuracy()

            if evaluator.is_best_avg_accuracy():
                evaluator.update_best_avg_accuracy()
                print("======================")
                print(
                    f"Found new best with avg accuracy {evaluator.best_avg_accuracy} on epoch "
                    f"{epoch+1}. Saving model!!!"
                )
                print("======================")
                torch.save(model.state_dict(), save_model_path)
            else:
                print(
                    f"Avg accuracy on epoch {epoch+1} is: {evaluator.current_average_accuracy}"
                )


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    finetune(
        args.ind2organ_path,
        args.organ2label_path,
        args.voxelman_images_path,
        args.train_json_path,
        args.val_json_path,
        args.epochs,
        args.batch_size,
        args.bert_path_or_name,
        args.mask_probability,
        args.checkpoint_path,
        args.save_model_path,
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
        description="Finetunes a sentence voxel mapping model."
    )
    parser.add_argument(
        "--ind2organ_path",
        type=str,
        default="data/data_organs/ind2organ.json",
        help="Path to the ind2organ path.",
    )
    parser.add_argument(
        "--organ2label_path",
        type=str,
        default="data/data_organs/organ2label.json",
        help="Path to the organ2label path.",
    )
    parser.add_argument(
        "--voxelman_images_path",
        type=str,
        default="data/voxelman/",
        help="Path to the voxel-man images",
    )
    parser.add_argument(
        "--train_json_path",
        type=str,
        default="data/train_dataset.json",
        help="Path to the training set",
    )
    parser.add_argument(
        "--val_json_path",
        type=str,
        default="data/val_dataset.json",
        help="Path to the validation set",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="models/sentence_finetuned.pt",
        help="Where to save the model.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="The number of epochs to train the model."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The size of the batch."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.00002, help="The learning rate."
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
        "--checkpoint_path",
        type=str,
        default="pretrained.pt",
        help="Path to a pretrained checkpoint.",
    )
    parser.add_argument(
        "--mask_probability", type=float, default=0.5, help="The mask probability."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
