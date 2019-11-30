import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from image_text_matching.datasets import JsonDataset, Subset, collate_pad_batch
from image_text_matching.evaluator import Evaluator
from image_text_matching.models import ImageTextMatchingModel


def pretrain(
    json_path: str,
    images_dir_path: str,
    train_size: int,
    batch_size: int,
    bert_path_or_name: str,
    checkpoint_path: str,
    joint_space: int,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = JsonDataset(json_path, images_dir_path, bert_path_or_name)
    print(f"The validation size is {len(dataset) - train_size}")
    val_dataset = Subset(dataset, list(range(train_size, len(dataset), 1)))

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4, collate_fn=collate_pad_batch
    )
    # Create model
    model = nn.DataParallel(
        ImageTextMatchingModel(bert_path_or_name, joint_space, finetune=False)
    ).to(device)
    # Load model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # Set model in evaluation mode
    model.train(False)

    evaluator = Evaluator(len(val_dataset), joint_space)
    with torch.no_grad():
        for images, sentences in tqdm(val_loader):
            images, sentences = images.to(device), sentences.to(device)
            embedded_images, embedded_sentences = model(images, sentences)
            evaluator.update_embeddings(
                embedded_images.cpu().numpy().copy(),
                embedded_sentences.cpu().numpy().copy(),
            )

    print("=============================")
    print(
        f"Image-text recall at 1, 5, 10: "
        f"{evaluator.image2text_recall_at_k()} \n"
        f"Text-image recall at 1, 5, 10: "
        f"{evaluator.text2image_recall_at_k()}"
    )
    print("=============================")


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    pretrain(
        args.json_path,
        args.images_dir_path,
        args.train_size,
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
    parser = argparse.ArgumentParser(description="Trains an image-text matching model.")
    parser.add_argument(
        "--json_path",
        type=str,
        default="data/dataset_image_text_matching.json",
        help="Path to the file where the image to caption mappings are.",
    )
    parser.add_argument(
        "--images_dir_path", type=str, default=".", help="Path to the images directory."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models/finetuned.pt",
        help="Path to a checkpoint.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The size of the batch."
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
        "--train_size",
        type=int,
        default=8000,
        help="The number of images in the training set.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
