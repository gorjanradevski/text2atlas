import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from image_text_matching.datasets import JsonDataset, Subset
from image_text_matching.evaluator import Evaluator
from image_text_matching.datasets import collate_pad_batch
from image_text_matching.models import ImageTextMatchingModel
from image_text_matching.losses import TripletLoss


def train(
    json_path: str,
    images_dir_path: str,
    train_size: int,
    epochs: int,
    batch_size: int,
    save_model_path: str,
    learning_rate: float,
    weight_decay: float,
    clip_val: float,
    joint_space: int,
    margin: float,
    batch_hard: bool,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = JsonDataset(json_path, images_dir_path)
    train_dataset = Subset(dataset, list(range(train_size)))
    val_dataset = Subset(dataset, list(range(len(dataset) - train_size)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_pad_batch,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4, collate_fn=collate_pad_batch
    )
    model = nn.DataParallel(ImageTextMatchingModel(joint_space, finetune=False)).to(
        device
    )
    criterion = TripletLoss(margin, batch_hard)
    # noinspection PyUnresolvedReferences
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    evaluator = Evaluator(len(val_dataset), joint_space)
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}...")
        evaluator.reset_all_vars()

        # Set model in train mode
        model.train(True)

        with tqdm(total=len(train_loader)) as pbar:
            for images, sentences in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                images, sentences = images.to(device), sentences.to(device)
                embedded_images, embedded_sentences = model(images, sentences)
                loss = criterion(embedded_images, embedded_sentences)
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
            for images, sentences in tqdm(val_loader):
                images, sentences = images.to(device), sentences.to(device)
                embedded_images, embedded_sentences = model(images, sentences)
                evaluator.update_embeddings(
                    embedded_images.cpu().numpy().copy(),
                    embedded_sentences.cpu().numpy().copy(),
                )

        if evaluator.is_best_recall_at_k():
            evaluator.update_best_recall_at_k()
            print("=============================")
            print(
                f"Found new best on epoch {epoch + 1}!! Saving model!\n"
                f"Current image-text recall at 1, 5, 10: "
                f"{evaluator.best_image2text_recall_at_k} \n"
                f"Current text-image recall at 1, 5, 10: "
                f"{evaluator.best_text2image_recall_at_k}"
            )
            print("=============================")
            torch.save(model.state_dict(), save_model_path)
        else:
            print("=============================")
            print(
                f"Metrics on epoch {epoch + 1}\n"
                f"Current image-text recall at 1, 5, 10: "
                f"{evaluator.cur_image2text_recall_at_k} \n"
                f"Current text-image recall at 1, 5, 10:"
                f"{evaluator.cur_text2image_recall_at_k}"
            )
            print("=============================")


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    train(
        args.json_path,
        args.images_dir_path,
        args.train_size,
        args.epochs,
        args.batch_size,
        args.save_model_path,
        args.learning_rate,
        args.weight_decay,
        args.clip_val,
        args.joint_space,
        args.margin,
        args.batch_hard,
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
        default="data/dataset.json",
        help="Path where all images are.",
    )
    parser.add_argument(
        "--images_dir_path",
        type=str,
        default=".",
        help="Path to the file where the image to caption mappings are.",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=8000,
        help="The number of images in the training set.",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="models/pretrained.pt",
        help="Where to save the model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="The number of epochs to train the model excluding the vgg.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The size of the batch."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0002, help="The learning rate."
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
        "--margin", type=float, default=0.2, help="The contrastive margin."
    )
    parser.add_argument(
        "--clip_val", type=float, default=2.0, help="The clipping threshold."
    )
    parser.add_argument(
        "--batch_hard", action="store_true", help="Whether to train on hard negatives."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
