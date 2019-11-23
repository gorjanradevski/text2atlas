import argparse
from data_processing.datasets import ImagesDataset
from image_text_matching.models import ImageEncoder
from tqdm import tqdm
from torch import nn
import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader


def dump_embeddings(images_path: str, embeddings_path: str, batch_size: int) -> None:
    """List the directory with images and dump the embeddings.

    Arguments:
        images_path: Path to the directory of images.
        embeddings_path: Where to dump the embeddings.
        batch_size: The batch size.
    """
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImagesDataset(
        images_path,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    model = nn.DataParallel(ImageEncoder()).to(device)
    for image, image_name in tqdm(dataloader):
        image = image.to(device)
        image_embedding = model(image)
        torch.save(
            image_embedding, os.path.join(embeddings_path, image_name[0]) + ".pt"
        )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Dumps image embeddings.")
    parser.add_argument(
        "--image_embeddings_path",
        type=str,
        default="data/image_embeddings",
        help="Where to dump image embeddings.",
    )
    parser.add_argument(
        "--images_path", type=str, default="data/images", help="Path to the images."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="The size of the batch."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    dump_embeddings(args.images_path, args.image_embeddings_path, args.batch_size)


if __name__ == "__main__":
    main()
