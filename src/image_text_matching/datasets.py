from torch.utils.data import Dataset
import os
from PIL import Image
from typing import Tuple
import torch


class ImagesDataset(Dataset):
    def __init__(self, images_path: str, transform=None):
        self.images_paths = [
            os.path.join(images_path, image_path)
            for image_path in os.listdir(images_path)
            if image_path.endswith(".jpg")
        ]
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = self.images_paths[idx]
        image_name = image_path.split("/")[-1][:-4]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return image, image_name


class JsonDataset(Dataset):
    # Makes assumption that the dataset is a json file that contains objects such as:
    # {"image_embedding_path": str "image_embedding_path": str, "label":int}

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(idx: int):
        pass
