from torch.utils.data import Dataset
import os
from PIL import Image
from typing import Tuple
import torch
import json
from transformers import BertTokenizer
from torchvision import transforms


class JsonDataset(Dataset):
    def __init__(
        self,
        json_file_path: str,
        images_dir_path: str,
        bert_tokenizer_path_or_name: str,
    ):
        data = json.load(open(json_file_path))
        self.images_dir_path = images_dir_path
        self.image_paths = [element["figure"] for element in data]
        self.captions = [element["caption"] for element in data]
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path_or_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(
            os.path.join(self.images_dir_path, self.image_paths[idx])
        ).convert("RGB")
        image_transformed = self.transform(image)
        caption = torch.tensor(
            self.tokenizer.encode(self.captions[idx], add_special_tokens=True)
        )

        return image_transformed, caption


class Subset:
    r"""
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def collate_pad_batch(batch: Tuple[torch.Tensor, torch.Tensor]):
    images, sentences = zip(*batch)
    images = torch.stack(images, 0)
    padded_sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)

    return images, padded_sentences
