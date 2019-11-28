from transformers import BertTokenizer
from torch.utils.data import Dataset
import json
import torch
from typing import Tuple
import random


class VoxelMappingDataset(Dataset):
    # Assumes that the dataset is: {
    # "sentence": str,
    # "keywords": set,
    # "location_map": List[[float, float, float], [float, float, float],...],
    # "bounding_box": List[[float, float], [float, float], [float, float]]
    # }
    def __init__(self, json_path: str):
        self.json_data = json.load(open(json_path))
        self.sentences = [element["sentence"] for element in self.json_data]
        self.mappings = [element["location_map"] for element in self.json_data]
        self.keywords = [set(element["keywords"]) for element in self.json_data]
        self.bounding_boxes = [element["bounding_box"] for element in self.json_data]
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        mask = {word: random.choice([0, 1]) for word in self.keywords[idx]}
        masked_sentence = " ".join(
            [
                "[MASK]" if word in mask and mask[word] == 1 else word
                for word in self.sentences[idx].split()
            ]
        )
        tokenized_sentence = torch.tensor(
            self.tokenizer.encode(masked_sentence, add_special_tokens=True)
        )
        mapping = torch.tensor(self.mappings[idx])
        bounding_box = torch.tensor(self.bounding_boxes[idx])
        num_organs = len(mapping)

        return (tokenized_sentence, mapping, num_organs, bounding_box)


def collate_pad_batch(
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    sentences, mappings, num_organs, bounding_boxes = zip(*batch)
    padded_sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)
    padded_mappings = torch.nn.utils.rnn.pad_sequence(mappings, batch_first=True)
    num_organs = torch.tensor([*num_organs])

    # IDK why num_organs is a Tuple
    return padded_sentences, padded_mappings, num_organs, bounding_boxes
