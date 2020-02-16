from transformers import BertTokenizer
from torch.utils.data import Dataset
import json
import torch
from typing import Tuple
import numpy as np
import nltk


class VoxelSentenceMappingRegDataset:
    def __init__(self, json_path: str, bert_tokenizer_path_or_name: str):
        self.json_data = json.load(open(json_path))
        self.sentences, self.mappings, self.keywords, self.organs_indices = (
            [],
            [],
            [],
            [],
        )
        for element in self.json_data:
            if len(element["text"]) > 200:
                continue
            self.sentences.append(element["text"])
            self.mappings.append(element["centers"])
            self.keywords.append(element["keywords"])
            self.organs_indices.append(element["organ_indices"])
        self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path_or_name)


class VoxelSentenceMappingTrainRegDataset(VoxelSentenceMappingRegDataset, Dataset):
    def __init__(
        self, json_path: str, bert_tokenizer_path_or_name: str, mask_probability: float
    ):
        super().__init__(json_path, bert_tokenizer_path_or_name)
        self.mask_probability = mask_probability

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        mask = {
            word: np.random.choice(
                [0, 1], p=[1 - self.mask_probability, self.mask_probability]
            )
            for word in self.keywords[idx]
        }
        masked_sentence = " ".join(
            [
                "[MASK]" if word in mask and mask[word] == 1 else word
                for word in nltk.word_tokenize(self.sentences[idx])
            ]
        )
        tokenized_sentence = torch.tensor(
            self.tokenizer.encode(masked_sentence, add_special_tokens=True)
        )
        mapping = torch.tensor(self.mappings[idx])
        organ_indices = torch.tensor(self.organs_indices[idx])
        num_organs = len(mapping)

        return (tokenized_sentence, mapping, num_organs, organ_indices)


class VoxelSentenceMappingTestRegDataset(VoxelSentenceMappingRegDataset, Dataset):
    def __init__(self, json_path: str, bert_tokenizer_path_or_name: str):
        super().__init__(json_path, bert_tokenizer_path_or_name)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        tokenized_sentence = torch.tensor(
            self.tokenizer.encode(self.sentences[idx], add_special_tokens=True)
        )
        mapping = torch.tensor(self.mappings[idx])
        organ_indices = torch.tensor(self.organs_indices[idx])
        num_organs = len(mapping)

        return (tokenized_sentence, mapping, num_organs, organ_indices)


class VoxelSentenceMappingTestMaskedRegDataset(VoxelSentenceMappingRegDataset, Dataset):
    def __init__(self, json_path: str, bert_tokenizer_path_or_name: str):
        super().__init__(json_path, bert_tokenizer_path_or_name)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        mask = {word for word in self.keywords[idx]}
        masked_sentence = " ".join(
            [
                "[MASK]" if word in mask else word
                for word in nltk.word_tokenize(self.sentences[idx])
            ]
        )
        tokenized_sentence = torch.tensor(
            self.tokenizer.encode(masked_sentence, add_special_tokens=True)
        )
        mapping = torch.tensor(self.mappings[idx])
        organ_indices = torch.tensor(self.organs_indices[idx])
        num_organs = len(mapping)

        return (tokenized_sentence, mapping, num_organs, organ_indices)


def collate_pad_sentence_reg_batch(
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    sentences, mappings, num_organs, organ_indices = zip(*batch)
    padded_sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)
    padded_mappings = torch.nn.utils.rnn.pad_sequence(mappings, batch_first=True)
    num_organs = torch.tensor([*num_organs])
    padded_organ_indices = torch.nn.utils.rnn.pad_sequence(
        organ_indices, batch_first=True, padding_value=-1
    )

    return padded_sentences, padded_mappings, num_organs, padded_organ_indices


class VoxelSentenceMappingClassDataset:
    def __init__(
        self, json_path: str, bert_tokenizer_path_or_name: str, num_classes: int
    ):
        self.json_data = json.load(open(json_path))
        self.sentences, self.organs_indices, self.keywords = [], [], []
        self.num_classes = num_classes
        for element in self.json_data:
            if len(element["text"]) > 200:
                continue
            self.sentences.append(element["text"])
            self.organs_indices.append(element["organ_indices"])
            self.keywords.append(element["keywords"])
        self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path_or_name)


class VoxelSentenceMappingTrainClassDataset(VoxelSentenceMappingClassDataset, Dataset):
    def __init__(
        self,
        json_path: str,
        bert_tokenizer_path_or_name: str,
        mask_probability: float,
        num_classes: int,
    ):
        super().__init__(json_path, bert_tokenizer_path_or_name, num_classes)
        self.mask_probability = mask_probability

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        mask = {
            word: np.random.choice(
                [0, 1], p=[1 - self.mask_probability, self.mask_probability]
            )
            for word in self.keywords[idx]
        }
        masked_sentence = " ".join(
            [
                "[MASK]" if word in mask and mask[word] == 1 else word
                for word in nltk.word_tokenize(self.sentences[idx])
            ]
        )
        tokenized_sentence = torch.tensor(
            self.tokenizer.encode(masked_sentence, add_special_tokens=True)
        )
        organ_indices = torch.tensor(self.organs_indices[idx])
        one_hot = torch.zeros(self.num_classes)
        one_hot[organ_indices] = 1

        return tokenized_sentence, one_hot


class VoxelSentenceMappingTestClassDataset(VoxelSentenceMappingClassDataset, Dataset):
    def __init__(
        self, json_path: str, bert_tokenizer_path_or_name: str, num_classes: int
    ):
        super().__init__(json_path, bert_tokenizer_path_or_name, num_classes)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        tokenized_sentence = torch.tensor(
            self.tokenizer.encode(self.sentences[idx], add_special_tokens=True)
        )
        organ_indices = torch.tensor(self.organs_indices[idx])
        one_hot = torch.zeros(self.num_classes)
        one_hot[organ_indices] = 1

        return tokenized_sentence, one_hot


class VoxelSentenceMappingTestMaskedClassDataset(
    VoxelSentenceMappingClassDataset, Dataset
):
    def __init__(
        self, json_path: str, bert_tokenizer_path_or_name: str, num_classes: int
    ):
        super().__init__(json_path, bert_tokenizer_path_or_name, num_classes)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        mask = {word for word in self.keywords[idx]}
        masked_sentence = " ".join(
            [
                "[MASK]" if word in mask else word
                for word in nltk.word_tokenize(self.sentences[idx])
            ]
        )
        tokenized_sentence = torch.tensor(
            self.tokenizer.encode(masked_sentence, add_special_tokens=True)
        )
        organ_indices = torch.tensor(self.organs_indices[idx])
        one_hot = torch.zeros(self.num_classes)
        one_hot[organ_indices] = 1

        return tokenized_sentence, one_hot


def collate_pad_sentence_class_batch(batch: Tuple[torch.Tensor, torch.Tensor]):
    sentences, organ_indices = zip(*batch)
    padded_sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)

    return padded_sentences, torch.stack([*organ_indices], dim=0)
