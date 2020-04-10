from transformers import BertTokenizer
from torch.utils.data import Dataset
import json
import torch
from typing import Tuple, List
import numpy as np
import nltk
from tqdm import tqdm
from typing import Dict


class VoxelSentenceMappingRegDataset:
    def __init__(self, json_path: str, tokenizer: str, ind2anchors: Dict):
        self.json_data = json.load(open(json_path))
        self.tokenizer = tokenizer
        self.sentences, self.mappings, self.keywords, self.organs_indices = (
            [],
            [],
            [],
            [],
        )
        for element in tqdm(self.json_data):
            if len(self.tokenizer.encode(element["text"])) > 512:
                continue
            self.sentences.append(element["text"])
            if ind2anchors:
                self.mappings.append(
                    [ind2anchors[ind] for ind in element["organ_indices"]]
                )
            else:
                self.mappings.append(element["centers"])
            self.keywords.append(element["keywords"])
            self.organs_indices.append(element["organ_indices"])


class VoxelSentenceMappingTrainRegDataset(VoxelSentenceMappingRegDataset, Dataset):
    def __init__(
        self,
        json_path: str,
        tokenizer: BertTokenizer,
        organs_list: List[str],
        ind2anchors: Dict = None,
    ):
        super().__init__(json_path, tokenizer, ind2anchors)
        self.organs_list = organs_list

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        # 0 - [MASK], 1 - keep word, 2 - random word from list
        mask = {
            word: np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
            for word in self.keywords[idx]
        }
        masked_sentence = " ".join(
            [
                "[MASK]"
                if word in mask and mask[word] == 0
                else np.random.choice(self.organs_list)
                if word in mask and mask[word] == 2
                else word
                for word in nltk.word_tokenize(self.sentences[idx])
            ]
        )
        tokenized_sentence = torch.tensor(self.tokenizer.encode(masked_sentence))
        mapping = torch.tensor(self.mappings[idx])
        organ_indices = torch.tensor(self.organs_indices[idx])
        num_organs = len(mapping)

        return tokenized_sentence, mapping, num_organs, organ_indices


class VoxelSentenceMappingTestRegDataset(VoxelSentenceMappingRegDataset, Dataset):
    def __init__(
        self, json_path: str, tokenizer: BertTokenizer, ind2anchors: Dict = None
    ):
        super().__init__(json_path, tokenizer, ind2anchors)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        tokenized_sentence = torch.tensor(self.tokenizer.encode(self.sentences[idx]))
        organ_indices = torch.tensor(self.organs_indices[idx])

        return tokenized_sentence, organ_indices


class VoxelSentenceMappingTestMaskedRegDataset(VoxelSentenceMappingRegDataset, Dataset):
    def __init__(
        self, json_path: str, tokenizer: BertTokenizer, ind2anchors: Dict = None
    ):
        super().__init__(json_path, tokenizer, ind2anchors)

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
        tokenized_sentence = torch.tensor(self.tokenizer.encode(masked_sentence))
        organ_indices = torch.tensor(self.organs_indices[idx])

        return tokenized_sentence, organ_indices


def collate_pad_sentence_reg_train_batch(
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


def collate_pad_sentence_reg_test_batch(batch: Tuple[torch.Tensor, torch.Tensor]):
    sentences, organ_indices = zip(*batch)
    padded_sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)
    padded_organ_indices = torch.nn.utils.rnn.pad_sequence(
        organ_indices, batch_first=True, padding_value=-1
    )

    return padded_sentences, padded_organ_indices


class VoxelSentenceMappingClassDataset:
    def __init__(self, json_path: str, tokenizer: BertTokenizer, num_classes: int):
        self.json_data = json.load(open(json_path))
        self.sentences, self.organs_indices, self.keywords = [], [], []
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        for element in tqdm(self.json_data):
            if len(self.tokenizer.encode(element["text"])) > 512:
                continue
            self.sentences.append(element["text"])
            self.organs_indices.append(element["organ_indices"])
            self.keywords.append(element["keywords"])


class VoxelSentenceMappingTrainClassDataset(VoxelSentenceMappingClassDataset, Dataset):
    def __init__(
        self,
        json_path: str,
        tokenizer: BertTokenizer,
        num_classes: int,
        organs_list: List[str],
    ):
        super().__init__(json_path, tokenizer, num_classes)
        self.organs_list = organs_list

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        # 0 - [MASK], 1 - keep word, 2 - random word from list
        mask = {
            word: np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
            for word in self.keywords[idx]
        }
        masked_sentence = " ".join(
            [
                "[MASK]"
                if word in mask and mask[word] == 0
                else np.random.choice(self.organs_list)
                if word in mask and mask[word] == 2
                else word
                for word in nltk.word_tokenize(self.sentences[idx])
            ]
        )
        tokenized_sentence = torch.tensor(self.tokenizer.encode(masked_sentence))
        organ_indices = torch.tensor(self.organs_indices[idx])
        one_hot = torch.zeros(self.num_classes)
        one_hot[organ_indices] = 1

        return tokenized_sentence, one_hot


class VoxelSentenceMappingTestClassDataset(VoxelSentenceMappingClassDataset, Dataset):
    def __init__(self, json_path: str, tokenizer: BertTokenizer, num_classes: int):
        super().__init__(json_path, tokenizer, num_classes)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        tokenized_sentence = torch.tensor(self.tokenizer.encode(self.sentences[idx]))
        organ_indices = torch.tensor(self.organs_indices[idx])
        one_hot = torch.zeros(self.num_classes)
        one_hot[organ_indices] = 1

        return tokenized_sentence, one_hot


class VoxelSentenceMappingTestMaskedClassDataset(
    VoxelSentenceMappingClassDataset, Dataset
):
    def __init__(self, json_path: str, tokenizer: BertTokenizer, num_classes: int):
        super().__init__(json_path, tokenizer, num_classes)

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
        tokenized_sentence = torch.tensor(self.tokenizer.encode(masked_sentence))
        organ_indices = torch.tensor(self.organs_indices[idx])
        one_hot = torch.zeros(self.num_classes)
        one_hot[organ_indices] = 1

        return tokenized_sentence, one_hot


def collate_pad_sentence_class_batch(batch: Tuple[torch.Tensor, torch.Tensor]):
    sentences, organ_indices = zip(*batch)
    padded_sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)

    return padded_sentences, torch.stack([*organ_indices], dim=0)
