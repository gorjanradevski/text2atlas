from transformers import BertTokenizer
from torch.utils.data import Dataset
import json
import torch
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from typing import Tuple, Dict, List
from tqdm import tqdm
import random
from utils.constants import VOXELMAN_CENTER


class VoxelSentenceMappingRegDataset:
    def __init__(self, json_path: str, tokenizer: str):
        self.json_data = json.load(open(json_path))
        self.tokenizer = tokenizer
        self.sentences = [element["text"] for element in tqdm(self.json_data)]
        self.center = torch.from_numpy(VOXELMAN_CENTER)


class VoxelSentenceMappingTrainRegDataset(VoxelSentenceMappingRegDataset, Dataset):
    def __init__(
        self,
        json_path: str,
        tokenizer: BertTokenizer,
        ind2organ: Dict[int, str],
        organ2voxels: str,
        num_anchors: int,
        masking: bool,
        use_occurences: bool,
    ):
        super().__init__(json_path, tokenizer)
        self.organ_indices = [
            element["occ_organ_indices"] if use_occurences else element["organ_indices"]
            for element in tqdm(self.json_data)
        ]
        self.keywords = [element["keywords"] for element in self.json_data]
        self.masking = masking
        self.detokenizer = TreebankWordDetokenizer()
        self.num_anchors = num_anchors
        self.organ2voxels = organ2voxels
        self.ind2organ = ind2organ

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        sentence = self.sentences[idx]
        if self.masking:
            mask = {
                word: torch.bernoulli(torch.tensor([0.5])).bool().item()
                for word in self.keywords[idx]
            }
            sentence = self.detokenizer.detokenize(
                [
                    "[MASK]" if word in mask and mask[word] else word
                    for word in word_tokenize(self.sentences[idx])
                ]
            )
        tokenized_sentence = truncate_sentence(self.tokenizer.encode(sentence))
        # Obtain mapping
        mapping = (
            torch.tensor(
                [
                    random.sample(
                        self.organ2voxels[self.ind2organ[str(index)]], self.num_anchors
                    )
                    for index in self.organ_indices[idx]
                ]
            )
            / self.center
        )
        num_organs = len(mapping)

        return tokenized_sentence, mapping, num_organs


class VoxelSentenceMappingTestRegDataset(VoxelSentenceMappingRegDataset, Dataset):
    def __init__(self, json_path: str, tokenizer: BertTokenizer):
        super().__init__(json_path, tokenizer)
        self.organ_indices = [
            element["organ_indices"] for element in tqdm(self.json_data)
        ]
        self.ids = [element["pmid"] for element in self.json_data]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        tokenized_sentence = truncate_sentence(
            self.tokenizer.encode(self.sentences[idx])
        )
        organ_indices = torch.tensor(self.organ_indices[idx])
        doc_ids = self.ids[idx]

        return tokenized_sentence, organ_indices, doc_ids


def collate_pad_sentence_reg_train_batch(
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    sentences, mappings, num_organs = zip(*batch)
    padded_sentences = torch.nn.utils.rnn.pad_sequence(
        sentences, batch_first=True, padding_value=0
    )
    padded_mappings = torch.nn.utils.rnn.pad_sequence(
        mappings, batch_first=True, padding_value=0
    )
    num_organs = torch.tensor([*num_organs])
    attn_mask = padded_sentences.clone()
    attn_mask[torch.where(attn_mask > 0)] = 1

    return padded_sentences, attn_mask, padded_mappings, num_organs


def collate_pad_sentence_reg_test_batch(
    batch: Tuple[torch.Tensor, torch.Tensor, List[int]]
):
    sentences, organ_indices, docs_ids = zip(*batch)
    padded_sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)
    padded_organ_indices = torch.nn.utils.rnn.pad_sequence(
        organ_indices, batch_first=True, padding_value=-1
    )
    attn_mask = padded_sentences.clone()
    attn_mask[torch.where(attn_mask > 0)] = 1

    return padded_sentences, attn_mask, padded_organ_indices, docs_ids


class VoxelSentenceMappingClassDataset:
    def __init__(self, json_path: str, tokenizer: BertTokenizer, num_classes: int):
        self.json_data = json.load(open(json_path))
        self.sentences = [element["text"] for element in tqdm(self.json_data)]
        self.num_classes = num_classes
        self.tokenizer = tokenizer


class VoxelSentenceMappingTrainClassDataset(VoxelSentenceMappingClassDataset, Dataset):
    def __init__(
        self,
        json_path: str,
        tokenizer: BertTokenizer,
        num_classes: int,
        masking: bool,
        use_occurences: bool,
    ):
        super().__init__(json_path, tokenizer, num_classes)
        self.keywords = [element["keywords"] for element in self.json_data]
        self.organ_indices = [
            element["occ_organ_indices"] if use_occurences else element["organ_indices"]
            for element in tqdm(self.json_data)
        ]
        self.masking = masking
        self.detokenizer = TreebankWordDetokenizer()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        sentence = self.sentences[idx]
        if self.masking:
            mask = {
                word: torch.bernoulli(torch.tensor([0.5])).bool().item()
                for word in self.keywords[idx]
            }
            sentence = self.detokenizer.detokenize(
                [
                    "[MASK]" if word in mask and mask[word] else word
                    for word in word_tokenize(self.sentences[idx])
                ]
            )
        tokenized_sentence = truncate_sentence(self.tokenizer.encode(sentence))
        organ_indices = torch.tensor(self.organ_indices[idx])
        one_hot = torch.zeros(self.num_classes)
        one_hot[organ_indices] = 1
        doc_ids = self.ids[idx]

        return tokenized_sentence, one_hot, doc_ids


class VoxelSentenceMappingTestClassDataset(VoxelSentenceMappingClassDataset, Dataset):
    def __init__(self, json_path: str, tokenizer: BertTokenizer, num_classes: int):
        super().__init__(json_path, tokenizer, num_classes)
        self.organ_indices = [
            element["organ_indices"] for element in tqdm(self.json_data)
        ]
        self.ids = [element["pmid"] for element in self.json_data]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        tokenized_sentence = truncate_sentence(
            self.tokenizer.encode(self.sentences[idx])
        )
        organ_indices = torch.tensor(self.organ_indices[idx])
        one_hot = torch.zeros(self.num_classes)
        one_hot[organ_indices] = 1
        doc_ids = self.ids[idx]

        return tokenized_sentence, one_hot, doc_ids


def collate_pad_sentence_class_batch(
    batch: Tuple[torch.Tensor, torch.Tensor, List[int]]
):
    sentences, organ_indices, docs_ids = zip(*batch)
    padded_sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)
    attn_mask = padded_sentences.clone()
    attn_mask[torch.where(attn_mask > 0)] = 1

    return padded_sentences, attn_mask, torch.stack([*organ_indices], dim=0), docs_ids


def truncate_sentence(tokenized_sentence: List[int], max_len: int = 512):
    if len(tokenized_sentence) > max_len:
        tokenized_sentence = tokenized_sentence[: max_len - 1] + tokenized_sentence[-1:]
    return torch.tensor(tokenized_sentence)
