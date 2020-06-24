from transformers import BertTokenizer
from torch.utils.data import Dataset
import json
import torch
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from typing import Tuple, List
from tqdm import tqdm
from typing import Dict
from utils.constants import VOXELMAN_CENTER


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
            self.sentences.append(element["text"])
            if ind2anchors:
                self.mappings.append(
                    [ind2anchors[ind] for ind in element["organ_indices"]]
                )
            else:
                self.mappings.append(element["centers"])
            self.keywords.append(element["keywords"])
            self.organs_indices.append(element["organ_indices"])
        self.center = torch.from_numpy(VOXELMAN_CENTER)


class VoxelSentenceMappingTrainRegDataset(VoxelSentenceMappingRegDataset, Dataset):
    def __init__(
        self,
        json_path: str,
        tokenizer: BertTokenizer,
        organs_list: List[str],
        ind2anchors: Dict,
        masking: bool,
    ):
        super().__init__(json_path, tokenizer, ind2anchors)
        self.organs_list = organs_list
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
        tokenized_sentence = torch.tensor(self.tokenizer.encode(sentence))
        mapping = torch.tensor(self.mappings[idx]) / self.center
        num_organs = len(mapping)

        return tokenized_sentence, mapping, num_organs


class VoxelSentenceMappingTestRegDataset(VoxelSentenceMappingRegDataset, Dataset):
    def __init__(self, json_path: str, tokenizer: BertTokenizer, ind2anchors: Dict):
        super().__init__(json_path, tokenizer, ind2anchors)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        tokenized_sentence = torch.tensor(self.tokenizer.encode(self.sentences[idx]))
        organ_indices = torch.tensor(self.organs_indices[idx])

        return tokenized_sentence, organ_indices


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


def collate_pad_sentence_reg_test_batch(batch: Tuple[torch.Tensor, torch.Tensor]):
    sentences, organ_indices = zip(*batch)
    padded_sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)
    padded_organ_indices = torch.nn.utils.rnn.pad_sequence(
        organ_indices, batch_first=True, padding_value=-1
    )
    attn_mask = padded_sentences.clone()
    attn_mask[torch.where(attn_mask > 0)] = 1

    return padded_sentences, attn_mask, padded_organ_indices


class VoxelSentenceMappingClassDataset:
    def __init__(self, json_path: str, tokenizer: BertTokenizer, num_classes: int):
        self.json_data = json.load(open(json_path))
        self.sentences, self.organs_indices, self.keywords = [], [], []
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        for element in tqdm(self.json_data):
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
        masking: bool,
    ):
        super().__init__(json_path, tokenizer, num_classes)
        self.organs_list = organs_list
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
        tokenized_sentence = torch.tensor(self.tokenizer.encode(sentence))
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


def collate_pad_sentence_class_batch(batch: Tuple[torch.Tensor, torch.Tensor]):
    sentences, organ_indices = zip(*batch)
    padded_sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)
    attn_mask = padded_sentences.clone()
    attn_mask[torch.where(attn_mask > 0)] = 1

    return padded_sentences, attn_mask, torch.stack([*organ_indices], dim=0)
