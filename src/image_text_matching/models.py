import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet152
from transformers import BertModel

from typing import Tuple


class L2Normalize(nn.Module):
    def __init__(self):
        super(L2Normalize, self).__init__()

    def forward(self, x) -> torch.Tensor:
        norm = torch.pow(x, 2).sum(dim=1, keepdim=True).sqrt()
        normalized = torch.div(x, norm)

        return normalized


class ImageEncoder(nn.Module):
    def __init__(self, finetune: bool):
        super(ImageEncoder, self).__init__()
        self.resnet = torch.nn.Sequential(
            *(list(resnet152(pretrained=True).children())[:-1])
        )

        for param in self.resnet.parameters():
            param.requires_grad = finetune

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embedded_images = torch.flatten(self.resnet(images), start_dim=1)

        return embedded_images


class SentenceEncoder(nn.Module):
    def __init__(self, finetune: bool, bert_path_or_name: str):
        super(SentenceEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path_or_name)
        #  https://arxiv.org/abs/1801.06146

        for param in self.bert.parameters():
            param.requires_grad = finetune

    def forward(self, sentences: torch.Tensor):
        # https://arxiv.org/abs/1801.06146
        hidden_states = self.bert(sentences)
        max_pooled = torch.max(hidden_states[0], dim=1)[0]
        mean_pooled = torch.mean(hidden_states[0], dim=1)
        last_state = hidden_states[0][:, 0, :]
        embedded_sentences = torch.cat([last_state, max_pooled, mean_pooled], dim=1)

        return embedded_sentences


class Projector(nn.Module):
    def __init__(self, input_space, joint_space: int):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(input_space, joint_space)
        self.bn = nn.BatchNorm1d(joint_space)
        self.fc2 = nn.Linear(joint_space, joint_space)
        self.l2_normalize = L2Normalize()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        projected_embeddings = self.fc2(self.bn(F.relu(self.fc1(embeddings))))

        return self.l2_normalize(projected_embeddings)


class ImageEmbeddingTextEmbeddingMatchingModel(nn.Module):
    def __init__(self, image_space: int, sentence_space: int, joint_space: int):
        super(ImageEmbeddingTextEmbeddingMatchingModel, self).__init__()
        self.image_projector = Projector(image_space, joint_space)
        self.sentence_projector = Projector(sentence_space, joint_space)

    def forward(
        self, images: torch.Tensor, sentences: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        return (self.image_projector(images), self.sentence_projector(sentences))


class ImageTextMatchingModel(nn.Module):
    def __init__(
        self, bert_path_or_name: str, joint_space: int, finetune: bool = False
    ):
        super(ImageTextMatchingModel, self).__init__()
        self.finetune = finetune
        # Image encoder
        self.image_encoder = ImageEncoder(finetune)
        self.image_encoder.eval()
        self.image_projector = Projector(2048, joint_space)
        # Sentence encoder
        self.sentence_encoder = SentenceEncoder(finetune, bert_path_or_name)
        self.sentence_encoder.eval()
        self.sentence_projector = Projector(768 * 3, joint_space)

    def forward(self, images: torch.Tensor, sentences: torch.Tensor):
        embedded_images = self.image_encoder(images)
        embedded_sentences = self.sentence_encoder(sentences)

        return (
            self.image_projector(embedded_images),
            self.sentence_projector(embedded_sentences),
        )

    def train(self, mode: bool = True):
        if self.finetune and mode:
            self.image_encoder.train()
            self.sentence_encoder.train()
            self.image_projector.train(True)
            self.sentence_projector.train(True)
        elif mode:
            self.image_projector.train(True)
            self.sentence_projector.train(True)
        else:
            self.image_encoder.train(False)
            self.sentence_encoder.train(False)
            self.image_projector.train(False)
            self.sentence_projector.train(False)
