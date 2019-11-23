import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet152

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


class ImageTextMatchingModel(nn.Module):
    def __init__(self, image_space: int, sentence_space: int, joint_space: int):
        super(ImageTextMatchingModel, self).__init__()
        self.image_projector = Projector(image_space, joint_space)
        self.sentence_projector = Projector(sentence_space, joint_space)

    def forward(
        self, images: torch.Tensor, sentences: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        return (self.image_projector(images), self.sentence_projector(sentences))
