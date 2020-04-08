from transformers import BertModel, BertConfig
from torch import nn
import torch
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceMappingsProducer(nn.Module):
    def __init__(
        self,
        bert_path_or_name: str,
        joint_space: int,
        config: BertConfig,
        reg_or_class: str = "reg",
        num_classes: int = 46,
    ):
        # emilyalsentzer/Bio_ClinicalBERT (https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
        # SciBERT
        # BioBERT
        # google/bert_uncased_L-2_H-128_A-2 (https://huggingface.co/google)
        super(SentenceMappingsProducer, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path_or_name)
        if reg_or_class == "reg":
            self.projector = RegressionProjector(config.hidden_size, joint_space)
        elif reg_or_class == "class":
            self.projector = ClassificationProjector(
                config.hidden_size, joint_space, num_classes
            )
        else:
            raise ValueError("The projector can be regression or classification.")

    def forward(self, sentences: torch.Tensor):
        # https://arxiv.org/abs/1801.06146
        hidden_states = self.bert(sentences)
        last_state = hidden_states[0][:, 0, :]

        return self.projector(last_state)


class RegressionProjector(nn.Module):
    def __init__(self, input_space, joint_space: int):
        super(RegressionProjector, self).__init__()
        self.fc1 = nn.Linear(input_space, joint_space)
        self.bn = nn.BatchNorm1d(joint_space)
        self.fc2 = nn.Linear(joint_space, 3)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        projected_embeddings = self.fc2(self.bn(F.relu(self.fc1(embeddings))))

        return projected_embeddings


class ClassificationProjector(nn.Module):
    def __init__(self, input_space, joint_space: int, num_classes):
        super(ClassificationProjector, self).__init__()
        self.fc1 = nn.Linear(input_space, joint_space)
        self.bn = nn.BatchNorm1d(joint_space)
        self.fc2 = nn.Linear(joint_space, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        projected_embeddings = self.fc2(self.bn(F.relu(self.fc1(embeddings))))

        return projected_embeddings
