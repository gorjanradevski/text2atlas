from transformers import BertModel, BertConfig
from transformers import BertOnlyMLMHead
from torch import nn
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceMappingsProducer(nn.Module):
    def __init__(
        self, bert_path_or_name: str, config: BertConfig, reg_or_class: str = "reg"
    ):
        # emilyalsentzer/Bio_ClinicalBERT (https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
        # SciBERT
        # BioBERT
        # google/bert_uncased_L-2_H-128_A-2 (https://huggingface.co/google)
        super(SentenceMappingsProducer, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path_or_name)
        if reg_or_class == "reg":
            config.vocab_size = 3
        elif reg_or_class == "class":
            config.vocab_size = 46
        else:
            raise ValueError("The projector can be regression or classification.")
        self.projector = BertOnlyMLMHead(config)

    def forward(self, sentences: torch.Tensor):
        hidden_states = self.bert(sentences)
        last_state = hidden_states[0][:, 0, :]

        return self.projector(last_state)
