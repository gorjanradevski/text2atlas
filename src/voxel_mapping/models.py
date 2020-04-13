from transformers import BertModel, BertConfig
from transformers import BertOnlyMLMHead
from torch import nn
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceMappingsProducer(nn.Module):
    def __init__(self, bert_name: str, config: BertConfig, final_project_size: int):
        # https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
        # https://huggingface.co/allenai/scibert_scivocab_uncased
        # https://huggingface.co/monologg/biobert_v1.1_pubmed
        # https://huggingface.co/google
        super(SentenceMappingsProducer, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        config.vocab_size = final_project_size
        self.projector = BertOnlyMLMHead(config)

    def forward(self, sentences: torch.Tensor):
        hidden_states = self.bert(sentences)
        last_state = hidden_states[0][:, 0, :]

        return self.projector(last_state)
