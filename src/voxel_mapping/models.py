from transformers import BertModel, BertConfig
from torch import nn
import torch


class SentenceMappingsProducer(nn.Module):
    def __init__(self, bert_name: str, config: BertConfig, final_project_size: int):
        # https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
        # https://huggingface.co/allenai/scibert_scivocab_uncased
        # https://huggingface.co/monologg/biobert_v1.1_pubmed
        # https://huggingface.co/google
        super(SentenceMappingsProducer, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.projector = nn.Linear(config.hidden_size, final_project_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_state = hidden_states[0][:, 0, :]

        return self.projector(last_state)
