import torch
from torch import nn
from transformers import BertConfig, BertModel

# https://huggingface.co/allenai/scibert_scivocab_uncased
# https://huggingface.co/monologg/biobert_v1.1_pubmed
# https://huggingface.co/google


class RegModel(nn.Module):
    def __init__(self, bert_name: str, config: BertConfig, final_project_size: int):
        super(RegModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.projector = nn.Linear(config.hidden_size, final_project_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return torch.tanh(self.projector(hidden_states[0][:, 0, :]))


class ClassModel(nn.Module):
    def __init__(self, bert_name: str, config: BertConfig, final_project_size: int):
        super(ClassModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.projector = nn.Linear(config.hidden_size, final_project_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.projector(hidden_states[0][:, 0, :])


class L2Normalize(nn.Module):
    def __init__(self):
        super(L2Normalize, self).__init__()

    def forward(self, x):
        norm = torch.pow(x, 2).sum(dim=1, keepdim=True).sqrt()
        normalized = torch.div(x, norm)

        return normalized


class SiameseModel(nn.Module):
    def __init__(self, bert_name: str, config: BertConfig, final_project_size: int):
        super(SiameseModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.projector = nn.Linear(config.hidden_size, final_project_size)
        self.l2_normalizer = L2Normalize()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.l2_normalizer(self.projector(hidden_states[0][:, 0, :]))


class OnlyPretrainedBert(nn.Module):
    def __init__(self, bert_name: str):
        super(OnlyPretrainedBert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.l2_normalizer = L2Normalize()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        max_pooled = torch.max(hidden_states[0], dim=1)[0]
        mean_pooled = torch.mean(hidden_states[0], dim=1)
        last_state = hidden_states[0][:, 0, :]
        return self.l2_normalizer(
            torch.cat([last_state, max_pooled, mean_pooled], dim=1)
        )


def model_factory(
    model_name: str, bert_name: str, config: BertConfig, final_project_size: int
):
    if model_name == "reg_model":
        print(f"Using {model_name}!")
        return RegModel(bert_name, config, final_project_size)
    elif model_name == "class_model":
        print(f"Using {model_name}!")
        return ClassModel(bert_name, config, final_project_size)
    elif model_name == "siamese_model":
        print(f"Using {model_name}!")
        return SiameseModel(bert_name, config, final_project_size)
    elif model_name == "pretrained_model":
        print(f"Using {model_name}!")
        return OnlyPretrainedBert(bert_name)

    else:
        raise ValueError(f"Invalid {model_name}!")
