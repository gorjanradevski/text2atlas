import torch
from torch import nn
import torch.nn.functional as F


class MinDistanceLoss(nn.Module):
    def __init__(self):
        super(MinDistanceLoss, self).__init__()

    def forward(
        self, predictions: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor
    ):
        """Computes the minimum distance to organ loss.

        Arguments:
            predictions: Tensor with shape [batch_size, 3]
            labels: Tensor with shape [batch_size, num_keywords, 3]
            lengths: Tensor with shape [batch_size]
        """
        mask = (
            torch.arange(torch.max(lengths)).expand(len(lengths), torch.max(lengths))
            < lengths.unsqueeze(1)
        ).float()
        mask[torch.where(mask == 0)] = 1e15
        predictions = predictions.unsqueeze(1)
        loss = (predictions - labels).norm(p=2, dim=2)
        loss_masked = loss * mask
        loss_softmined = F.softmin(loss_masked, dim=1)

        return torch.mean(torch.sum(loss_masked * loss_softmined, dim=1))
