import torch
from torch import nn
import torch.nn.functional as F


class OrganDistanceLoss(nn.Module):
    def __init__(self, device):
        super(OrganDistanceLoss, self).__init__()
        self.device = device

    def forward(
        self, predictions: torch.Tensor, anchors: torch.Tensor, lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the minimum distance to organ loss.

        Arguments:
            predictions: Tensor with shape [batch_size, 3]
            anchors: Tensor with shape [batch_size, max_organs_in_batch, num_sampled_points, 3]
            lengths: Tensor with shape [batch_size]
            devce: A torch device - either cpu or gpu
        """
        mask = (
            torch.arange(torch.max(lengths))
            .expand(lengths.size()[0], torch.max(lengths))
            .to(self.device)
            < lengths.unsqueeze(1)
        ).float()
        mask[torch.where(mask == 0)] = 1e15
        mask = mask.unsqueeze(2)
        predictions = predictions.unsqueeze(1).unsqueeze(2)
        distances = (predictions - anchors).norm(p=2, dim=3)
        distances_masked = distances * mask
        distances_weights = F.softmin(distances_masked, dim=2)
        organ_distances_masked = (distances_masked * distances_weights).sum(dim=2)
        organ_distances_weights = F.softmin(organ_distances_masked, dim=1)
        loss = (organ_distances_masked * organ_distances_weights).sum(dim=1).mean(dim=0)
        return loss


class MinDistanceLoss(nn.Module):
    def __init__(self, device):
        self.device = device
        super(MinDistanceLoss, self).__init__()

    def forward(
        self, predictions: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor,
    ):
        """Computes the minimum distance to organ loss.

        Arguments:
            predictions: Tensor with shape [batch_size, 3]
            labels: Tensor with shape [batch_size, max_organs_in_batch, 3]
            lengths: Tensor with shape [batch_size]
            devce: A torch device - either cpu or gpu
        """
        mask = (
            torch.arange(torch.max(lengths))
            .expand(lengths.size()[0], torch.max(lengths))
            .to(self.device)
            < lengths.unsqueeze(1)
        ).float()
        mask[torch.where(mask == 0)] = 1e15
        predictions = predictions.unsqueeze(1)
        loss = (predictions - labels).norm(p=2, dim=2)
        loss_masked = loss * mask
        softmin_weights = F.softmin(loss_masked, dim=1)

        return (loss_masked * softmin_weights).sum(-1).mean()
