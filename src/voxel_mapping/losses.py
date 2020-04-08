import torch
from torch import nn
import torch.nn.functional as F


class OrganDistanceLoss(nn.Module):
    def __init__(self):
        super(OrganDistanceLoss, self).__init__()

    def forward(
        self,
        predictions: torch.Tensor,
        anchors: torch.Tensor,
        lengths: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Computes the minimum distance to organ loss.

        Arguments:
            predictions: Tensor with shape [batch_size, 3]
            anchors: Tensor with shape [batch_size, num_keywords, num_anchor_points, 3]
            lengths: Tensor with shape [batch_size]
            device: torch.device
        """
        if len(anchors.shape) < 4:
            anchors = anchors.unsqueeze(1)

        mask = (
            torch.arange(torch.max(lengths).item())
            .expand(lengths.shape[0], torch.max(lengths).item())
            .to(device)
            < lengths.unsqueeze(1)
        ).float()
        mask[torch.where(mask == 0)] = 1e15
        mask = mask.unsqueeze(2)
        predictions = predictions.unsqueeze(1).unsqueeze(2)
        distances = (predictions - anchors).norm(p=2, dim=3)
        distances_masked = distances * mask
        distances_weights = F.softmin(distances_masked, dim=2)
        organ_distances_masked = torch.sum(distances_masked * distances_weights, dim=2)
        organ_distances_weights = F.softmin(organ_distances_masked, dim=1)
        loss = torch.sum(organ_distances_masked * organ_distances_weights, dim=1)
        loss = torch.mean(loss, dim=0)
        return loss
