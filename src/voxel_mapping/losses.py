import torch
from torch import nn
import torch.nn.functional as F


class OrganDistanceLoss(nn.Module):
    def __init__(self, device, voxel_temperature, organ_temperature):
        super(OrganDistanceLoss, self).__init__()
        self.device = device
        self.voxel_temperature = voxel_temperature
        self.organ_temperature = organ_temperature

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
        distances_weights = F.softmin(distances_masked / self.voxel_temperature, dim=2)
        organ_distances_masked = (distances_masked * distances_weights).sum(dim=2)
        organ_distances_weights = F.softmin(
            organ_distances_masked / self.organ_temperature, dim=1
        )
        loss = (organ_distances_masked * organ_distances_weights).sum(dim=1).mean(dim=0)
        return loss


class BaselineRegLoss(nn.Module):
    def __init__(self):
        super(BaselineRegLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        return self.mse(predictions, anchors.view(3, -1, 3).mean(1))
