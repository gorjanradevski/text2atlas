# Adapted from: https://github.com/omoindrot/tensorflow-triplet-loss

import torch
from torch import nn


class BatchHard(nn.Module):
    def __init__(self, margin: float, device: str):
        """Build the batch-hard triplet loss over a batch of embeddings.

        Arguments:
            margin: margin for triplet loss.
            device: on which device to compute the loss.
        """
        super(BatchHard).__init__()
        self.margin = margin
        self.device = device

    def forward(self, labels: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """Computes the triplet loss using batch-hard mining.

        Arguments:
            labels: The labels for each of the embeddings.
            embeddings: The embeddings.

        Returns:
            Scalar loss containing the triplet loss.
        """
        # Get the distances
        pairwise_dist = torch.matmul(embeddings, embeddings.t())

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same
        # label)
        mask_anchor_positive = _get_anchor_positive_triplet_mask(
            labels, self.device
        ).float()

        # We put to 0 any element where (a, p) is not valid (valid if a != p and
        # label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have
        # different labels)
        mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

        # We add the maximum value in each row to the invalid negatives
        # (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
            1.0 - mask_anchor_negative
        )

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        tl = hardest_positive_dist - hardest_negative_dist + self.margin
        tl[tl < 0] = 0
        triplet_loss = tl.mean()

        return triplet_loss


class BatchAll(nn.Module):
    def __init__(self, margin: float, device: str):
        """Build the batch-all triplet loss over a batch of embeddings.

        Arguments:
            margin: margin for triplet loss
            device: on which device to compute the loss.
        """
        super(BatchAll).__init__()
        self.margin = margin
        self.device = device

    def forward(self, labels: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """Computes the triplet loss using batch-hard mining.

        Arguments:
            labels: The labels for each of the embeddings.
            embeddings: The embeddings.

        Returns:
            Scalar loss containing the triplet loss.

        """
        # Get the distance matrix
        pairwise_dist = torch.matmul(embeddings, embeddings.t())

        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j,
        # negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = _get_triplet_mask(labels)
        triplet_loss = mask.float() * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss[triplet_loss < 0] = 0

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss[triplet_loss > 1e-16]
        num_positive_triplets = valid_triplets.size(0)
        num_valid_triplets = mask.sum()

        fraction_positive_triplets = num_positive_triplets / (
            num_valid_triplets.float() + 1e-16
        )

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

        return triplet_loss, fraction_positive_triplets


def _get_triplet_mask(labels: torch.Tensor) -> torch.Tensor:
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:

        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: torch.int32 `Tensor` with shape [batch_size].

    Returns:
        The triplet mask.

    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0)).byte()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels & distinct_indices


def _get_anchor_positive_triplet_mask(labels: torch.Tensor, device: str):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same
    label.

    Args:
        labels: torch.int32 `Tensor` with shape [batch_size].
        device: The torch device.

    Returns:
        torch.bool `Tensor` with shape [batch_size, batch_size].

    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0)).byte().to(device)
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd
    # (batch_size, 1)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: torch.int32 `Tensor` with shape [batch_size]

    Returns:
        torch.bool `Tensor` with shape [batch_size, batch_size]

    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd
    # (batch_size, 1)
    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))
