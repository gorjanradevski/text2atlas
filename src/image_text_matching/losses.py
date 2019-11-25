import torch
from torch import nn


class BatchAll(nn.Module):
    # Adapted from: https://github.com/omoindrot/tensorflow-triplet-loss

    def __init__(self, margin: float, device: str):
        """Build the batch-all triplet loss over a batch of embeddings.
        Arguments:
            margin: margin for triplet loss
            device: on which device to compute the loss.
        """
        super(BatchAll, self).__init__()
        self.margin = margin
        self.device = device

    def forward(
        self, labels: torch.Tensor, image_embeddings: torch.Tensor, sentence_embeddings
    ) -> torch.Tensor:
        """Computes the triplet loss using batch-hard mining.
        Arguments:
            labels: The labels for each of the embeddings.
            embeddings: The embeddings.
        Returns:
            Scalar loss containing the triplet loss.
        """
        # Get the distance matrix
        pairwise_dist = torch.matmul(image_embeddings, sentence_embeddings.t())

        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j,
        # negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = self.margin + anchor_negative_dist - anchor_positive_dist

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = _get_triplet_mask(labels).float()
        triplet_loss = mask * triplet_loss

        # Remove negative losses (i.e. the easy triplets) and sum
        return triplet_loss.clamp(min=0).sum()


def _get_triplet_mask(labels: torch.Tensor) -> torch.Tensor:
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:

        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: torch.int32 `Tensor` with shape [batch_size].

    Returns:
        The triplet mask.

    """
    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels
