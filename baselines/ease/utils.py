import torch
from torch import Tensor
from data.amazon import AmazonReviews


def _build_user_item_matrix(
    dataset: AmazonReviews, split: str, device: torch.device
) -> Tensor:
    """Build a binary user-item interaction matrix from the train history.

    Parameters
    ----------
    dataset : AmazonReviews
        Loaded dataset instance.
    split : str
        Which sequence split to use when populating the matrix
        (typically ``"train"``).
    device : torch.device
        Target device for the returned tensor.

    Returns
    -------
    Tensor
        Float32 binary matrix of shape ``[U, I]``.
    """
    history = dataset[0]["user", "rated", "item"].history[split]
    item_ids: Tensor = history["itemId"]  # [U, seq_len], padded with -1
    num_users = item_ids.size(0)
    num_items = dataset[0]["item"].x.size(0)

    matrix = torch.zeros(num_users, num_items, device=device)
    for u, seq in enumerate(item_ids):
        valid = seq[seq >= 0]
        matrix[u, valid] = 1.0
    return matrix
