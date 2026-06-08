import torch
from torch import Tensor


class EASE:
    """Embarrassingly Shallow Autoencoder item-item model.

    Parameters
    ----------
    reg : float
        L2 regularisation strength applied to the diagonal of the Gram matrix.
    """

    def __init__(self, reg: float = 250.0) -> None:
        self.reg = reg
        self.weights: Tensor | None = None

    def fit(self, user_item: Tensor) -> "EASE":
        """Compute the closed-form EASE weight matrix ``B`` and cache it.

        Parameters
        ----------
        user_item : Tensor
            Binary user-item interaction matrix of shape ``[U, I]``.
        """
        gram = user_item.t() @ user_item
        idx = torch.arange(gram.size(0), device=gram.device)
        gram[idx, idx] += self.reg

        inv = torch.linalg.inv(gram)
        weights = inv / (-torch.diag(inv))
        weights[idx, idx] = 0.0

        self.weights = weights

        return self

    def predict(self, user_item: Tensor) -> Tensor:
        """Return predicted scores for all users.

        Parameters
        ----------
        user_item : Tensor
            Binary user-item interaction matrix of shape ``[U, I]``.

        Returns
        -------
        Tensor
            Score matrix of shape ``[U, I]``.
        """

        if self.weights is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        
        return user_item @ self.weights
