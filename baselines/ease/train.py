"""Training and evaluation entry points for EASE."""

from typing import List

import torch
from torch import Tensor

from data.amazon import AmazonReviews
from evaluate.metrics import TopKAccumulator
from model import EASE
from utils import _build_user_item_matrix

DEFAULT_REG = 250.0
DEFAULT_REG_GRID = [100.0, 250.0, 500.0, 1000.0]
DEFAULT_KS = [5, 10, 50, 100]
EVAL_BATCH_SIZE = 1024


def train(user_item: Tensor, reg: float) -> EASE:
    """Fit an EASE model and return it.

    Parameters
    ----------
    user_item : Tensor
        Binary user-item interaction matrix of shape ``[U, I]``.
    reg : float
        L2 regularisation strength.

    Returns
    -------
    EASE
        Fitted model instance.
    """
    model = EASE(reg=reg)
    model.fit(user_item)
    return model


def evaluate(
    dataset: AmazonReviews,
    user_item: Tensor,
    model: EASE,
    split: str,
    ks: List[int],
) -> dict:
    """Score every evaluation user against the full catalogue and reduce metrics.

    Items seen during training are masked out before ranking so they cannot
    appear in the top-k predictions.

    Parameters
    ----------
    dataset : AmazonReviews
        Loaded dataset instance providing history and targets.
    user_item : Tensor
        Binary user-item interaction matrix built from the *train* split,
        shape ``[U, I]``.
    model : EASE
        Fitted EASE model used to generate scores.
    split : str
        Which evaluation split to use (``"eval"`` or ``"test"``).
    ks : List[int]
        Cut-off values for ranking metrics.

    Returns
    -------
    dict
        Aggregated ranking metrics produced by
        :class:`evaluate.metrics.TopKAccumulator`.
    """
    device = model.weights.device
    history = dataset[0]["user", "rated", "item"].history[split]
    item_ids: Tensor = history["itemId"]  # [U, seq_len]
    targets: Tensor = history["itemId_fut"]  # [U]

    accumulator = TopKAccumulator(ks=ks)

    for start in range(0, item_ids.size(0), EVAL_BATCH_SIZE):
        end = start + EVAL_BATCH_SIZE
        batch_history = item_ids[start:end].to(device)
        batch_targets = targets[start:end].to(device)

        scores = model.predict(user_item[start:end])

        seen = batch_history.clamp(min=0)
        scores.scatter_(1, seen, float("-inf"))

        target_scores = scores.gather(1, batch_targets.unsqueeze(1))
        ranks = (scores > target_scores).sum(dim=1).long()

        accumulator.accumulate_from_ranks(ranks)

    return accumulator.reduce()


def run(root: str, split: str, reg: float, ks: List[int], tune: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AmazonReviews(root=root, split=split)
    user_item = _build_user_item_matrix(dataset, split="train", device=device)

    num_items = dataset[0]["item"].x.size(0)
    print(f"[EASE] split={split} users={user_item.size(0)} items={num_items}")

    if tune:
        best_reg, best_score = None, -1.0
        for candidate in DEFAULT_REG_GRID:
            model = train(user_item, candidate)
            val_metrics = evaluate(dataset, user_item, model, "eval", ks)
            primary = val_metrics.get(f"ndcg@{max(ks)}", 0.0)

            metrics_str = "  ".join(
                f"{k}={v:.4f}" for k, v in sorted(val_metrics.items())
            )
            print(f"[EASE] reg={candidate:>7.1f} val: {metrics_str}")

            if primary > best_score:
                best_reg, best_score = candidate, primary

        reg = best_reg
        print(f"[EASE] best reg={reg} (val ndcg@{max(ks)}={best_score:.4f})")

    model = train(user_item, reg)
    val_metrics = evaluate(dataset, user_item, model, "eval", ks)
    test_metrics = evaluate(dataset, user_item, model, "test", ks)

    def fmt(m: dict) -> str:
        return "  ".join(f"{k}={v:.4f}" for k, v in sorted(m.items()))

    print(f"[EASE] reg={reg}")
    print(f"[EASE] VAL : {fmt(val_metrics)}")
    print(f"[EASE] TEST: {fmt(test_metrics)}")


def run_from_config(config: dict) -> None:
    run(
        root=config["root"],
        split=config["split"],
        reg=float(config.get("reg", DEFAULT_REG)),
        ks=list(config.get("ks", DEFAULT_KS)),
        tune=bool(config.get("tune", False)),
    )
