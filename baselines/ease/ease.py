"""
EASE -- Embarrassingly Shallow Autoencoder (Steck, 2019; arXiv:1905.03375).

A closed-form item-item linear model. Given the binary user-item matrix ``X``
(shape ``[U, I]``), it solves for a weight matrix ``B`` (shape ``[I, I]``) that
reconstructs ``X`` from itself under the constraint ``diag(B) = 0``:

    G = X^T X + reg * I
    P = G^{-1}
    B = -P / diag(P)           (off-diagonal)
    B_ii = 0

User scores are then ``S = X @ B``. The only hyperparameter is the L2
regularisation strength ``reg``, tuned on the validation split.
"""

from typing import List

import torch
from torch import Tensor

from baselines.data import AmazonSequenceData, build_seen_mask
from baselines.metrics import RankingMetrics, format_metrics

DEFAULT_REG = 250.0
DEFAULT_REG_GRID = [100.0, 250.0, 500.0, 1000.0]
DEFAULT_KS = [5, 10, 50, 100]
EVAL_BATCH_SIZE = 1024


def evaluate(
    data: AmazonSequenceData,
    user_item: Tensor,
    weights: Tensor,
    split: str,
    ks: List[int],
) -> dict:
    """Score every evaluation user against the full catalogue and reduce metrics."""
    examples = data.eval_examples(split)
    metrics = RankingMetrics(ks)
    device = weights.device

    for start in range(0, len(examples), EVAL_BATCH_SIZE):
        batch = examples[start : start + EVAL_BATCH_SIZE]
        scores = user_item[start : start + len(batch)] @ weights
        targets = torch.tensor([ex.target for ex in batch], device=device)
        seen_mask = build_seen_mask([ex.seen for ex in batch], data.num_items, device)
        metrics.accumulate(scores, targets, seen_mask)

    return metrics.reduce()


def train_ease(user_item: Tensor, reg: float) -> Tensor:
    """Compute the EASE item-item weight matrix ``B`` (closed form)."""
    gram = user_item.t() @ user_item
    idx = torch.arange(gram.size(0), device=gram.device)
    gram[idx, idx] += reg

    inv = torch.linalg.inv(gram)
    weights = inv / (-torch.diag(inv))
    weights[idx, idx] = 0.0

    return weights


def run(split: str, reg: float, ks: List[int], tune: bool) -> None:
    data = AmazonSequenceData(split=split)
    user_item = data.user_item_matrix()

    print(f"[EASE] split={split} users={user_item.size(0)} items={data.num_items}")

    if tune:
        best_reg, best_score = None, -1.0
        for candidate in DEFAULT_REG_GRID:
            weights = train_ease(user_item, candidate)
            val_metrics = evaluate(data, user_item, weights, "val", ks)
            primary = val_metrics[f"ndcg@{max(ks)}"]

            print(f"[EASE] reg={candidate:>7.1f} val: {format_metrics(val_metrics)}")

            if primary > best_score:
                best_reg, best_score = candidate, primary

        reg = best_reg

        print(f"[EASE] best reg={reg} (val ndcg@{max(ks)}={best_score:.4f})")

    weights = train_ease(user_item, reg)
    val_metrics = evaluate(data, user_item, weights, "val", ks)
    test_metrics = evaluate(data, user_item, weights, "test", ks)

    print(f"[EASE] reg={reg}")
    print(f"[EASE] VAL : {format_metrics(val_metrics)}")
    print(f"[EASE] TEST: {format_metrics(test_metrics)}")


def run_from_config(config: dict) -> None:
    run(
        split=config["split"],
        reg=float(config.get("reg", DEFAULT_REG)),
        ks=list(config.get("ks", DEFAULT_KS)),
        tune=bool(config.get("tune", False)),
    )
