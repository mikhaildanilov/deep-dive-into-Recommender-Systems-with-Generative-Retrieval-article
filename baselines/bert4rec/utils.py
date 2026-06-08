"""Shared utilities for sequential baselines (SASRec, BERT4Rec).

Provides left/right padding of variable-length item sequences into fixed-length
tensors and a common full-catalogue evaluation loop. Item ids stay in their
original ``[0, num_items)`` range; the padding slot uses a dedicated id equal to
``num_items`` so embeddings are sized ``num_items + 1`` with
``padding_idx=num_items``.
"""

from typing import List, Tuple

import torch
from torch import Tensor

from baselines.data import AmazonSequenceData, build_seen_mask
from baselines.metrics import RankingMetrics, format_metrics


def pad_id(num_items: int) -> int:
    """Dedicated padding item id (one past the valid range)."""
    return num_items


def left_pad(seq: List[int], max_len: int, pad: int) -> List[int]:
    """Truncate to the most recent ``max_len`` items and left-pad with ``pad``."""
    seq = seq[-max_len:]
    return [pad] * (max_len - len(seq)) + seq


def right_pad(seq: List[int], max_len: int, pad: int) -> List[int]:
    """Truncate to the most recent ``max_len`` items and right-pad with ``pad``.

    Right padding keeps every real token at a position that can attend to
    itself under a causal mask, avoiding fully-masked attention rows (which
    produce NaNs) for SASRec when ``src_key_padding_mask`` hides the pad slots.
    """
    seq = seq[-max_len:]
    return seq + [pad] * (max_len - len(seq))


def build_training_pairs(
    sequences: List[List[int]], max_len: int, pad: int
) -> Tuple[Tensor, Tensor]:
    """Next-item training tensors for autoregressive (SASRec-style) models.

    For a sequence ``[i0, i1, ..., in]`` the input is ``[i0..i_{n-1}]`` and the
    target is ``[i1..in]`` (shifted by one). Both are left-padded to ``max_len``;
    padded target positions are set to ``pad`` so the caller can mask the loss.
    """
    inputs, targets = [], []
    for seq in sequences:
        if len(seq) < 2:
            continue
        src = seq[:-1]
        tgt = seq[1:]
        # Right-pad so real tokens occupy leading positions (positions 0..L-1);
        # trailing pad targets are ignored in the loss via ignore_index=pad.
        inputs.append(right_pad(src, max_len, pad))
        targets.append(right_pad(tgt, max_len, pad))
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


@torch.no_grad()
def evaluate_next_item(
    data: AmazonSequenceData,
    score_fn,
    split: str,
    max_len: int,
    ks: List[int],
    device,
    eval_batch_size: int = 256,
) -> dict:
    """Evaluate a sequential model that maps padded histories to item scores.

    Histories are right-padded, so ``score_fn(input_ids, lengths)`` receives a
    ``[B, max_len]`` long tensor of item ids plus a ``[B]`` long tensor with the
    number of real tokens per row, and must return ``[B, num_items]`` next-item
    scores (taken at the last real position).
    """
    examples = data.eval_examples(split)
    metrics = RankingMetrics(ks)
    pad = pad_id(data.num_items)

    for start in range(0, len(examples), eval_batch_size):
        batch = examples[start : start + eval_batch_size]
        input_ids = torch.tensor(
            [right_pad(ex.history, max_len, pad) for ex in batch],
            dtype=torch.long,
            device=device,
        )
        # Number of real tokens (clamped to [1, max_len]); the prediction is
        # read from position lengths-1.
        lengths = torch.tensor(
            [min(max(len(ex.history), 1), max_len) for ex in batch],
            dtype=torch.long,
            device=device,
        )
        scores = score_fn(input_ids, lengths)
        targets = torch.tensor([ex.target for ex in batch], device=device)
        seen_mask = build_seen_mask([ex.seen for ex in batch], data.num_items, device)
        metrics.accumulate(scores, targets, seen_mask)
    return metrics.reduce()
