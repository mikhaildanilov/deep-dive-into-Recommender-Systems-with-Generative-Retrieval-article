"""Ranking metrics (Recall@k, NDCG@k) for leave-one-out evaluation.

Every evaluation example has exactly one held-out target item, so for a cutoff
``k``:

* ``Recall@k`` reduces to Hit@k -- 1 if the target appears in the top-k, else 0,
* ``NDCG@k`` reduces to ``1 / log2(rank + 2)`` when the (0-based) ``rank`` of the
  target is ``< k`` and 0 otherwise (the ideal DCG is 1 for a single relevant
  item).

The accumulator consumes a dense score matrix ``[B, num_items]`` together with
the target ids and a mask of already-seen items, which keeps it reusable across
all baselines regardless of how the scores are produced.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch import Tensor

NEG_INF = float("-inf")


class RankingMetrics:
    """Accumulates Recall@k and NDCG@k over batches of scored items."""

    def __init__(self, ks: List[int] = [5, 10]) -> None:
        self.ks = sorted(ks)
        self.reset()

    def reset(self) -> None:
        self.total = 0
        self.recall: Dict[int, float] = defaultdict(float)
        self.ndcg: Dict[int, float] = defaultdict(float)

    @torch.no_grad()
    def accumulate(
        self,
        scores: Tensor,
        targets: Tensor,
        seen_mask: Optional[Tensor] = None,
    ) -> None:
        """Update metrics from a batch.

        Args:
            scores:    ``[B, num_items]`` float scores (higher = better).
            targets:   ``[B]`` long tensor of held-out target item ids.
            seen_mask: optional ``[B, num_items]`` bool mask of items to exclude
                       from ranking (e.g. items already in the user history).
        """
        scores = scores.clone().float()
        targets = targets.long()
        batch_size = scores.size(0)
        rows = torch.arange(batch_size, device=scores.device)

        target_scores = scores[rows, targets].clone()
        if seen_mask is not None:
            scores = scores.masked_fill(seen_mask, NEG_INF)
        # The held-out target must never be masked out of its own ranking.
        scores[rows, targets] = target_scores

        # 0-based rank = number of items strictly better than the target.
        ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1)

        for k in self.ks:
            hit = ranks < k
            self.recall[k] += hit.sum().item()
            if hit.any():
                gains = 1.0 / torch.log2(ranks[hit].float() + 2.0)
                self.ndcg[k] += gains.sum().item()
        self.total += batch_size

    def accumulate_from_ranks(self, ranks: Tensor) -> None:
        """Update metrics directly from 0-based target ranks ``[B]``.

        Useful for models that score only a candidate subset rather than the
        full item catalogue.
        """
        ranks = ranks.long()
        for k in self.ks:
            hit = ranks < k
            self.recall[k] += hit.sum().item()
            if hit.any():
                gains = 1.0 / torch.log2(ranks[hit].float() + 2.0)
                self.ndcg[k] += gains.sum().item()
        self.total += ranks.numel()

    def reduce(self) -> Dict[str, float]:
        if self.total == 0:
            return {}
        out: Dict[str, float] = {}
        for k in self.ks:
            out[f"recall@{k}"] = self.recall[k] / self.total
            out[f"ndcg@{k}"] = self.ndcg[k] / self.total
        return out


def format_metrics(metrics: Dict[str, float]) -> str:
    return ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
