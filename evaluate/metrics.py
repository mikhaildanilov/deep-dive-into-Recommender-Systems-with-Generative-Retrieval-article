from collections import defaultdict
from einops import rearrange
from torch import Tensor

import torch


class TopKAccumulator:
    """Accumulates top-k retrieval metrics for leave-one-out evaluation.

    Each example has a single held-out target semantic ID, so for a cutoff ``k``:

    * ``h@k`` / ``recall@k`` -- 1 if the target tuple appears in the top-k
      candidates, else 0 (identical for a single target; both are reported for
      backward compatibility and clarity).
    * ``ndcg@k`` -- ``1 / log2(rank + 2)`` when the 0-based ``rank`` of the
      target is ``< k`` and 0 otherwise (ideal DCG is 1 for one relevant item).
    """

    def __init__(self, ks=[1, 5, 10]):
        self.ks = ks
        self.reset()

    def reset(self):
        self.total = 0
        self.metrics = defaultdict(float)

    def _update(self, matched_rank: Tensor) -> None:
        """Update all cutoffs from the 0-based ranks of the matched targets.

        ``matched_rank`` contains one entry per example whose target was found
        among the candidates (examples with no match contribute nothing).
        """
        for k in self.ks:
            in_top_k = matched_rank[matched_rank < k]
            hits = in_top_k.numel()
            ndcg = (
                (1.0 / torch.log2(in_top_k.float() + 2.0)).sum().item()
                if hits > 0
                else 0.0
            )
            # Always touch every key so reduce() reports all cutoffs, even when
            # a given k had no hits in this update.
            self.metrics[f"h@{k}"] += hits
            self.metrics[f"recall@{k}"] += hits
            self.metrics[f"ndcg@{k}"] += ndcg

    def accumulate(self, actual: Tensor, top_k: Tensor) -> None:
        """Update from generated candidates.

        Args:
            actual: ``[B, D]`` target semantic-ID tuples.
            top_k:  ``[B, K, D]`` candidate tuples ordered best-first (padding
                    slots may use ``-1`` and simply never match).
        """
        B, D = actual.shape
        pos_match = rearrange(actual, "b d -> b 1 d") == top_k
        match_found, rank = pos_match.all(axis=-1).max(axis=-1)
        matched_rank = rank[match_found]
        self._update(matched_rank)
        self.total += B

    def accumulate_from_ranks(self, ranks: Tensor) -> None:
        """Update directly from 0-based target ranks ``[B]``.

        Convenient for the sampling decoder, which computes the rank of each
        target within a deduplicated candidate pool. Use a large sentinel rank
        (e.g. ``>= max(ks)``) for targets absent from the pool.
        """
        ranks = ranks.long()
        self._update(ranks)
        self.total += ranks.numel()

    def reduce(self) -> dict:
        if self.total == 0:
            return {}
        return {k: v / self.total for k, v in self.metrics.items()}
