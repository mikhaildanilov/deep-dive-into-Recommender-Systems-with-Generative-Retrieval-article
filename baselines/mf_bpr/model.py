"""MF-BPR -- Matrix Factorization with Bayesian Personalized Ranking.

Classic collaborative filtering baseline (Rendle et al., 2009; arXiv:1205.2618).
Each user and item gets a latent embedding; the score of an (user, item) pair is
their dot product plus a per-item bias. The model is trained on (user, positive,
negative) triples with the pairwise BPR loss

    L = -log sigmoid(score_pos - score_neg)

where the positive item is sampled from the user's training interactions and the
negative item is sampled uniformly from items the user has not interacted with.

Users are indexed by their *position* in :class:`AmazonSequenceData` (row order
of ``sequences``), which is exactly the order produced by ``eval_examples``.

Run:
    python -m baselines.mf_bpr --split beauty --epochs 50 --dim 64
"""

import argparse
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from baselines.data import AmazonSequenceData, build_seen_mask
from baselines.metrics import RankingMetrics, format_metrics

DEFAULT_DIM = 64
DEFAULT_EPOCHS = 50
DEFAULT_LR = 0.01
DEFAULT_REG = 1e-5
DEFAULT_BATCH_SIZE = 2048
DEFAULT_KS = [5, 10, 50, 100]
EVAL_BATCH_SIZE = 512
EMBEDDING_INIT_STD = 0.01


class MFBPR(nn.Module):
    """Biased matrix factorization scored with dot products."""

    def __init__(self, num_users: int, num_items: int, dim: int = DEFAULT_DIM) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        self.item_bias = nn.Embedding(num_items, 1)

        nn.init.normal_(self.user_emb.weight, std=EMBEDDING_INIT_STD)
        nn.init.normal_(self.item_emb.weight, std=EMBEDDING_INIT_STD)
        nn.init.zeros_(self.item_bias.weight)

    def score_users(self, user_idx: Tensor) -> Tensor:
        """Full-catalogue scores ``[B, num_items]`` for the given users."""
        u = self.user_emb(user_idx)
        return u @ self.item_emb.weight.t() + self.item_bias.weight.squeeze(1)

    def bpr_loss(self, users: Tensor, pos: Tensor, neg: Tensor, reg: float) -> Tensor:
        u = self.user_emb(users)
        i = self.item_emb(pos)
        j = self.item_emb(neg)
        pos_score = (u * i).sum(-1) + self.item_bias(pos).squeeze(-1)
        neg_score = (u * j).sum(-1) + self.item_bias(neg).squeeze(-1)
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()
        l2 = reg * (u.pow(2).sum() + i.pow(2).sum() + j.pow(2).sum()) / users.size(0)
        return loss + l2


def _build_triples(train_interactions: List[List[int]]) -> tuple:
    """Flatten per-user positives into parallel (user_row, pos_item) tensors."""
    users, positives = [], []
    for row, items in enumerate(train_interactions):
        for item in items:
            users.append(row)
            positives.append(item)
    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(positives, dtype=torch.long),
    )


def _sample_negatives(
    user_rows: Tensor,
    positive_sets: List[set],
    num_items: int,
    generator: torch.Generator,
) -> Tensor:
    """Uniformly sample one negative item per row, rejecting known positives."""
    neg = torch.randint(num_items, (user_rows.size(0),), generator=generator)
    for idx in range(user_rows.size(0)):
        row = int(user_rows[idx])
        while int(neg[idx]) in positive_sets[row]:
            neg[idx] = torch.randint(num_items, (1,), generator=generator)
    return neg

