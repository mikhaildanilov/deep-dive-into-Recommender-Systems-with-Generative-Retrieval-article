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


@torch.no_grad()
def evaluate(
    data: AmazonSequenceData,
    model: MFBPR,
    split: str,
    ks: List[int],
    device,
) -> dict:
    examples = data.eval_examples(split)
    metrics = RankingMetrics(ks, num_items=data.num_items)
    for start in range(0, len(examples), EVAL_BATCH_SIZE):
        batch = examples[start : start + EVAL_BATCH_SIZE]
        # Row position equals user-embedding index (see module docstring).
        user_idx = torch.arange(start, start + len(batch), device=device)
        scores = model.score_users(user_idx)
        targets = torch.tensor([ex.target for ex in batch], device=device)
        seen_mask = build_seen_mask([ex.seen for ex in batch], data.num_items, device)
        metrics.accumulate(scores, targets, seen_mask)
    return metrics.reduce()


def train_mf_bpr(
    data: AmazonSequenceData,
    dim: int = DEFAULT_DIM,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    reg: float = DEFAULT_REG,
    batch_size: int = DEFAULT_BATCH_SIZE,
    ks: List[int] = DEFAULT_KS,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
) -> MFBPR:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    num_users = len(data)
    interactions = data.train_interactions()
    positive_sets = [set(items) for items in interactions]

    model = MFBPR(num_users, data.num_items, dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    users_all, pos_all = _build_triples(interactions)
    num_triples = users_all.size(0)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(num_triples, generator=generator)
        total_loss = 0.0
        for start in range(0, num_triples, batch_size):
            idx = perm[start : start + batch_size]
            batch_users = users_all[idx]
            batch_pos = pos_all[idx]
            batch_neg = _sample_negatives(
                batch_users, positive_sets, data.num_items, generator
            )

            optimizer.zero_grad()
            loss = model.bpr_loss(
                batch_users.to(device),
                batch_pos.to(device),
                batch_neg.to(device),
                reg,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * idx.size(0)

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            val_metrics = evaluate(data, model, "val", ks, device)
            print(
                f"[MF-BPR] epoch {epoch + 1:>3}/{epochs} "
                f"loss={total_loss / num_triples:.4f} val: {format_metrics(val_metrics)}"
            )
    return model


def run(split: str, dim: int, epochs: int, lr: float, ks: List[int], seed: int) -> None:
    data = AmazonSequenceData(split=split)
    print(f"[MF-BPR] split={split} users={len(data)} items={data.num_items} seed={seed}")
    model = train_mf_bpr(data, dim=dim, epochs=epochs, lr=lr, ks=ks, seed=seed)
    val_metrics = evaluate(data, model, "val", ks, "cpu")
    test_metrics = evaluate(data, model, "test", ks, "cpu")
    print(f"[MF-BPR] VAL : {format_metrics(val_metrics)}")
    print(f"[MF-BPR] TEST: {format_metrics(test_metrics)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MF-BPR baseline on Amazon Reviews.")
    parser.add_argument("--split", type=str, default="beauty")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(split=args.split, dim=args.dim, epochs=args.epochs, lr=args.lr, ks=args.ks, seed=args.seed)


if __name__ == "__main__":
    main()
