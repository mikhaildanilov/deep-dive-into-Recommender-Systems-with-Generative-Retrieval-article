"""SASRec -- Self-Attentive Sequential Recommendation (Kang & McAuley, 2018).

A unidirectional (causal) Transformer over the user's item history. At every
position it predicts the next item; the representation of the last position is
used to rank the catalogue at inference time. Trained with full-softmax
cross-entropy over all items (padded positions are masked out of the loss).

Run:
    python -m baselines.sasrec --split beauty --epochs 100 --dim 64
"""

import argparse
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from baselines.data import AmazonSequenceData
from baselines.metrics import format_metrics
from baselines.sequential import (
    build_training_pairs,
    evaluate_next_item,
    pad_id,
)
from tqdm import trange

DEFAULT_DIM = 64
DEFAULT_LAYERS = 2
DEFAULT_HEADS = 2
DEFAULT_MAX_LEN = 50
DEFAULT_EPOCHS = 100
DEFAULT_LR = 0.001
DEFAULT_DROPOUT = 0.2
DEFAULT_BATCH_SIZE = 128
DEFAULT_KS = [5, 10, 50, 100]
EMBEDDING_INIT_STD = 0.02


class SASRec(nn.Module):
    """Causal self-attention sequence encoder with tied item scoring."""

    def __init__(
        self,
        num_items: int,
        dim: int = DEFAULT_DIM,
        num_layers: int = DEFAULT_LAYERS,
        num_heads: int = DEFAULT_HEADS,
        max_len: int = DEFAULT_MAX_LEN,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.pad_idx = pad_id(num_items)

        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=self.pad_idx)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(dim)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )

        nn.init.normal_(self.item_emb.weight, std=EMBEDDING_INIT_STD)
        nn.init.constant_(self.item_emb.weight[self.pad_idx], 0.0)
        nn.init.normal_(self.pos_emb.weight, std=EMBEDDING_INIT_STD)

    def encode(self, input_ids: Tensor) -> Tensor:
        """Return per-position hidden states ``[B, L, dim]``."""
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        hidden = self.item_emb(input_ids) + self.pos_emb(positions)
        hidden = self.input_norm(self.dropout(hidden))

        causal_mask = torch.triu(
            torch.ones(length, length, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        pad_mask = input_ids == self.pad_idx
        return self.encoder(
            hidden, mask=causal_mask, src_key_padding_mask=pad_mask
        )

    def _item_logits(self, hidden: Tensor) -> Tensor:
        """Score hidden states against the real-item embedding table (no pad)."""
        return hidden @ self.item_emb.weight[: self.num_items].t()

    def forward(self, input_ids: Tensor) -> Tensor:
        """Logits over all items at every position: ``[B, L, num_items]``."""
        return self._item_logits(self.encode(input_ids))

    def score_at_last(self, input_ids: Tensor, lengths: Tensor) -> Tensor:
        """Next-item scores from the last real position: ``[B, num_items]``.

        With right padding the prediction lives at index ``lengths - 1`` (the
        last non-pad token), not at the fixed final column.
        """
        hidden = self.encode(input_ids)  # [B, L, dim]
        batch = input_ids.size(0)
        last_idx = (lengths - 1).clamp(min=0)
        last_hidden = hidden[torch.arange(batch, device=hidden.device), last_idx]
        return self._item_logits(last_hidden)


def train_sasrec(
    data: AmazonSequenceData,
    dim: int = DEFAULT_DIM,
    num_layers: int = DEFAULT_LAYERS,
    num_heads: int = DEFAULT_HEADS,
    max_len: int = DEFAULT_MAX_LEN,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    dropout: float = DEFAULT_DROPOUT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    ks: List[int] = DEFAULT_KS,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
) -> SASRec:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    pad = pad_id(data.num_items)

    model = SASRec(data.num_items, dim, num_layers, num_heads, max_len, dropout)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad)

    inputs, targets = build_training_pairs(data.train_sequences(), max_len, pad)
    num_samples = inputs.size(0)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(num_samples, generator=generator)
        total_loss = 0.0
        for start in trange(0, num_samples, batch_size):
            idx = perm[start : start + batch_size]
            batch_in = inputs[idx].to(device)
            batch_tgt = targets[idx].to(device)

            optimizer.zero_grad()
            logits = model(batch_in)  # [B, L, num_items]
            loss = loss_fn(
                logits.reshape(-1, data.num_items), batch_tgt.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * idx.size(0)

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            model.eval()
            val_metrics = evaluate_next_item(
                data, model.score_at_last, "val", max_len, ks, device
            )
            print(
                f"[SASRec] epoch {epoch + 1:>3}/{epochs} "
                f"loss={total_loss / num_samples:.4f} val: {format_metrics(val_metrics)}"
            )
    return model


def run(
    split: str,
    dim: int,
    epochs: int,
    max_len: int,
    ks: List[int],
    device: str = "cpu",
    seed: int = 42,
) -> None:
    data = AmazonSequenceData(split=split)
    print(f"[SASRec] split={split} users={len(data)} items={data.num_items} seed={seed}")
    model = train_sasrec(data, dim=dim, epochs=epochs, max_len=max_len,
                         ks=ks, device=device, seed=seed)
    model.eval()
    val_metrics = evaluate_next_item(data, model.score_at_last, "val", max_len, ks, device)
    test_metrics = evaluate_next_item(data, model.score_at_last, "test", max_len, ks, device)
    print(f"[SASRec] VAL : {format_metrics(val_metrics)}")
    print(f"[SASRec] TEST: {format_metrics(test_metrics)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SASRec baseline on Amazon Reviews.")
    parser.add_argument("--split", type=str, default="beauty")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(
        split=args.split,
        dim=args.dim,
        epochs=args.epochs,
        max_len=args.max_len,
        ks=args.ks,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
