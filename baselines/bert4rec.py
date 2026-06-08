"""BERT4Rec -- Bidirectional Transformer recommender (Sun et al., 2019).

Trains a bidirectional Transformer with the Cloze objective: a random subset of
items in each sequence is replaced by a special ``[MASK]`` token and the model
predicts the originals from both left and right context. At inference time a
``[MASK]`` token is appended to the user's history and the catalogue is ranked
from that position's representation.

Special item ids extend the embedding table:
    pad  = num_items
    mask = num_items + 1

Run:
    python -m baselines.bert4rec --split beauty --epochs 100 --dim 64
"""

import argparse
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from baselines.data import AmazonSequenceData, build_seen_mask
from baselines.metrics import RankingMetrics, format_metrics
from baselines.sequential import left_pad, pad_id
from tqdm import trange

DEFAULT_DIM = 64
DEFAULT_LAYERS = 2
DEFAULT_HEADS = 2
DEFAULT_MAX_LEN = 50
DEFAULT_EPOCHS = 100
DEFAULT_LR = 0.001
DEFAULT_DROPOUT = 0.2
DEFAULT_MASK_PROB = 0.2
DEFAULT_BATCH_SIZE = 128
DEFAULT_KS = [5, 10, 50, 100]
EVAL_BATCH_SIZE = 256
EMBEDDING_INIT_STD = 0.02
IGNORE_INDEX = -100


def mask_id(num_items: int) -> int:
    """Dedicated ``[MASK]`` item id (two past the valid range)."""
    return num_items + 1


class BERT4Rec(nn.Module):
    """Bidirectional Transformer encoder with tied item scoring."""

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
        self.mask_idx = mask_id(num_items)

        # +2 special tokens: padding and mask.
        self.item_emb = nn.Embedding(num_items + 2, dim, padding_idx=self.pad_idx)
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
        """Bidirectional per-position hidden states ``[B, L, dim]``."""
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        hidden = self.item_emb(input_ids) + self.pos_emb(positions)
        hidden = self.input_norm(self.dropout(hidden))

        pad_mask = input_ids == self.pad_idx
        # No causal mask -> full bidirectional attention.
        return self.encoder(hidden, src_key_padding_mask=pad_mask)

    def _item_logits(self, hidden: Tensor) -> Tensor:
        return hidden @ self.item_emb.weight[: self.num_items].t()

    def forward(self, input_ids: Tensor) -> Tensor:
        """Logits over all items at every position: ``[B, L, num_items]``."""
        return self._item_logits(self.encode(input_ids))

    def score_last(self, input_ids: Tensor) -> Tensor:
        """Scores from the final position (assumed to be ``[MASK]``)."""
        return self._item_logits(self.encode(input_ids)[:, -1, :])


def _mask_sequence(
    seq: List[int],
    max_len: int,
    pad: int,
    mask: int,
    mask_prob: float,
    generator: torch.Generator,
) -> tuple:
    """Apply the Cloze mask to one sequence, returning (input_ids, labels).

    Masked positions hold ``mask`` in the input and the original item in the
    labels; all other label positions are ``IGNORE_INDEX``. At least one item is
    always masked so every training sequence contributes a learning signal.
    """
    seq = seq[-max_len:]
    probs = torch.rand(len(seq), generator=generator)
    masked = probs < mask_prob
    if not masked.any():
        masked[torch.randint(len(seq), (1,), generator=generator)] = True

    input_ids, labels = [], []
    for item, is_masked in zip(seq, masked.tolist()):
        if is_masked:
            input_ids.append(mask)
            labels.append(item)
        else:
            input_ids.append(item)
            labels.append(IGNORE_INDEX)

    pad_len = max_len - len(seq)
    input_ids = [pad] * pad_len + input_ids
    labels = [IGNORE_INDEX] * pad_len + labels
    return input_ids, labels


@torch.no_grad()
def evaluate(
    data: AmazonSequenceData,
    model: BERT4Rec,
    split: str,
    max_len: int,
    ks: List[int],
    device,
) -> dict:
    """Append ``[MASK]`` to each history and rank items from that position."""
    examples = data.eval_examples(split)
    metrics = RankingMetrics(ks, num_items=data.num_items)
    pad = pad_id(data.num_items)
    mask = mask_id(data.num_items)
    model.eval()

    for start in range(0, len(examples), EVAL_BATCH_SIZE):
        batch = examples[start : start + EVAL_BATCH_SIZE]
        input_ids = torch.tensor(
            [left_pad(ex.history + [mask], max_len, pad) for ex in batch],
            dtype=torch.long,
            device=device,
        )
        scores = model.score_last(input_ids)
        targets = torch.tensor([ex.target for ex in batch], device=device)
        seen_mask = build_seen_mask([ex.seen for ex in batch], data.num_items, device)
        metrics.accumulate(scores, targets, seen_mask)
    return metrics.reduce()


def train_bert4rec(
    data: AmazonSequenceData,
    dim: int = DEFAULT_DIM,
    num_layers: int = DEFAULT_LAYERS,
    num_heads: int = DEFAULT_HEADS,
    max_len: int = DEFAULT_MAX_LEN,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    dropout: float = DEFAULT_DROPOUT,
    mask_prob: float = DEFAULT_MASK_PROB,
    batch_size: int = DEFAULT_BATCH_SIZE,
    ks: List[int] = DEFAULT_KS,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
) -> BERT4Rec:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    pad = pad_id(data.num_items)
    mask = mask_id(data.num_items)

    model = BERT4Rec(data.num_items, dim, num_layers, num_heads, max_len, dropout)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    sequences = [s for s in data.train_sequences() if len(s) >= 1]
    num_samples = len(sequences)

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(num_samples, generator=generator).tolist()
        total_loss = 0.0
        for start in trange(0, num_samples, batch_size):
            chunk = order[start : start + batch_size]
            built = [
                _mask_sequence(sequences[i], max_len, pad, mask, mask_prob, generator)
                for i in chunk
            ]
            batch_in = torch.tensor([b[0] for b in built], device=device)
            batch_labels = torch.tensor([b[1] for b in built], device=device)

            optimizer.zero_grad()
            logits = model(batch_in)  # [B, L, num_items]
            loss = loss_fn(
                logits.reshape(-1, data.num_items), batch_labels.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(chunk)

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            val_metrics = evaluate(data, model, "val", max_len, ks, device)
            print(
                f"[BERT4Rec] epoch {epoch + 1:>3}/{epochs} "
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
    print(f"[BERT4Rec] split={split} users={len(data)} items={data.num_items} seed={seed}")
    model = train_bert4rec(data, dim=dim, epochs=epochs, max_len=max_len, ks=ks, device=device, seed=seed)
    val_metrics = evaluate(data, model, "val", max_len, ks, device)
    test_metrics = evaluate(data, model, "test", max_len, ks, device)
    print(f"[BERT4Rec] VAL : {format_metrics(val_metrics)}")
    print(f"[BERT4Rec] TEST: {format_metrics(test_metrics)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT4Rec baseline on Amazon Reviews.")
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
