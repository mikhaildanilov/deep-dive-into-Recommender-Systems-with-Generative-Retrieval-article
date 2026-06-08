
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
    metrics = RankingMetrics(ks)
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
