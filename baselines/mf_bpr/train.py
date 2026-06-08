
@torch.no_grad()
def evaluate(
    data: AmazonSequenceData,
    model: MFBPR,
    split: str,
    ks: List[int],
    device,
) -> dict:
    examples = data.eval_examples(split)
    metrics = RankingMetrics(ks)
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
