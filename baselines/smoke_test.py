"""Fast smoke tests for the retrieval baselines.

These run every model on a tiny synthetic dataset (a handful of users/items and
a couple of optimisation steps) purely to verify that shapes line up, training
executes, and the metric plumbing returns sensible numbers. They are NOT a
quality benchmark -- real evaluation happens on the Amazon data via each
module's ``run`` entry point.

    python -m baselines.smoke_test
"""

from typing import List

import torch

from baselines.data import EvalExample
from baselines.metrics import RankingMetrics, format_metrics

NUM_ITEMS = 12
SEED = 0


class SyntheticData:
    """Mimics the surface of :class:`baselines.data.AmazonSequenceData`."""

    def __init__(self) -> None:
        self.num_items = NUM_ITEMS
        # Six users with short, partly overlapping item sequences.
        self.sequences: List[List[int]] = [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            [5, 6, 7, 8, 9],
            [2, 4, 6, 8, 10],
            [0, 3, 6, 9, 11],
            [1, 4, 7, 10, 2],
        ]
        self.num_users = len(self.sequences)
        self.user_ids = list(range(self.num_users))

    def __len__(self) -> int:
        return len(self.sequences)

    def train_sequences(self) -> List[List[int]]:
        return [s[:-2] for s in self.sequences]

    def train_interactions(self) -> List[List[int]]:
        return [sorted(set(s[:-2])) for s in self.sequences]

    def eval_examples(self, split: str) -> List[EvalExample]:
        out = []
        for user, items in zip(self.user_ids, self.sequences):
            if split == "val":
                history, target = items[:-2], items[-2]
            else:
                history, target = items[:-1], items[-1]
            out.append(EvalExample(user, history, target, list(history)))
        return out

    def user_item_matrix(self) -> torch.Tensor:
        mat = torch.zeros((self.num_users, self.num_items))
        for row, items in enumerate(self.train_interactions()):
            mat[row, torch.tensor(items)] = 1.0
        return mat


def _check(name: str, metrics: dict) -> None:
    assert metrics, f"{name}: empty metrics"
    for key, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{name}: {key}={value} out of range"
    print(f"[OK] {name}: {format_metrics(metrics)}")


def test_metrics_ranking() -> None:
    """Hand-checked ranking: target at known rank yields known NDCG."""
    metrics = RankingMetrics([1, 2, 5])
    # Two items (0.9, 0.8) rank above the target (0.5) -> 0-based rank 2.
    scores = torch.tensor([[0.9, 0.8, 0.5, 0.3, 0.2]])
    targets = torch.tensor([2])
    metrics.accumulate(scores, targets)
    out = metrics.reduce()
    assert out["recall@1"] == 0.0 and out["recall@2"] == 0.0
    assert out["recall@5"] == 1.0
    expected_ndcg5 = 1.0 / torch.log2(torch.tensor(2.0 + 2.0)).item()
    assert abs(out["ndcg@5"] - expected_ndcg5) < 1e-6
    print(f"[OK] metrics ranking: {format_metrics(out)}")


def test_metrics_seen_mask() -> None:
    """Seen items must be excluded from the ranking but never the target."""
    metrics = RankingMetrics([1])
    scores = torch.tensor([[0.9, 0.5, 0.4]])  # item 0 scores highest
    targets = torch.tensor([1])
    seen = torch.tensor([[True, False, False]])  # mask out item 0
    metrics.accumulate(scores, targets, seen)
    out = metrics.reduce()
    assert out["recall@1"] == 1.0, out  # target now rank 0
    print(f"[OK] metrics seen-mask: {format_metrics(out)}")


def test_ease() -> None:
    from baselines.ease import train_ease, evaluate

    data = SyntheticData()
    X = data.user_item_matrix()
    B = train_ease(X, reg=1.0)
    assert B.shape == (NUM_ITEMS, NUM_ITEMS)
    assert torch.allclose(torch.diag(B), torch.zeros(NUM_ITEMS), atol=1e-6)
    _check("EASE", evaluate(data, X, B, "test", [1, 5, 10]))


def test_mf_bpr() -> None:
    from baselines.mf_bpr import train_mf_bpr, evaluate

    data = SyntheticData()
    model = train_mf_bpr(
        data, dim=8, epochs=3, batch_size=8, ks=[1, 5], seed=SEED, verbose=False
    )
    _check("MF-BPR", evaluate(data, model, "test", [1, 5, 10], "cpu"))


def test_sasrec() -> None:
    from baselines.sasrec import train_sasrec
    from baselines.sequential import evaluate_next_item

    data = SyntheticData()
    max_len = 8
    model = train_sasrec(
        data,
        dim=16,
        num_layers=1,
        num_heads=2,
        max_len=max_len,
        epochs=3,
        batch_size=4,
        ks=[1, 5],
        seed=SEED,
        verbose=False,
    )
    model.eval()
    metrics = evaluate_next_item(
        data, model.score_at_last, "test", max_len, [1, 5, 10], "cpu"
    )
    _check("SASRec", metrics)


def test_bert4rec() -> None:
    from baselines.bert4rec import train_bert4rec, evaluate

    data = SyntheticData()
    max_len = 8
    model = train_bert4rec(
        data,
        dim=16,
        num_layers=1,
        num_heads=2,
        max_len=max_len,
        epochs=3,
        batch_size=4,
        ks=[1, 5],
        seed=SEED,
        verbose=False,
    )
    _check("BERT4Rec", evaluate(data, model, "test", max_len, [1, 5, 10], "cpu"))


def test_tiger_ids() -> None:
    from baselines.tiger_ids import build_id_table, lsh_ids

    n_layers, codebook = 3, 16
    # Random ids: shape, range, dedup column makes every row unique.
    table = build_id_table("random", NUM_ITEMS, n_layers, codebook, seed=SEED)
    assert table.shape == (NUM_ITEMS, n_layers + 1)
    assert table[:, :n_layers].max() < codebook and table[:, :n_layers].min() >= 0
    rows = {tuple(r.tolist()) for r in table}
    assert len(rows) == NUM_ITEMS, "dedup column must make rows unique"

    # LSH: identical embeddings -> identical codes; codes stay in range.
    emb = torch.randn(NUM_ITEMS, 32)
    codes = lsh_ids(emb, n_layers, codebook, seed=SEED)
    assert codes.shape == (NUM_ITEMS, n_layers)
    assert codes.max() < codebook and codes.min() >= 0
    same = lsh_ids(emb.clone(), n_layers, codebook, seed=SEED)
    assert torch.equal(codes, same), "LSH must be deterministic for same input"
    print("[OK] tiger id generators (random + lsh)")


def _tiny_tiger(id_method: str, gen_mode: str, ks):
    """Train a tiny TIGER ablation and evaluate it with the given decoder."""
    from baselines.tiger import train_tiger, evaluate_tiger

    data = SyntheticData()
    max_len = 8
    embeddings = torch.randn(NUM_ITEMS, 32) if id_method == "lsh" else None
    model, tokenizer = train_tiger(
        data,
        id_method=id_method,
        embeddings=embeddings,
        n_layers=3,
        codebook_size=8,
        max_len=max_len,
        epochs=3,
        batch_size=4,
        warmup_steps=2,
        ks=ks,
        gen_mode=gen_mode,
        num_samples=64,
        t5_d_model=32,
        t5_num_heads=2,
        t5_d_ff=64,
        t5_num_layers=1,
        seed=SEED,
        verbose=False,
    )
    return evaluate_tiger(
        data, model, tokenizer, "test", max_len, ks, "cpu",
        gen_mode=gen_mode, num_samples=64,
    )


def test_tiger_random() -> None:
    # Sampling decoder must support large cutoffs (k = 50, 100).
    metrics = _tiny_tiger("random", gen_mode="sample", ks=[1, 10, 50, 100])
    assert "recall@100" in metrics and "ndcg@50" in metrics
    _check("TIGER-random (sample, k<=100)", metrics)


def test_tiger_lsh() -> None:
    # Exact beam decoder, small cutoffs (beam is intended for small k only).
    _check("TIGER-lsh (beam)", _tiny_tiger("lsh", gen_mode="beam", ks=[1, 5]))


def test_tiger_beam_rejects_large_k() -> None:
    """Beam decoding must refuse large cutoffs with a clear error."""
    from baselines.tiger import evaluate_tiger, BEAM_MAX_K

    try:
        _tiny_tiger("random", gen_mode="beam", ks=[BEAM_MAX_K + 50])
    except ValueError as e:
        assert "sample" in str(e)
        print("[OK] beam rejects large k with guidance to use sampling")
        return
    raise AssertionError("beam mode should reject k > BEAM_MAX_K")


def test_tiger_sample_ranking() -> None:
    """Unit-test the sample dedup + ranking helpers on hand-built inputs."""
    from baselines.tiger import _encode_code_id, _ranks_from_samples

    codebook = 8
    # Two users, four samples each (with a duplicate per user).
    codes = torch.tensor(
        [
            [[1, 2, 3], [4, 5, 6], [1, 2, 3], [0, 0, 0]],
            [[7, 7, 7], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
        ]
    )
    log_probs = torch.tensor(
        [
            [-0.5, -2.0, -0.9, -1.0],  # tuple (1,2,3) kept at best lp -0.5
            [-3.0, -0.1, -0.2, -0.3],
        ]
    )
    sample_ids = _encode_code_id(codes, codebook)
    # User 0 target = (4,5,6): unique tuples better than its lp(-2.0) are
    # (1,2,3)@-0.5 and (0,0,0)@-1.0 -> rank 2.
    # User 1 target = (3,3,3)@-0.3: better are (1,1,1)@-0.1, (2,2,2)@-0.2 -> rank 2.
    targets = _encode_code_id(torch.tensor([[4, 5, 6], [3, 3, 3]]), codebook)
    ranks = _ranks_from_samples(sample_ids, log_probs, targets)
    assert ranks.tolist() == [2, 2], ranks.tolist()

    # A target absent from the pool gets the NO_RANK sentinel.
    from baselines.tiger import NO_RANK

    missing = _encode_code_id(torch.tensor([[5, 5, 5], [6, 6, 6]]), codebook)
    miss_ranks = _ranks_from_samples(sample_ids, log_probs, missing)
    assert miss_ranks.tolist() == [NO_RANK, NO_RANK]
    print("[OK] tiger sample dedup + ranking helpers")


def main() -> None:
    torch.manual_seed(SEED)
    test_metrics_ranking()
    test_metrics_seen_mask()
    test_ease()
    test_mf_bpr()
    test_sasrec()
    test_bert4rec()
    test_tiger_ids()
    test_tiger_sample_ranking()
    test_tiger_random()
    test_tiger_lsh()
    test_tiger_beam_rejects_large_k()
    print("\nAll available smoke tests passed.")


if __name__ == "__main__":
    main()
