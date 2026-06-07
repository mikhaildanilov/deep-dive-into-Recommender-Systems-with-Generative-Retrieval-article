"""Fast tests for semantic-ID generation metrics and the sampling decoder.

Runs the real ``EncoderDecoderRetrievalModel`` on a tiny synthetic corpus to
verify, without any heavy data, that:

* :class:`evaluate.metrics.TopKAccumulator` reports recall@k / ndcg@k / h@k
  with the correct values, and
* sampling-based decoding (``gen_mode='sample'``) returns deduplicated
  candidate pools large enough for big cutoffs (k = 50, 100).

    python -m evaluate.test_semids_generation
"""

import torch

from data.schemas import TokenizedSeqBatch
from evaluate.metrics import TopKAccumulator
from modules.model import EncoderDecoderRetrievalModel

SEED = 0
N_LAYERS = 3
CODEBOOK_SIZE = 8
NUM_ITEMS = 12


def _toy_codebooks() -> torch.Tensor:
    """A small set of valid semantic-ID tuples (the item corpus)."""
    generator = torch.Generator().manual_seed(SEED)
    return torch.randint(
        CODEBOOK_SIZE, (NUM_ITEMS, N_LAYERS), generator=generator, dtype=torch.long
    )


def _toy_model(codebooks: torch.Tensor) -> EncoderDecoderRetrievalModel:
    return EncoderDecoderRetrievalModel(
        codebooks=codebooks,
        num_hierarchies=N_LAYERS,
        num_embeddings_per_hierarchy=CODEBOOK_SIZE,
        t5_d_model=32,
        t5_num_heads=2,
        t5_d_ff=64,
        t5_num_layers=1,
        top_k_for_generation=5,
        should_add_sep_token=True,
        num_user_bins=None,
    )


def _toy_batch(codebooks: torch.Tensor, batch_size: int = 4) -> TokenizedSeqBatch:
    """Build a length-2 history + target batch in the tokenizer's layout.

    Sequences carry the dedup column (sem_ids_dim = n_layers + 1) so the value
    matches what :class:`SemanticIdTokenizer` produces; the model strips it.
    """
    sem_ids_dim = N_LAYERS + 1
    seq_len = 2
    rows = []
    futs = []
    for b in range(batch_size):
        items = [(b % NUM_ITEMS), ((b + 1) % NUM_ITEMS)]
        row = []
        for it in items:
            row.extend(codebooks[it].tolist() + [0])  # +dedup col
        rows.append(row)
        futs.append(codebooks[(b + 2) % NUM_ITEMS].tolist() + [0])

    sem_ids = torch.tensor(rows, dtype=torch.long)
    sem_ids_fut = torch.tensor(futs, dtype=torch.long)
    seq_mask = torch.ones(batch_size, seq_len * sem_ids_dim, dtype=torch.bool)
    token_type_ids = torch.arange(sem_ids_dim).repeat(batch_size, seq_len)
    token_type_ids_fut = torch.arange(sem_ids_dim).repeat(batch_size, 1)
    return TokenizedSeqBatch(
        user_ids=-torch.ones(batch_size, dtype=torch.long),
        sem_ids=sem_ids,
        sem_ids_fut=sem_ids_fut,
        seq_mask=seq_mask,
        token_type_ids=token_type_ids,
        token_type_ids_fut=token_type_ids_fut,
    )


def test_metrics_values() -> None:
    """Hand-checked recall/ndcg/h on a known candidate ordering."""
    acc = TopKAccumulator(ks=[1, 2, 5])
    actual = torch.tensor([[1, 2, 3]])
    # Target sits at 0-based rank 2 in the candidate list.
    top_k = torch.tensor([[[9, 9, 9], [8, 8, 8], [1, 2, 3], [0, 0, 0], [7, 7, 7]]])
    acc.accumulate(actual=actual, top_k=top_k)
    out = acc.reduce()
    assert out["recall@1"] == 0.0 and out["recall@2"] == 0.0
    assert out["recall@5"] == 1.0 and out["h@5"] == 1.0
    expected_ndcg5 = 1.0 / torch.log2(torch.tensor(2.0 + 2.0)).item()
    assert abs(out["ndcg@5"] - expected_ndcg5) < 1e-6
    assert out["ndcg@2"] == 0.0
    print(f"[OK] metrics recall/ndcg/h: {out}")


def test_metrics_from_ranks() -> None:
    """rank-based path: large sentinel rank counts as a miss."""
    acc = TopKAccumulator(ks=[1, 10, 50, 100])
    ranks = torch.tensor([0, 9, 49, 10**9])  # last is absent from the pool
    acc.accumulate_from_ranks(ranks)
    out = acc.reduce()
    assert out["recall@1"] == 0.25  # only rank 0
    assert out["recall@10"] == 0.5  # ranks 0, 9
    assert out["recall@50"] == 0.75  # ranks 0, 9, 49
    assert out["recall@100"] == 0.75  # sentinel never counts
    assert abs(out["ndcg@1"] - 0.25) < 1e-6  # rank 0 -> gain 1, /4 examples
    print(f"[OK] metrics from ranks (k up to 100): {out}")


def test_sampling_decoder_large_k() -> None:
    """Sampling decoder yields a deduplicated pool covering k = 50, 100."""
    torch.manual_seed(SEED)
    codebooks = _toy_codebooks()
    model = _toy_model(codebooks)
    model.eval()
    batch = _toy_batch(codebooks)

    generated = model.generate_next_sem_id(
        batch,
        gen_mode="sample",
        num_samples=200,
        num_return=100,
        temperature=1.0,
        generator=torch.Generator().manual_seed(SEED),
    )
    assert generated.sem_ids.shape == (batch.sem_ids.size(0), 100, N_LAYERS)
    assert generated.log_probas.shape == (batch.sem_ids.size(0), 100)

    # Per row, non-padding tuples must be unique and sorted by log-prob desc.
    for b in range(generated.sem_ids.size(0)):
        valid = (generated.sem_ids[b] >= 0).all(dim=-1)
        tuples = [tuple(t.tolist()) for t in generated.sem_ids[b][valid]]
        assert len(tuples) == len(set(tuples)), "candidate pool has duplicates"
        lps = generated.log_probas[b][valid]
        assert torch.all(lps[:-1] >= lps[1:] - 1e-6), "candidates not sorted"

    acc = TopKAccumulator(ks=[1, 5, 10, 50, 100])
    acc.accumulate(actual=batch.sem_ids_fut[:, :N_LAYERS], top_k=generated.sem_ids)
    out = acc.reduce()
    assert {"recall@50", "ndcg@100"}.issubset(out.keys())
    print(f"[OK] sampling decoder (k up to 100): {out}")


def test_beam_decoder_still_works() -> None:
    """Default beam decoding path remains functional for small k."""
    torch.manual_seed(SEED)
    codebooks = _toy_codebooks()
    model = _toy_model(codebooks)
    model.eval()
    batch = _toy_batch(codebooks)

    generated = model.generate_next_sem_id(batch, gen_mode="beam")
    assert generated.sem_ids.shape[-1] == N_LAYERS
    acc = TopKAccumulator(ks=[1, 5])
    acc.accumulate(actual=batch.sem_ids_fut[:, :N_LAYERS], top_k=generated.sem_ids)
    print(f"[OK] beam decoder: {acc.reduce()}")


def main() -> None:
    test_metrics_values()
    test_metrics_from_ranks()
    test_sampling_decoder_large_k()
    test_beam_decoder_still_works()
    print("\nAll semantic-ID generation tests passed.")


if __name__ == "__main__":
    main()
