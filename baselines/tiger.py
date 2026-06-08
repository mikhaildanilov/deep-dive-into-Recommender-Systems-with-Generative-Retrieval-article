"""TIGER ablations: Random-ID and LSH-ID semantic identifiers.

These ablations keep TIGER's full encoder-decoder generative-retrieval model
(:class:`modules.model.EncoderDecoderRetrievalModel`) untouched and only swap
out how each item's discrete code tuple is produced -- random codes or LSH
codes instead of RQ-VAE Semantic IDs. This isolates the contribution of
content-based quantization, exactly as in the original paper.

The model autoregressively generates the ``n_layers`` codes of the next item.
Two decoding strategies are supported at evaluation time:

* ``"beam"`` -- the model's exact sampling-based beam search. Precise, but its
  cost grows with the beam width, so it is only practical for small cutoffs
  (k up to ~20).
* ``"sample"`` -- stochastic autoregressive sampling. We draw ``num_samples``
  full code tuples per user (sampling one token per hierarchy from the softmax,
  optionally temperature-scaled), then deduplicate and rank the candidates by
  their cumulative log-probability. This yields a large but *imprecise*
  candidate pool and is what makes long recommendation lists (k = 50, 100, ...)
  feasible for the generative model.

The generated code tuples are matched against the held-out target's codes to
obtain its rank and hence Recall@k / NDCG@k.

Run:
    python -m baselines.tiger --split beauty --id-method random --epochs 5000
    python -m baselines.tiger --split beauty --id-method lsh --gen-mode sample --ks 10 50 100
"""

import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from baselines.data import AmazonSequenceData
from baselines.metrics import RankingMetrics, format_metrics
from baselines.tiger_ids import build_id_table
from modules.model import EncoderDecoderRetrievalModel, _strip_dedup_col
from tqdm import trange

DEFAULT_N_LAYERS = 3
DEFAULT_CODEBOOK_SIZE = 256
DEFAULT_MAX_LEN = 20
DEFAULT_EPOCHS = 20000
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_BATCH_SIZE = 256
DEFAULT_KS = [5, 10, 50, 100]
DEFAULT_T5_D_MODEL = 128
DEFAULT_T5_HEADS = 6
DEFAULT_T5_D_FF = 1024
DEFAULT_T5_LAYERS = 4
DEFAULT_WARMUP = 10000
EVAL_BATCH_SIZE = 128
NO_RANK = 10**9  # sentinel rank when the target is absent from the candidate pool

# Decoding strategy for evaluation.
GEN_MODE_BEAM = "beam"
GEN_MODE_SAMPLE = "sample"
DEFAULT_GEN_MODE = GEN_MODE_SAMPLE
DEFAULT_TEMPERATURE = 1.0
# The exact beam search (modules.model.generate) is only intended for short
# recommendation lists; its candidate width must stay small. For larger cutoffs
# use the sampling decoder, which is what makes k = 50, 100, ... tractable.
BEAM_MAX_K = 20
# Oversampling factor: draw this many samples per requested candidate so that,
# after deduplication, the pool comfortably covers the largest k.
SAMPLE_OVERSAMPLE = 10
MIN_SAMPLES = 256
# Cap on (users x samples) rows pushed through the decoder at once, to bound
# peak memory when sampling many candidates.
SAMPLE_MAX_ROWS = 4096
LOG_EPS = 1e-12


def default_num_samples(ks: List[int]) -> int:
    """Sample budget that, after dedup, reliably covers the largest cutoff."""
    return max(SAMPLE_OVERSAMPLE * max(ks), MIN_SAMPLES)


class PrecomputedIdTokenizer:
    """Maps item-id sequences to TIGER ``TokenizedSeqBatch`` code sequences.

    Wraps a precomputed ``[num_items, n_layers + 1]`` code table (the extra
    column is TIGER's deduplication index) and reproduces the exact tensor
    layout expected by :class:`EncoderDecoderRetrievalModel`.
    """

    def __init__(self, id_table: Tensor, n_layers: int) -> None:
        self.cached_ids = id_table
        self.n_layers = n_layers
        self.sem_ids_dim = id_table.size(1)  # n_layers + 1

    def to(self, device) -> "PrecomputedIdTokenizer":
        self.cached_ids = self.cached_ids.to(device)
        return self

    @property
    def codebooks(self) -> Tensor:
        """Code table without the dedup column, used to validate beam prefixes."""
        return self.cached_ids[:, : self.n_layers]

    def _lookup(self, ids: Tensor) -> Tensor:
        """``[B, N] -> [B, N * sem_ids_dim]`` code lookup (padding handled by caller)."""
        return rearrange(
            self.cached_ids[ids.flatten(), :], "(b n) d -> b (n d)", n=ids.shape[1]
        )

    def tokenize(self, item_ids: Tensor, seq_mask: Tensor, target_ids: Tensor, user_ids: Tensor):
        """Build a ``TokenizedSeqBatch`` from padded id sequences.

        Args:
            item_ids:   ``[B, N]`` long ids; padding positions hold ``-1``.
            seq_mask:   ``[B, N]`` bool, ``True`` for real items.
            target_ids: ``[B]`` long ids of the next item to predict.
            user_ids:   ``[B]`` long user ids (unused unless user bins enabled).
        """
        from data.schemas import TokenizedSeqBatch

        device = self.cached_ids.device
        B, N = item_ids.shape
        D = self.sem_ids_dim

        sem_ids = self._lookup(item_ids)
        seq_mask_expanded = seq_mask.repeat_interleave(D, dim=1)
        sem_ids[~seq_mask_expanded] = -1

        sem_ids_fut = self.cached_ids[target_ids]  # [B, D]

        token_type_ids = torch.arange(D, device=device).repeat(B, N)
        token_type_ids_fut = torch.arange(D, device=device).repeat(B, 1)

        return TokenizedSeqBatch(
            user_ids=user_ids,
            sem_ids=sem_ids,
            sem_ids_fut=sem_ids_fut,
            seq_mask=seq_mask_expanded,
            token_type_ids=token_type_ids,
            token_type_ids_fut=token_type_ids_fut,
        )


def _pad_batch(histories: List[List[int]], max_len: int, num_items: int, device):
    """Right-pad histories with ``-1`` and build the matching boolean mask.

    ``-1`` keeps the TIGER tokenizer convention (it indexes ``cached_ids`` and is
    immediately overwritten via the mask), distinct from the sequential models'
    ``pad_id``.
    """
    padded, masks = [], []
    for hist in histories:
        hist = hist[-max_len:]
        pad_len = max_len - len(hist)
        padded.append(hist + [-1] * pad_len)
        masks.append([True] * len(hist) + [False] * pad_len)
    item_ids = torch.tensor(padded, dtype=torch.long, device=device)
    seq_mask = torch.tensor(masks, dtype=torch.bool, device=device)
    return item_ids, seq_mask


def build_training_examples(sequences: List[List[int]]) -> List[tuple]:
    """Expand each training sequence into next-item ``(history, target)`` pairs.

    For ``[i0, i1, ..., in]`` yields ``([i0..i_{k-1}], i_k)`` for every
    ``k >= 1``, giving the autoregressive model dense supervision while staying
    strictly within the training prefix (no val/test leakage).
    """
    examples = []
    for seq in sequences:
        for k in range(1, len(seq)):
            examples.append((seq[:k], seq[k]))
    return examples


def _ranks_from_beam(generated: Tensor, target_codes: Tensor) -> Tensor:
    """Rank of each target's code tuple within the beam output.

    Args:
        generated:    ``[B, top_k, n_layers]`` beam tuples, best-first.
        target_codes: ``[B, n_layers]`` target code tuples.

    Returns:
        ``[B]`` 0-based ranks, or ``NO_RANK`` where the target is absent.
    """
    matches = (generated == target_codes.unsqueeze(1)).all(dim=-1)  # [B, top_k]
    top_k = generated.size(1)
    positions = torch.arange(top_k, device=generated.device).unsqueeze(0)
    ranked = torch.where(matches, positions, torch.full_like(positions, NO_RANK))
    return ranked.min(dim=1).values


def _encode_code_id(codes: Tensor, codebook_size: int) -> Tensor:
    """Pack a ``[..., n_layers]`` code tuple into a single integer id ``[...]``.

    Lets us deduplicate and compare whole code tuples with cheap integer ops.
    ``codebook_size ** n_layers`` stays well within int64 for the configured
    sizes (e.g. 256**4 < 2**63).
    """
    n = codes.shape[-1]
    powers = (codebook_size ** torch.arange(n, device=codes.device)).long()
    return (codes.long() * powers).sum(dim=-1)


@torch.no_grad()
def sample_candidates(
    model: EncoderDecoderRetrievalModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    user_ids: Tensor,
    num_samples: int,
    n_layers: int,
    temperature: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    """Stochastically sample whole code tuples for a batch of users.

    Encodes the histories once, replicates the encoder memory ``num_samples``
    times per user, and samples one token per hierarchy from the (temperature-
    scaled) softmax. Unlike the model's beam search this performs no validity
    masking or pruning -- it is a fast, imprecise generator of a large candidate
    pool, which is exactly what large cutoffs (k = 50, 100, ...) require.

    Args:
        input_ids:      ``[U, L]`` history codes (dedup column already stripped).
        attention_mask: ``[U, L]`` long mask matching ``input_ids``.
        user_ids:       ``[U]`` user ids (used only if the model has user bins).
        num_samples:    samples drawn per user (before deduplication).
        temperature:    softmax temperature; ``< 1`` sharpens, ``> 1`` flattens.

    Returns:
        codes:     ``[U, num_samples, n_layers]`` sampled code tuples.
        log_probs: ``[U, num_samples]`` cumulative log-probability of each tuple.
    """
    enc_out, enc_mask = model.encoder_forward_pass(
        attention_mask=attention_mask, input_ids=input_ids, user_id=user_ids
    )
    num_users = enc_out.size(0)
    # Bound peak memory: process at most SAMPLE_MAX_ROWS (user x sample) rows.
    users_per_chunk = max(1, SAMPLE_MAX_ROWS // num_samples)

    code_chunks, logp_chunks = [], []
    for chunk_start in range(0, num_users, users_per_chunk):
        chunk_end = min(num_users, chunk_start + users_per_chunk)
        rep_enc = enc_out[chunk_start:chunk_end].repeat_interleave(num_samples, dim=0)
        rep_mask = enc_mask[chunk_start:chunk_end].repeat_interleave(num_samples, dim=0)

        generated = None  # [rows, h], grows by one hierarchy each step
        log_probs = torch.zeros(rep_enc.size(0), device=rep_enc.device)
        for h in range(n_layers):
            # No KV cache: re-run the (short) decoder prefix each hierarchy.
            dec_out = model.decoder_forward_pass(
                future_ids=generated,
                encoder_output=rep_enc,
                attention_mask_for_encoder=rep_mask,
                use_cache=False,
            )
            logits = model.decoder_mlp[h](dec_out[:, -1, :]) / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
            token_logp = torch.log(torch.gather(probs, 1, nxt).squeeze(1) + LOG_EPS)
            log_probs = log_probs + token_logp
            generated = nxt if generated is None else torch.cat([generated, nxt], dim=1)

        rows = chunk_end - chunk_start
        code_chunks.append(generated.view(rows, num_samples, n_layers))
        logp_chunks.append(log_probs.view(rows, num_samples))

    return torch.cat(code_chunks, dim=0), torch.cat(logp_chunks, dim=0)


def _ranks_from_samples(
    sample_code_ids: Tensor, log_probs: Tensor, target_ids: Tensor
) -> Tensor:
    """Rank of each target among the deduplicated sampled candidate pool.

    Duplicate samples of the same tuple are collapsed, keeping their best
    (highest) log-probability. The target's 0-based rank is the number of unique
    tuples scoring strictly higher (consistent with :class:`RankingMetrics`).
    Targets absent from the pool get :data:`NO_RANK`.

    Args:
        sample_code_ids: ``[U, S]`` integer-encoded sampled tuples.
        log_probs:       ``[U, S]`` cumulative log-probabilities.
        target_ids:      ``[U]`` integer-encoded target tuples.
    """
    codes = sample_code_ids.tolist()
    lps = log_probs.tolist()
    targets = target_ids.tolist()

    ranks: List[int] = []
    for code_row, lp_row, target in zip(codes, lps, targets):
        best: dict = {}
        for code, lp in zip(code_row, lp_row):
            current = best.get(code)
            if current is None or lp > current:
                best[code] = lp
        target_lp = best.get(target)
        if target_lp is None:
            ranks.append(NO_RANK)
        else:
            ranks.append(sum(1 for lp in best.values() if lp > target_lp))
    return torch.tensor(ranks, dtype=torch.long)


@torch.no_grad()
def evaluate_tiger(
    data: AmazonSequenceData,
    model: EncoderDecoderRetrievalModel,
    tokenizer: PrecomputedIdTokenizer,
    split: str,
    max_len: int,
    ks: List[int],
    device,
    gen_mode: str = DEFAULT_GEN_MODE,
    num_samples: Optional[int] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = 42,
) -> dict:
    """Evaluate a TIGER ablation.

    ``gen_mode='beam'`` uses the model's exact beam search (only sensible for
    small ``ks``); ``gen_mode='sample'`` uses stochastic sampling and supports
    arbitrarily large cutoffs. ``num_samples`` defaults to a budget that, after
    deduplication, comfortably covers ``max(ks)``.
    """
    examples = data.eval_examples(split)
    metrics = RankingMetrics(ks)

    model.eval()
    n_layers = tokenizer.n_layers
    codebook_size = model.num_embeddings_per_hierarchy

    # Coverage: build reverse lookup encoded_code_id -> item_id
    covered: Dict[int, set] = {k: set() for k in ks}
    all_codes = tokenizer.cached_ids[:, :n_layers].to(device)
    all_code_ids = _encode_code_id(all_codes, codebook_size)  # [num_items]
    code_to_item = {int(cid): iid for iid, cid in enumerate(all_code_ids.cpu().tolist())}
    num_items = len(code_to_item)

    sem_ids_dim = n_layers + 1

    if gen_mode == GEN_MODE_SAMPLE and num_samples is None:
        num_samples = default_num_samples(ks)
    generator = (
        torch.Generator(device=device).manual_seed(seed)
        if gen_mode == GEN_MODE_SAMPLE
        else None
    )

    for start in trange(0, len(examples), EVAL_BATCH_SIZE, desc='Evaluate'):
        batch = examples[start : start + EVAL_BATCH_SIZE]
        item_ids, seq_mask = _pad_batch(
            [ex.history for ex in batch], max_len, data.num_items, device
        )
        target_ids = torch.tensor([ex.target for ex in batch], device=device)
        user_ids = torch.tensor([ex.user for ex in batch], device=device)
        tokenized = tokenizer.tokenize(item_ids, seq_mask, target_ids, user_ids)
        target_codes = tokenizer.cached_ids[target_ids][:, :n_layers]

        if gen_mode == GEN_MODE_BEAM:
            if max(ks) > BEAM_MAX_K:
                raise ValueError(
                    f"gen_mode='beam' supports only small cutoffs (max k <= "
                    f"{BEAM_MAX_K}); got max(ks)={max(ks)}. Use gen_mode='sample' "
                    f"for large recommendation lists."
                )
            generated = model.generate_next_sem_id(
                tokenized, top_k=True, temperature=1
            )
            ranks = _ranks_from_beam(generated.sem_ids, target_codes)
        elif gen_mode == GEN_MODE_SAMPLE:
            input_ids = _strip_dedup_col(tokenized.sem_ids, sem_ids_dim, n_layers)
            attention_mask = _strip_dedup_col(
                tokenized.seq_mask.long(), sem_ids_dim, n_layers
            )
            codes, log_probs = sample_candidates(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                user_ids=tokenized.user_ids,
                num_samples=num_samples,
                n_layers=n_layers,
                temperature=temperature,
                generator=generator,
            )
            sample_code_ids = _encode_code_id(codes, codebook_size)
            target_code_ids = _encode_code_id(target_codes, codebook_size)
            ranks = _ranks_from_samples(sample_code_ids, log_probs, target_code_ids)

            # Coverage tracking: top-K unique codes per user by log_prob
            for user_codes, user_lps in zip(sample_code_ids.tolist(), log_probs.tolist()):
                best: dict = {}
                for code, lp in zip(user_codes, user_lps):
                    if code not in best or lp > best[code]:
                        best[code] = lp
                sorted_codes = sorted(best, key=best.__getitem__, reverse=True)
                for k in ks:
                    for code in sorted_codes[:k]:
                        if code in code_to_item:
                            covered[k].add(code_to_item[code])
        else:
            raise ValueError(
                f"Unknown gen_mode {gen_mode!r}; use "
                f"{GEN_MODE_BEAM!r} or {GEN_MODE_SAMPLE!r}."
            )

        metrics.accumulate_from_ranks(ranks)

    result = metrics.reduce()
    for k in ks:
        result[f"coverage@{k}"] = len(covered[k]) / num_items

    return result


def build_model(
    tokenizer: PrecomputedIdTokenizer,
    n_layers: int,
    codebook_size: int,
    top_k_for_generation: int,
    t5_d_model: int = DEFAULT_T5_D_MODEL,
    t5_num_heads: int = DEFAULT_T5_HEADS,
    t5_d_ff: int = DEFAULT_T5_D_FF,
    t5_num_layers: int = DEFAULT_T5_LAYERS,
) -> EncoderDecoderRetrievalModel:
    return EncoderDecoderRetrievalModel(
        codebooks=tokenizer.codebooks.cpu(),
        num_hierarchies=n_layers,
        num_embeddings_per_hierarchy=codebook_size,
        t5_d_model=t5_d_model,
        t5_num_heads=t5_num_heads,
        t5_d_ff=t5_d_ff,
        t5_num_layers=t5_num_layers,
        top_k_for_generation=top_k_for_generation,
        should_add_sep_token=True,
        num_user_bins=None,
    )


def train_tiger(
    data: AmazonSequenceData,
    id_method: str,
    embeddings: Optional[Tensor] = None,
    n_layers: int = DEFAULT_N_LAYERS,
    codebook_size: int = DEFAULT_CODEBOOK_SIZE,
    max_len: int = DEFAULT_MAX_LEN,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    warmup_steps: int = DEFAULT_WARMUP,
    ks: List[int] = DEFAULT_KS,
    device: str = "cpu",
    seed: int = 42,
    eval_every: int = 0,
    gen_mode: str = DEFAULT_GEN_MODE,
    num_samples: Optional[int] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    t5_d_model: int = DEFAULT_T5_D_MODEL,
    t5_num_heads: int = DEFAULT_T5_HEADS,
    t5_d_ff: int = DEFAULT_T5_D_FF,
    t5_num_layers: int = DEFAULT_T5_LAYERS,
    verbose: bool = True,
):
    """Train a TIGER ablation; ``epochs`` counts optimisation steps (batches).

    ``gen_mode`` selects the decoding strategy used at evaluation time (see
    :func:`evaluate_tiger`): ``'sample'`` (default) for large cutoffs, ``'beam'``
    for exact small-k decoding.
    """
    from torch.optim import AdamW
    from modules.scheduler.inv_sqrt import InverseSquareRootScheduler

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    id_table = build_id_table(
        method=id_method,
        num_items=data.num_items,
        n_layers=n_layers,
        codebook_size=codebook_size,
        embeddings=embeddings,
        seed=seed,
    )
    tokenizer = PrecomputedIdTokenizer(id_table, n_layers).to(device)

    top_k_for_generation = max(max(ks), 1)
    model = build_model(
        tokenizer,
        n_layers,
        codebook_size,
        top_k_for_generation,
        t5_d_model,
        t5_num_heads,
        t5_d_ff,
        t5_num_layers,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = InverseSquareRootScheduler(optimizer=optimizer, warmup_steps=warmup_steps)

    train_examples = build_training_examples(data.train_sequences())
    num_examples = len(train_examples)
    if verbose:
        print(
            f"[TIGER-{id_method}] items={data.num_items} train_pairs={num_examples} "
            f"params={sum(p.numel() for p in model.parameters())}"
        )

    for step in trange(epochs):
        model.train()
        idx = torch.randint(num_examples, (batch_size,), generator=generator).tolist()
        histories = [train_examples[i][0] for i in idx]
        targets = [train_examples[i][1] for i in idx]

        item_ids, seq_mask = _pad_batch(histories, max_len, data.num_items, device)
        target_ids = torch.tensor(targets, dtype=torch.long, device=device)
        user_ids = torch.full((batch_size,), -1, dtype=torch.long, device=device)

        tokenized = tokenizer.tokenize(item_ids, seq_mask, target_ids, user_ids)

        optimizer.zero_grad()
        loss = model(tokenized).loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose and eval_every and (step + 1) % eval_every == 0:
            val_metrics = evaluate_tiger(
                data, model, tokenizer, "val", max_len, ks, device,
                gen_mode=gen_mode, num_samples=num_samples, temperature=temperature,
                seed=seed,
            )
            print(
                f"[TIGER-{id_method}] step {step + 1:>6}/{epochs} "
                f"loss={loss.item():.4f} val: {format_metrics(val_metrics)}"
            )

    return model, tokenizer


def run(
    split: str,
    id_method: str,
    n_layers: int,
    codebook_size: int,
    max_len: int,
    epochs: int,
    ks: List[int],
    gen_mode: str = DEFAULT_GEN_MODE,
    num_samples: Optional[int] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = 42,
    device: str = "cpu",
) -> None:
    data = AmazonSequenceData(split=split)
    resolved_samples = (
        num_samples
        if (gen_mode == GEN_MODE_BEAM or num_samples is not None)
        else default_num_samples(ks)
    )
    print(
        f"[TIGER-{id_method}] split={split} users={len(data)} items={data.num_items} "
        f"gen_mode={gen_mode} num_samples={resolved_samples} temperature={temperature}"
    )

    embeddings = None
    if id_method == "lsh":
        embeddings = load_item_embeddings(split, data.num_items)

    model, tokenizer = train_tiger(
        data,
        id_method=id_method,
        embeddings=embeddings,
        n_layers=n_layers,
        codebook_size=codebook_size,
        max_len=max_len,
        epochs=epochs,
        ks=ks,
        eval_every=max(1, epochs // 10),
        gen_mode=gen_mode,
        num_samples=num_samples,
        temperature=temperature,
        device=device,
        seed=seed,
    )
    eval_kwargs = dict(
        gen_mode=gen_mode, num_samples=num_samples, temperature=temperature
    )
    val_metrics = evaluate_tiger(
        data, model, tokenizer, "val", max_len, ks, device, **eval_kwargs
    )
    test_metrics = evaluate_tiger(
        data, model, tokenizer, "test", max_len, ks, device, **eval_kwargs
    )
    print(f"[TIGER-{id_method}] VAL : {format_metrics(val_metrics)}")
    print(f"[TIGER-{id_method}] TEST: {format_metrics(test_metrics)}")


def load_item_embeddings(split: str, num_items: int) -> Tensor:
    """Load the content embeddings used by the LSH ablation.

    Reuses the repository's processed item features (the same 768-dim sentence-T5
    embeddings that feed RQ-VAE), so LSH operates on identical content.
    """
    from data.processed import ItemData, RecDataset

    item_data = ItemData(
        root="dataset/amazon",
        dataset=RecDataset.AMAZON,
        split=split,
        train_test_split="all",
    )
    embeddings = item_data.item_data[:, :768]
    if embeddings.size(0) != num_items:
        raise ValueError(
            f"Loaded {embeddings.size(0)} item embeddings but expected {num_items}."
        )
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="TIGER ablations on Amazon Reviews.")
    parser.add_argument("--split", type=str, default="beauty")
    parser.add_argument(
        "--id-method", type=str, default="random", choices=["random", "lsh"]
    )
    parser.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--codebook-size", type=int, default=DEFAULT_CODEBOOK_SIZE)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument(
        "--gen-mode",
        type=str,
        default=DEFAULT_GEN_MODE,
        choices=[GEN_MODE_BEAM, GEN_MODE_SAMPLE],
        help="Decoding at eval: 'sample' (large k) or 'beam' (exact, small k).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Candidate samples per user (sample mode). Default scales with max(ks).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Softmax temperature for sampling (>1 flatter, <1 sharper).",
    )
    parser.add_argument(
        "--seed",
        type=float,
        default=42,
        help="Seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device.",
    )
    args = parser.parse_args()
    run(
        split=args.split,
        id_method=args.id_method,
        n_layers=args.n_layers,
        codebook_size=args.codebook_size,
        max_len=args.max_len,
        epochs=args.epochs,
        ks=args.ks,
        gen_mode=args.gen_mode,
        num_samples=args.num_samples,
        temperature=args.temperature,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
