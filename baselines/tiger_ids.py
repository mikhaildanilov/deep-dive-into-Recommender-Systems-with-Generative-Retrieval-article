"""Discrete ID generators for the TIGER ablations.

TIGER represents every item by a tuple of ``n_layers`` integer codes (a
"semantic ID") and trains a sequence-to-sequence model to autoregressively
generate the codes of the next item. The two ablations here replace RQ-VAE
quantization with simpler code-assignment schemes, isolating the contribution
of content-based quantization:

* :func:`random_ids` -- each item gets ``n_layers`` codes drawn uniformly at
  random. No item content is used at all.
* :func:`lsh_ids` -- codes come from Locality-Sensitive Hashing (SimHash) of the
  item content embeddings: random hyperplanes partition the embedding space and
  the sign pattern is packed into per-layer integer codes.

Both return a ``[num_items, n_layers]`` ``LongTensor`` of codes in
``[0, codebook_size)``. :func:`add_dedup_column` then appends the extra column
that TIGER uses to disambiguate items colliding on the same code tuple, yielding
the ``[num_items, n_layers + 1]`` table consumed by the decoder.
"""

import math
from typing import List

import torch
from torch import Tensor


def random_ids(
    num_items: int,
    n_layers: int,
    codebook_size: int,
    seed: int = 42,
) -> Tensor:
    """Uniformly random codes, ``[num_items, n_layers]`` in ``[0, codebook_size)``."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        codebook_size, (num_items, n_layers), generator=generator, dtype=torch.long
    )


def lsh_ids(
    embeddings: Tensor,
    n_layers: int,
    codebook_size: int,
    seed: int = 42,
) -> Tensor:
    """SimHash LSH codes for content embeddings.

    ``codebook_size`` must be a power of two; ``log2(codebook_size)`` random
    hyperplanes are used per layer. Each embedding is projected onto the
    hyperplanes, the signs form a bit string, and the bits are packed into one
    integer code per layer.

    Args:
        embeddings:    ``[num_items, dim]`` content embeddings.
        n_layers:      number of code positions.
        codebook_size: codes per layer (power of two).

    Returns:
        ``[num_items, n_layers]`` LongTensor of codes in ``[0, codebook_size)``.
    """
    bits_per_layer = int(round(math.log2(codebook_size)))
    if 2**bits_per_layer != codebook_size:
        raise ValueError(
            f"lsh_ids requires codebook_size to be a power of two, got {codebook_size}."
        )

    num_items, dim = embeddings.shape
    total_bits = n_layers * bits_per_layer

    generator = torch.Generator().manual_seed(seed)
    hyperplanes = torch.randn(dim, total_bits, generator=generator)

    projections = embeddings.float() @ hyperplanes  # [num_items, total_bits]
    bits = (projections > 0).long().view(num_items, n_layers, bits_per_layer)

    powers = (2 ** torch.arange(bits_per_layer)).long()  # [bits_per_layer]
    codes = (bits * powers).sum(dim=-1)  # [num_items, n_layers]
    return codes


def add_dedup_column(codes: Tensor) -> Tensor:
    """Append TIGER's deduplication column.

    For each item, the appended value counts how many earlier items share the
    exact same code tuple, making every row of the returned
    ``[num_items, n_layers + 1]`` table unique.
    """
    num_items = codes.size(0)
    seen: dict = {}
    dedup: List[int] = []
    for i in range(num_items):
        key = tuple(codes[i].tolist())
        count = seen.get(key, 0)
        dedup.append(count)
        seen[key] = count + 1
    dedup_col = torch.tensor(dedup, dtype=torch.long).unsqueeze(1)
    return torch.cat([codes, dedup_col], dim=1)


def build_id_table(
    method: str,
    num_items: int,
    n_layers: int,
    codebook_size: int,
    embeddings: Tensor = None,
    seed: int = 42,
) -> Tensor:
    """Build the deduplicated ``[num_items, n_layers + 1]`` code table.

    ``method`` is ``"random"`` or ``"lsh"``; ``embeddings`` is required for LSH.
    """
    if method == "random":
        codes = random_ids(num_items, n_layers, codebook_size, seed)
    elif method == "lsh":
        if embeddings is None:
            raise ValueError("LSH id generation requires item embeddings.")
        if embeddings.size(0) != num_items:
            raise ValueError(
                f"Embedding count {embeddings.size(0)} != num_items {num_items}."
            )
        codes = lsh_ids(embeddings, n_layers, codebook_size, seed)
    else:
        raise ValueError(f"Unknown id method {method!r}; use 'random' or 'lsh'.")
    return add_dedup_column(codes)
