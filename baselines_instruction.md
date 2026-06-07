# Retrieval Baselines — Launch Instructions

This document explains how to run the retrieval baselines on the **Amazon
Reviews** dataset that ships with this repository and how to read the reported
metrics (**Recall@k**, **NDCG@k**).

All baselines:

| Model              | Type                          | Config                              |
|--------------------|-------------------------------|-------------------------------------|
| **EASE**           | closed-form linear            | `config/baselines/ease.gin`         |
| **MF-BPR**         | matrix factorization (BPR)    | `config/baselines/mf_bpr.gin`       |
| **SASRec**         | causal Transformer            | `config/baselines/sasrec.gin`       |
| **BERT4Rec**       | bidirectional Transformer     | `config/baselines/bert4rec.gin`     |
| **TIGER (Random)** | TIGER + random integer codes  | `config/baselines/tiger_random.gin` |
| **TIGER (LSH)**    | TIGER + LSH (SimHash) codes   | `config/baselines/tiger_lsh.gin`    |

---

## 1. Prerequisites

* The Amazon raw data must be present under
  `dataset/amazon/raw/<split>/` (`sequential_data.txt`, `datamaps.json`, …).
  The `beauty`, `sports` and `toys` splits are already included.
* Use the project virtual environment. On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> The **TIGER (LSH)** baseline additionally reads the processed item content
> embeddings (`dataset/amazon/processed/data_<split>.pt`). If they are missing
> for your split, run the decoder/tokenizer pipeline once (see
> `Launch_instruction.md`) so the processed file is generated.

---

## 2. Recommended: run a baseline from a config

Every baseline is launched the same way — pass a gin config that selects the
model and sets its hyperparameters. This keeps all models tuned consistently in
one place.

```powershell
# EASE
.\.venv\Scripts\python.exe -m baselines.run_config config/baselines/ease.gin

# MF-BPR
.\.venv\Scripts\python.exe -m baselines.run_config config/baselines/mf_bpr.gin

# SASRec
.\.venv\Scripts\python.exe -m baselines.run_config config/baselines/sasrec.gin

# BERT4Rec
.\.venv\Scripts\python.exe -m baselines.run_config config/baselines/bert4rec.gin

# TIGER with Random IDs
.\.venv\Scripts\python.exe -m baselines.run_config config/baselines/tiger_random.gin

# TIGER with LSH IDs
.\.venv\Scripts\python.exe -m baselines.run_config config/baselines/tiger_lsh.gin
```

On Linux/macOS replace `.\.venv\Scripts\python.exe` with `python`.

### Changing the dataset split or hyperparameters

Edit the chosen `.gin` file. Every knob is a `run_baseline.*` binding, e.g.:

```python
run_baseline.split = "sports"     # beauty | sports | toys
run_baseline.ks = [5, 10, 20]     # evaluation cutoffs k
run_baseline.epochs = 200
run_baseline.lr = 0.001
```

The full list of available parameters (with defaults) is documented in
`baselines/run_config.py` (the `run_baseline` function).

---

## 3. Output format

Each run prints the configuration and the metrics on the **validation** and
**test** splits using the leave-one-out protocol (test target = last item,
validation target = second-to-last item):

```
[run_baseline] model=ease split=beauty users=22363 items=12101 ks=[5, 10]
[EASE] VAL : recall@5=0.0512, ndcg@5=0.0361, recall@10=0.0788, ndcg@10=0.0449
[EASE] TEST: recall@5=0.0473, ndcg@5=0.0331, recall@10=0.0742, ndcg@10=0.0418
```

* `recall@k` — fraction of users whose held-out item is in the top-k
  (equivalent to Hit@k under a single held-out target).
* `ndcg@k`   — Normalized Discounted Cumulative Gain at k.

Already-seen items (the user's history) are excluded from the ranking.

---

## 4. Alternative: run a baseline directly (no config)

Each baseline is also a standalone module with its own CLI flags — handy for
quick experiments without editing a config:

```powershell
.\.venv\Scripts\python.exe -m baselines.ease      --split beauty --reg 250.0
.\.venv\Scripts\python.exe -m baselines.ease      --split beauty --tune        # grid-search reg on val
.\.venv\Scripts\python.exe -m baselines.mf_bpr    --split beauty --epochs 50 --dim 64
.\.venv\Scripts\python.exe -m baselines.sasrec    --split beauty --epochs 100 --dim 64 --max-len 50
.\.venv\Scripts\python.exe -m baselines.bert4rec  --split beauty --epochs 100 --dim 64 --max-len 50
.\.venv\Scripts\python.exe -m baselines.tiger     --split beauty --id-method random --epochs 20000
.\.venv\Scripts\python.exe -m baselines.tiger     --split beauty --id-method lsh    --epochs 20000

# TIGER with long recommendation lists (k = 50, 100) via stochastic sampling:
.\.venv\Scripts\python.exe -m baselines.tiger --split beauty --id-method random `
    --gen-mode sample --ks 5 10 50 100 --num-samples 1000 --temperature 1.0
```

There is also a convenience launcher that can run several baselines in a row:

```powershell
.\.venv\Scripts\python.exe -m baselines.run_all --model all   --split beauty
.\.venv\Scripts\python.exe -m baselines.run_all --model sasrec --split beauty --epochs 100
```

---

## 5. Quick sanity check (no Amazon data needed)

A fast smoke test trains every baseline for a few steps on a tiny synthetic
dataset and verifies the metric plumbing. Use it to confirm the code runs end to
end before launching real (longer) training:

```powershell
.\.venv\Scripts\python.exe -m baselines.smoke_test
```

Expected tail of the output:

```
All available smoke tests passed.
```

---

## 6. Notes on the TIGER ablations

Both TIGER ablations reuse the **exact same** generative-retrieval model as the
main TIGER pipeline (`modules/model.py`,
`EncoderDecoderRetrievalModel`) and only change how each item's discrete code
tuple is produced:

* **Random IDs** — codes are drawn uniformly at random (no item content used).
* **LSH IDs** — codes come from SimHash of the item content embeddings (the same
  768-dim sentence-T5 features that feed RQ-VAE).

This isolates the contribution of RQ-VAE's content-based quantization, matching
the ablations in the original paper.

> For **LSH**, `codebook_size` must be a power of two (it is bit-packed from
> `log2(codebook_size)` random hyperplanes per level). The default `256` works.

---

## 7. Long recommendation lists (large k = 50, 100, ...)

The **non-generative** baselines (EASE, MF-BPR, SASRec, BERT4Rec) score the
**entire** item catalogue (`[B, num_items]`) on every evaluation, so any cutoff
up to `num_items` is supported for free. The default cutoffs are now
`ks = [5, 10, 50, 100]`; edit `run_baseline.ks` (config) or pass `--ks 5 10 50
100` (CLI) to change them.

The **generative** TIGER ablations do not score the whole catalogue — they
*generate* the next item's codes autoregressively. Two decoding strategies are
available, selected by `gen_mode`:

| `gen_mode` | What it does                                              | Good for          |
|------------|-----------------------------------------------------------|-------------------|
| `"beam"`   | exact sampling-based beam search (the model's own search) | small k (≤ 20)    |
| `"sample"` | stochastic autoregressive sampling of many code tuples    | large k (50, 100) |

**Sampling** (`gen_mode="sample"`, the default) draws `num_samples` full code
tuples per user — one token per hierarchy from the (temperature-scaled) softmax
— then deduplicates them and ranks the pool by cumulative log-probability. This
produces a large but *approximate* candidate set, which is what makes
Recall@50 / NDCG@100 tractable for the generative model. Relevant knobs:

```python
run_baseline.gen_mode = "sample"   # or "beam"
run_baseline.num_samples = None    # None -> auto-scales to comfortably cover max(ks)
run_baseline.temperature = 1.0     # >1 = more diverse candidates, <1 = sharper
```

or on the CLI:

```powershell
.\.venv\Scripts\python.exe -m baselines.tiger --split beauty --id-method lsh `
    --gen-mode sample --ks 5 10 50 100 --num-samples 1000 --temperature 1.2
```

> `gen_mode="beam"` raises a clear error if asked for `max(k) > 20`; switch to
> `gen_mode="sample"` for long lists. Larger `num_samples` improves recall at big
> k at the cost of more decoder passes.
