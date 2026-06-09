# MovieLens-1M — Launch Instructions

This document explains how to run **everything** on the **MovieLens-1M** dataset:

* the retrieval **baselines** (EASE, MF-BPR, SASRec, BERT4Rec, TIGER-Random,
  TIGER-LSH), and
* the **main generative-retrieval pipeline** — RQ-VAE training
  (`train_rqvae.py`) followed by the TIGER encoder-decoder training
  (`train_decoder.py`).

It is the MovieLens-1M counterpart of `baselines_instruction.md` (Amazon) and
`Launch_instruction.md`.

---

## 1. Prerequisites

### 1.1. Python environment

Use the project virtual environment. On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux/macOS replace `.\.venv\Scripts\python.exe` with `python` in every
command below.

### 1.2. Get the MovieLens-1M raw data

All MovieLens-1M code reads the raw GroupLens files from
`dataset/ml-1m/raw/` (`ratings.dat`, `movies.dat`, `users.dat`).

There are two ways to obtain them:

**Option A — let the pipeline download them (recommended).**
The first time you launch `train_rqvae.py` with a ml-1m config (see §3) the
`torch_geometric` `MovieLens1M` dataset downloads and extracts the archive into
`dataset/ml-1m/raw/` automatically. After that the baselines can read it too.

**Option B — download manually.**
Download <https://files.grouplens.org/datasets/movielens/ml-1m.zip>, then place
the three `*.dat` files so the layout is:

```
dataset/ml-1m/raw/ratings.dat
dataset/ml-1m/raw/movies.dat
dataset/ml-1m/raw/users.dat
```

> The baselines only need `ratings.dat` (and, for **TIGER-LSH**, `movies.dat`
> for item content). They do **not** trigger the download themselves — if the
> file is missing they raise a clear `FileNotFoundError` telling you to fetch it.

---

## 2. Baselines on MovieLens-1M

Every baseline understands the dataset through the shared
`baselines.data.make_sequence_data` factory; selecting MovieLens-1M is just
`dataset = "ml-1m"`.

### 2.1. Temporal split

MovieLens-1M is split with a **global temporal 80/10/10** protocol. Two global
timestamp thresholds — the 80th and 90th percentiles over *all* ratings — cut
every user's chronological history into three contiguous segments:

```
train = interactions with timestamp <= p80
val   = interactions with p80 < timestamp <= p90
test  = interactions with timestamp >  p90
```

A k-core filter (>= 5 interactions per user and per item) is applied first, and
item ids are remapped to a dense `[0, num_items)` range. Evaluation is
**per-interaction**: each held-out val/test interaction becomes its own example
whose history is the full time-ordered prefix before it. This mirrors the
global-quantile temporal split used by the main pipeline
(`PreprocessingMixin._ordered_train_test_split`).

### 2.2. Recommended: run from a config

Each baseline has a ready-made ml-1m gin config under `config/baselines/`:

| Model              | Config                                   |
|--------------------|------------------------------------------|
| **EASE**           | `config/baselines/ease_ml1m.gin`         |
| **MF-BPR**         | `config/baselines/mf_bpr_ml1m.gin`       |
| **SASRec**         | `config/baselines/sasrec_ml1m.gin`       |
| **BERT4Rec**       | `config/baselines/bert4rec_ml1m.gin`     |
| **TIGER (Random)** | `config/baselines/tiger_random_ml1m.gin` |
| **TIGER (LSH)**    | `config/baselines/tiger_lsh_ml1m.gin`    |

```powershell
# EASE
.\.venv\Scripts\python.exe -m baselines.run_config config/baselines/ease_ml1m.gin

# MF-BPR
.\.venv\Scripts\python.exe -m baselines.run_config config/baselines/mf_bpr_ml1m.gin

# SASRec
.\.venv\Scripts\python.exe -m baselines.run_config config/baselines/sasrec_ml1m.gin

# BERT4Rec
.\.venv\Scripts\python.exe -m baselines.run_config config/baselines/bert4rec_ml1m.gin

# TIGER with Random IDs
.\.venv\Scripts\python.exe -m baselines.run_config config/baselines/tiger_random_ml1m.gin

# TIGER with LSH IDs
.\.venv\Scripts\python.exe -m baselines.run_config config/baselines/tiger_lsh_ml1m.gin
```

Every knob is a `run_baseline.*` binding in the `.gin` file. The key one is:

```python
run_baseline.dataset = "ml-1m"    # selects MovieLens-1M
run_baseline.ks = [5, 10, 50, 100]
```

### 2.3. Alternative: run a baseline directly (no config)

Each baseline is also a standalone module that now takes a `--dataset` flag:

```powershell
.\.venv\Scripts\python.exe -m baselines.ease     --dataset ml-1m --reg 250.0
.\.venv\Scripts\python.exe -m baselines.ease     --dataset ml-1m --tune          # grid-search reg on val
.\.venv\Scripts\python.exe -m baselines.mf_bpr   --dataset ml-1m --epochs 50 --dim 64
.\.venv\Scripts\python.exe -m baselines.sasrec   --dataset ml-1m --epochs 100 --dim 64 --max-len 200
.\.venv\Scripts\python.exe -m baselines.bert4rec --dataset ml-1m --epochs 100 --dim 64 --max-len 200
.\.venv\Scripts\python.exe -m baselines.tiger    --dataset ml-1m --id-method random --epochs 20000 --max-len 200
.\.venv\Scripts\python.exe -m baselines.tiger    --dataset ml-1m --id-method lsh    --epochs 20000 --max-len 200
```

Run several baselines in a row:

```powershell
.\.venv\Scripts\python.exe -m baselines.run_all --dataset ml-1m --model all
.\.venv\Scripts\python.exe -m baselines.run_all --dataset ml-1m --model sasrec --epochs 100
```

> MovieLens-1M users have long histories, so the configs use a larger context
> (`max_len = 200`) than the Amazon defaults (`50` for SASRec/BERT4Rec, `20` for
> TIGER).

### 2.4. Output format

```
[run_baseline] model=ease dataset=ml-1m split=beauty users=<U> items=<I> ks=[5, 10, 50, 100]
[EASE] VAL : recall@5=..., ndcg@5=..., recall@10=..., ndcg@10=...
[EASE] TEST: recall@5=..., ndcg@5=..., recall@10=..., ndcg@10=...
```

(`<U>`/`<I>` are the user/item counts after the k-core filter.)

(`split` is printed but ignored for MovieLens-1M, which is a single dataset.)
Already-seen items in each user's history are excluded from the ranking.

---

## 3. Main pipeline — Step 1: RQ-VAE training (`train_rqvae.py`)

This learns the RQ-VAE that turns the 768-dim sentence-T5 item embeddings
(movie title + genres) into Semantic IDs.

```powershell
.\.venv\Scripts\python.exe train_rqvae.py configs/rqvae_ml1m.gin
```

What the config (`configs/rqvae_ml1m.gin`) sets:

* `train.dataset = %data.processed.RecDataset.ML_1M`,
  `train.dataset_folder = "dataset/ml-1m"`,
* `train.force_dataset_process = True` — on the first run it downloads the raw
  MovieLens-1M data and encodes the item text with sentence-T5, caching the
  result under `dataset/ml-1m/processed/`,
* `train.vae_input_dim = 768`, `train.vae_n_cat_feats = 0`,
  `train.vae_n_layers = 3`, `train.vae_codebook_size = 256`,
* checkpoints are written to `out/rqvae/ml1m/checkpoint_<iter>.pt`.

With the default `iterations = 50000` / `save_model_every = 10000`, the final
checkpoint is `out/rqvae/ml1m/checkpoint_49999.pt` — this is the path the
decoder config points at.

> The first run is slow because of the one-time sentence-T5 encoding of all
> movies. Subsequent runs reuse `dataset/ml-1m/processed/data.pt`
> (set `train.force_dataset_process = False` to skip re-processing).

---

## 4. Main pipeline — Step 2: Decoder / TIGER training (`train_decoder.py`)

This trains the T5 encoder-decoder retrieval model on top of the Semantic IDs
produced by the RQ-VAE checkpoint.

```powershell
.\.venv\Scripts\python.exe train_decoder.py configs/decoder_ml1m.gin
```

Before launching, make sure `train.pretrained_rqvae_path` in
`configs/decoder_ml1m.gin` points at the checkpoint produced in §3:

```python
train.pretrained_rqvae_path = "out/rqvae/ml1m/checkpoint_49999.pt"
```

Other notable bindings:

* `train.dataset = %data.processed.RecDataset.ML_1M`,
  `train.dataset_folder = "dataset/ml-1m"`,
* `train.train_data_subsample = False` — MovieLens train sequences are stored as
  fixed-length padded tensors, so the Amazon-style variable-length list
  subsampling is disabled,
* `train.top_k_eval_list = [1, 5, 10, 50, 100]`,
* `train.eval_gen_mode = "sample"` with `train.eval_num_samples = 1000` — uses
  stochastic sampling at eval time so large cutoffs (k = 50, 100) are tractable
  for the generative model; switch to `"beam"` only for small k,
* checkpoints are written to `out/decoder/ml1m/checkpoint_<iter>.pt`.

### Note on the main pipeline's split

The main pipeline builds per-user histories with
`PreprocessingMixin._generate_user_history`, which applies a **global temporal
split** (80th-percentile threshold on each window's `max_timestamp`) and yields
`train` / `eval` partitions (the held-out `eval` interactions act as the test
targets; there is no separate third segment here, unlike the baselines' 80/10/10
split).

---

## 5. End-to-end quick start

```powershell
# 0. (once) install deps
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1. Train RQ-VAE (also downloads + processes ml-1m on first run)
.\.venv\Scripts\python.exe train_rqvae.py configs/rqvae_ml1m.gin

# 2. Train the TIGER encoder-decoder on the RQ-VAE Semantic IDs
#    (edit train.pretrained_rqvae_path first if you changed iterations)
.\.venv\Scripts\python.exe train_decoder.py configs/decoder_ml1m.gin

# 3. (independent) Run the baselines for comparison
.\.venv\Scripts\python.exe -m baselines.run_config config/baselines/ease_ml1m.gin
.\.venv\Scripts\python.exe -m baselines.run_all    --dataset ml-1m --model all
```

---

## 6. Troubleshooting

* **`FileNotFoundError: Could not find 'dataset/ml-1m/raw/ratings.dat'`** —
  the raw data is missing. Run §3 once (auto-download) or fetch it manually
  (§1.2, Option B).
* **TIGER-LSH on ml-1m is slow to start** — it encodes movie content
  (title + genres) with sentence-T5 the first time to build the SimHash codes;
  this is expected.
* **Out of memory / too slow** — reduce `run_baseline.batch_size`,
  `run_baseline.epochs`, or `run_baseline.max_len` in the relevant config, or
  set `run_baseline.device = "cpu"`.
