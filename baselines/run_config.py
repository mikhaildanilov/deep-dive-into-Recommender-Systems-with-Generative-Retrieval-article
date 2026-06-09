"""Config-driven launcher: pick and run a single baseline from a gin config.

Mirrors the repository's existing gin workflow (see ``configs/*.gin`` and
``modules.utils.parse_config``). A config file selects the baseline via
``run_baseline.model`` and sets that model's hyperparameters; every parameter is
read from the config so all models are tuned consistently in one place.

    python -m baselines.run_config config/baselines/ease.gin
    python -m baselines.run_config config/baselines/sasrec.gin

The single ``run_baseline`` function holds the union of all baselines'
hyperparameters with sensible defaults; each config only sets the ones relevant
to its model, and the dispatcher passes the right subset to the matching trainer.
"""

import argparse
from typing import Callable, List

import gin

from baselines.data import make_sequence_data
from baselines.metrics import format_metrics

from baselines.ease import run as ease_run
from baselines.mf_bpr import train_mf_bpr, evaluate as mf_evaluate
from baselines.sasrec import train_sasrec
from baselines.sequential import evaluate_next_item
from baselines.bert4rec import train_bert4rec, evaluate as bert_evaluate
from baselines.tiger import (
    train_tiger,
    evaluate_tiger,
    load_item_embeddings,
    DEFAULT_GEN_MODE as TIGER_DEFAULT_GEN_MODE,
    DEFAULT_TEMPERATURE as TIGER_DEFAULT_TEMPERATURE,
)

VALID_MODELS = ["ease", "mf_bpr", "sasrec", "bert4rec", "tiger_random", "tiger_lsh"]

# Per-model fallbacks for the shared training-budget knobs, used only when the
# config leaves them unset (``None``).
DEFAULT_EPOCHS = {
    "mf_bpr": 50,
    "sasrec": 100,
    "bert4rec": 100,
    "tiger_random": 20000,
    "tiger_lsh": 20000,
}
DEFAULT_LR = {
    "mf_bpr": 0.01,
    "sasrec": 0.001,
    "bert4rec": 0.001,
    "tiger_random": 0.001,
    "tiger_lsh": 0.001,
}
DEFAULT_BATCH_SIZE = {
    "mf_bpr": 2048,
    "sasrec": 128,
    "bert4rec": 128,
    "tiger_random": 256,
    "tiger_lsh": 256,
}


def _default(value, fallback):
    return fallback if value is None else value


def _eval_and_report(name: str, eval_fn: Callable[[str], dict]) -> None:
    """Evaluate a trained model on val + test and print both metric rows."""
    val_metrics = eval_fn("val")
    test_metrics = eval_fn("test")
    print(f"[{name}] VAL : {format_metrics(val_metrics)}")
    print(f"[{name}] TEST: {format_metrics(test_metrics)}")


@gin.configurable
def run_baseline(
    model: str = None,
    dataset: str = "amazon",
    split: str = "beauty",
    ks: List[int] = [5, 10, 50, 100],
    seed: int = 42,
    device: str = "cpu",
    # Shared training budget (None -> per-model default).
    epochs: int = None,
    lr: float = None,
    batch_size: int = None,
    dim: int = 64,
    # EASE.
    ease_reg: float = 250.0,
    ease_tune: bool = False,
    # MF-BPR.
    mf_weight_decay: float = 1e-5,
    # Transformers (SASRec / BERT4Rec).
    num_layers: int = 2,
    num_heads: int = 2,
    max_len: int = 50,
    dropout: float = 0.2,
    mask_prob: float = 0.2,
    # TIGER ablations.
    n_layers: int = 3,
    codebook_size: int = 256,
    weight_decay: float = 0.01,
    warmup_steps: int = 10000,
    t5_d_model: int = 128,
    t5_num_heads: int = 6,
    t5_d_ff: int = 1024,
    t5_num_layers: int = 4,
    # TIGER decoding (how next-item candidates are generated at eval time).
    gen_mode: str = TIGER_DEFAULT_GEN_MODE,
    num_samples: int = None,
    temperature: float = TIGER_DEFAULT_TEMPERATURE,
) -> None:
    """Dispatch to the baseline named by ``model`` using config hyperparameters."""
    if model is None:
        raise ValueError(
            "Config must set run_baseline.model to one of " f"{VALID_MODELS}."
        )
    if model not in VALID_MODELS:
        raise ValueError(f"Unknown model {model!r}; valid options: {VALID_MODELS}.")

    ks = list(ks)
    data = make_sequence_data(dataset, split)
    print(
        f"[run_baseline] model={model} dataset={dataset} split={split} "
        f"users={len(data)} items={data.num_items} ks={ks}"
    )

    if model == "ease":
        # EASE is closed-form: its own runner handles train + val/test + print.
        ease_run(split=split, reg=ease_reg, ks=ks, tune=ease_tune, dataset=dataset)
        return

    epochs = _default(epochs, DEFAULT_EPOCHS[model])
    lr = _default(lr, DEFAULT_LR[model])
    batch_size = _default(batch_size, DEFAULT_BATCH_SIZE[model])

    if model == "mf_bpr":
        m = train_mf_bpr(
            data,
            dim=dim,
            epochs=epochs,
            lr=lr,
            reg=mf_weight_decay,
            batch_size=batch_size,
            ks=ks,
            device=device,
            seed=seed,
        )
        _eval_and_report("MF-BPR", lambda s: mf_evaluate(data, m, s, ks, device))

    elif model == "sasrec":
        m = train_sasrec(
            data,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_len=max_len,
            epochs=epochs,
            lr=lr,
            dropout=dropout,
            batch_size=batch_size,
            ks=ks,
            device=device,
            seed=seed,
        )
        m.eval()
        _eval_and_report(
            "SASRec",
            lambda s: evaluate_next_item(
                data, m.score_at_last, s, max_len, ks, device
            ),
        )

    elif model == "bert4rec":
        m = train_bert4rec(
            data,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_len=max_len,
            epochs=epochs,
            lr=lr,
            dropout=dropout,
            mask_prob=mask_prob,
            batch_size=batch_size,
            ks=ks,
            device=device,
            seed=seed,
        )
        _eval_and_report(
            "BERT4Rec", lambda s: bert_evaluate(data, m, s, max_len, ks, device)
        )

    else:  # tiger_random / tiger_lsh
        id_method = "random" if model == "tiger_random" else "lsh"
        embeddings = (
            load_item_embeddings(data, split, dataset=dataset)
            if id_method == "lsh"
            else None
        )
        m, tokenizer = train_tiger(
            data,
            id_method=id_method,
            embeddings=embeddings,
            n_layers=n_layers,
            codebook_size=codebook_size,
            max_len=max_len,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            ks=ks,
            device=device,
            seed=seed,
            # eval_every=max(1, epochs // 10),
            eval_every=epochs,
            gen_mode=gen_mode,
            num_samples=num_samples,
            temperature=temperature,
            t5_d_model=t5_d_model,
            t5_num_heads=t5_num_heads,
            t5_d_ff=t5_d_ff,
            t5_num_layers=t5_num_layers,
        )
        _eval_and_report(
            f"TIGER-{id_method}",
            lambda s: evaluate_tiger(
                data, m, tokenizer, s, max_len, ks, device,
                gen_mode=gen_mode, num_samples=num_samples, temperature=temperature,
                seed=seed,
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single retrieval baseline from a gin config file."
    )
    parser.add_argument(
        "config_path", type=str, help="Path to a gin config (config/baselines/*.gin)."
    )
    args = parser.parse_args()

    gin.parse_config_file(args.config_path)
    print(f"[run_config] loaded {args.config_path}")
    run_baseline()


if __name__ == "__main__":
    main()
