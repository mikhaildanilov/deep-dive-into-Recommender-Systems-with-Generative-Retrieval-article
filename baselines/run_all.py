"""Single entry point to run any retrieval baseline on the Amazon data.

Dispatches to the per-model ``run`` functions so every baseline can be launched
through one consistent CLI with shared evaluation cutoffs.

    python -m baselines.run_all --model ease       --split beauty
    python -m baselines.run_all --model mf_bpr     --split beauty --epochs 50
    python -m baselines.run_all --model sasrec     --split beauty --epochs 100
    python -m baselines.run_all --model bert4rec   --split beauty --epochs 100
    python -m baselines.run_all --model tiger_random --split beauty --epochs 20000
    python -m baselines.run_all --model tiger_lsh    --split beauty --epochs 20000
    python -m baselines.run_all --model all        --split beauty
"""

import argparse
from typing import List

DEFAULT_KS = [5, 10, 50, 100]
SEQUENTIAL_MODELS = {"sasrec", "bert4rec"}
TIGER_MODELS = {"tiger_random", "tiger_lsh"}
ALL_MODELS = ["ease", "mf_bpr", "sasrec", "bert4rec", "tiger_random", "tiger_lsh"]


def _run_one(model: str, split: str, epochs: int, ks: List[int], dataset: str) -> None:
    print(f"\n{'=' * 70}\n>>> {model.upper()}\n{'=' * 70}")

    if model == "ease":
        from baselines.ease import run as ease_run

        ease_run(split=split, reg=250.0, ks=ks, tune=False, dataset=dataset)

    elif model == "mf_bpr":
        from baselines.mf_bpr import run as mf_run

        mf_run(split=split, dim=64, epochs=epochs, lr=0.01, ks=ks, seed=42, dataset=dataset)

    elif model == "sasrec":
        from baselines.sasrec import run as sasrec_run

        sasrec_run(split=split, dim=64, epochs=epochs, max_len=50, ks=ks, dataset=dataset)

    elif model == "bert4rec":
        from baselines.bert4rec import run as bert_run

        bert_run(split=split, dim=64, epochs=epochs, max_len=50, ks=ks, dataset=dataset)

    elif model in TIGER_MODELS:
        from baselines.tiger import run as tiger_run

        id_method = "random" if model == "tiger_random" else "lsh"
        tiger_run(
            split=split,
            id_method=id_method,
            n_layers=3,
            codebook_size=256,
            max_len=20,
            epochs=epochs,
            ks=ks,
            dataset=dataset,
        )
    else:
        raise ValueError(f"Unknown model {model!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval baselines.")
    parser.add_argument(
        "--model", type=str, default="all", choices=ALL_MODELS + ["all"]
    )
    parser.add_argument("--dataset", type=str, default="amazon")
    parser.add_argument("--split", type=str, default="beauty")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optimisation steps/epochs; sensible per-model default if omitted.",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    args = parser.parse_args()

    models = ALL_MODELS if args.model == "all" else [args.model]
    for model in models:
        if args.epochs is not None:
            epochs = args.epochs
        elif model in TIGER_MODELS:
            epochs = 20000
        elif model in SEQUENTIAL_MODELS:
            epochs = 100
        else:
            epochs = 50
        _run_one(model, args.split, epochs, args.ks, args.dataset)


if __name__ == "__main__":
    main()
