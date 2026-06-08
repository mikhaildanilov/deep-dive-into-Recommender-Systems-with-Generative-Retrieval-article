from pathlib import Path
import argparse
import yaml

import baselines.ease.train


def dummy_call(_):
    return None


MODEL_TO_RUN_ENTRYPOINT = {
    "EXAMPLE": lambda _: print("Hello, world!"),
    "EASE": baselines.ease.train.run_from_config,
    "MF-BPR": dummy_call,
    "SASRec": dummy_call,
    "BERT4Rec": dummy_call,
    "Tiger with Random IDs": dummy_call,
    "TIGER with LSH IDs": dummy_call,
    "TIGER with RQ-VAE": dummy_call,
}


def run_from_config(config):
    print("Starting end-to-end training. Parsed configuration:")
    for k, v in config.items():
        print(f"{k}: {v}")

    if "model" not in config:
        raise ValueError("Configuration must specify 'model' field")
    if config["model"] not in MODEL_TO_RUN_ENTRYPOINT:
        raise ValueError(
            f"{config["model"]} is not valid 'model' name. Valid names are: {list(MODEL_TO_RUN_ENTRYPOINT.keys())}"
        )

    MODEL_TO_RUN_ENTRYPOINT[config["model"]](config)


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end training from a YAML config file."
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML config file, e.g. config/example.yaml",
    )

    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_from_config(config)


if __name__ == "__main__":
    main()
