import gzip
import json
import numpy as np
import os
import os.path as osp
import pandas as pd
import polars as pl
import torch

from collections import defaultdict
from data.preprocessing import PreprocessingMixin
from torch_geometric.data import download_google_url
from torch_geometric.data import extract_zip
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import fs
from typing import Callable
from typing import List
from typing import Literal
from typing import Optional


def parse(path):
    g = gzip.open(path, "r")
    for l in g:
        yield eval(l)


EVAL_STRATEGIES = ("leave-one-out", "temporal-split")


class AmazonReviews(InMemoryDataset, PreprocessingMixin):
    gdrive_id = "1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G"
    gdrive_filename = "P5_data.zip"

    def __init__(
        self,
        root: str,
        split: str,  # 'beauty', 'sports', 'toys'
        eval_strategy: Literal["leave-one-out", "temporal-split"] = "leave-one-out",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        if eval_strategy not in EVAL_STRATEGIES:
            raise ValueError(
                f"Unknown eval_strategy '{eval_strategy}'. "
                f"Supported: {EVAL_STRATEGIES}"
            )
        self.split = split
        self.eval_strategy = eval_strategy
        super(AmazonReviews, self).__init__(
            root, transform, pre_transform, force_reload
        )
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.split]

    @property
    def processed_file_names(self) -> str:
        # Include strategy in filename to avoid cache collisions when
        # switching between strategies on the same split.
        return f"data_{self.split}_{self.eval_strategy}.pt"

    def download(self) -> None:
        path = download_google_url(self.gdrive_id, self.root, self.gdrive_filename)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, "data")
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def _remap_ids(self, x):
        return x - 1

    # ------------------------------------------------------------------
    # Leave-one-out (original behaviour)
    #   train : all items except the last two
    #   eval  : second-to-last item is the target
    #   test  : last item is the target
    # ------------------------------------------------------------------
    def _split_leave_one_out(self, max_seq_len: int = 20):
        splits = ["train", "eval", "test"]
        sequences = {sp: defaultdict(list) for sp in splits}
        user_ids = []

        with open(
            os.path.join(self.raw_dir, self.split, "sequential_data.txt"), "r"
        ) as f:
            for line in f:
                parsed_line = list(map(int, line.strip().split()))
                user_ids.append(parsed_line[0])
                items = [self._remap_ids(id) for id in parsed_line[1:]]

                # Train: everything except the two held-out items.
                # We keep the whole sequence without padding to allow
                # flexible training-time subsampling.
                train_items = items[:-2]
                sequences["train"]["itemId"].append(train_items)
                sequences["train"]["itemId_fut"].append(items[-2])

                # Eval: second-to-last item is the prediction target.
                eval_items = items[-(max_seq_len + 2) : -2]
                sequences["eval"]["itemId"].append(
                    eval_items + [-1] * (max_seq_len - len(eval_items))
                )
                sequences["eval"]["itemId_fut"].append(items[-2])

                # Test: last item is the prediction target.
                test_items = items[-(max_seq_len + 1) : -1]
                sequences["test"]["itemId"].append(
                    test_items + [-1] * (max_seq_len - len(test_items))
                )
                sequences["test"]["itemId_fut"].append(items[-1])

        for sp in splits:
            sequences[sp]["userId"] = user_ids
            sequences[sp] = pl.from_dict(sequences[sp])

        return sequences

    # ------------------------------------------------------------------
    # Temporal split
    #   Reads timestamps from sequential_data.txt (expected format:
    #   <userId> <itemId_1>:<ts_1> <itemId_2>:<ts_2> ... )
    #   and delegates to PreprocessingMixin._ordered_train_test_split.
    #
    #   Fallback: if no timestamps are present, sorts by position index
    #   (i.e., the ordinal position within each user's sequence acts as
    #   a proxy timestamp), which is equivalent to the leave-one-out
    #   split but uses 80/20 proportion rather than a fixed held-out set.
    # ------------------------------------------------------------------
    def _split_temporal(self, max_seq_len: int = 20, train_frac: float = 0.8):
        rows = []  # (userId, itemId, timestamp)

        with open(
            os.path.join(self.raw_dir, self.split, "sequential_data.txt"), "r"
        ) as f:
            for line in f:
                tokens = line.strip().split()
                user_id = int(tokens[0])
                for pos, token in enumerate(tokens[1:]):
                    if ":" in token:
                        raw_item, ts = token.split(":", 1)
                        timestamp = int(ts)
                    else:
                        # No timestamp available: use positional index as proxy.
                        raw_item = token
                        timestamp = pos
                    item_id = self._remap_ids(int(raw_item))
                    rows.append({"userId": user_id, "itemId": item_id, "timestamp": timestamp})

        ratings_df = pl.from_dicts(rows)

        # _generate_user_history returns {"train": tensor_dict, "eval": tensor_dict}
        # It performs the temporal split internally using _ordered_train_test_split.
        history = self._generate_user_history(
            ratings_df,
            features=["itemId"],
            window_size=max_seq_len + 1,  # context + 1 future item
            stride=1,
            train_split=train_frac,
        )

        # Wrap into the same {split: pl.DataFrame} schema expected by process().
        # Note: _generate_user_history already returns tensor dicts, so we store
        # them verbatim; process() checks for this and skips _df_to_tensor_dict.
        sequences = {
            "train": history["train"],
            "eval": history["eval"],
            # temporal-split has no separate test set beyond eval
            "test": history["eval"],
        }
        return sequences

    # ------------------------------------------------------------------
    # Public dispatcher — called by process()
    # ------------------------------------------------------------------
    def train_test_split(self, max_seq_len: int = 20):
        if self.eval_strategy == "leave-one-out":
            return self._split_leave_one_out(max_seq_len=max_seq_len)
        elif self.eval_strategy == "temporal-split":
            return self._split_temporal(max_seq_len=max_seq_len)
        else:
            raise ValueError(f"Unsupported eval_strategy: '{self.eval_strategy}'")

    def process(self, max_seq_len=20) -> None:
        data = HeteroData()

        with open(os.path.join(self.raw_dir, self.split, "datamaps.json"), "r") as f:
            data_maps = json.load(f)

        sequences = self.train_test_split(max_seq_len=max_seq_len)

        # sequences values are either pl.DataFrame (leave-one-out) or
        # already-computed tensor dicts (temporal-split).
        def _to_tensor_dict(seq_val):
            if isinstance(seq_val, pl.DataFrame):
                return self._df_to_tensor_dict(seq_val, ["itemId"])
            # Already a tensor dict produced by _generate_user_history.
            return seq_val

        data["user", "rated", "item"].history = {
            k: _to_tensor_dict(v) for k, v in sequences.items()
        }

        # Compute item features
        asin2id = pd.DataFrame(
            [
                {"asin": k, "id": self._remap_ids(int(v))}
                for k, v in data_maps["item2id"].items()
            ]
        )
        item_data = (
            pd.DataFrame(
                [
                    meta
                    for meta in parse(
                        path=os.path.join(self.raw_dir, self.split, "meta.json.gz")
                    )
                ]
            )
            .merge(asin2id, on="asin")
            .sort_values(by="id")
            .fillna({"brand": "Unknown"})
        )

        sentences = item_data.apply(
            lambda row: (
                "Title: "
                + str(row["title"])
                + "; "
                + "Brand: "
                + str(row["brand"])
                + "; "
                + "Categories: "
                + str(row["categories"][0])
                + "; "
                + "Price: "
                + str(row["price"])
                + "; "
            ),
            axis=1,
        )

        item_emb = self._encode_text_feature(sentences)
        data["item"].x = item_emb
        data["item"].text = np.array(sentences)

        gen = torch.Generator()
        gen.manual_seed(42)
        data["item"].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05

        self.save([data], self.processed_paths[0])