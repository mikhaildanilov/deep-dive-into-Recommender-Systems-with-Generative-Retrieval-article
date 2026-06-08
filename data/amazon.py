import gzip
import json
import numpy as np
import os
import os.path as osp
import pandas as pd
import pickle
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
from typing import Optional


SEQUENTIAL_DATA_FILE = "sequential_data.txt"
DATAMAPS_FILE = "datamaps.json"
REVIEW_SPLITS_FILE = "review_splits.pkl"
META_FILE = "meta.json.gz"

# Shortest sequence for which the leave-one-out fallback can hold out a
# validation and a test item (train prefix + val + test).
MIN_SEQUENCE_LENGTH = 3


def parse(path):
    g = gzip.open(path, "r")
    for l in g:
        yield eval(l)


class AmazonReviews(InMemoryDataset, PreprocessingMixin):
    gdrive_id = "1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G"
    gdrive_filename = "P5_data.zip"

    def __init__(
        self,
        root: str,
        split: str,  # 'beauty', 'sports', 'toys'
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.split = split
        super(AmazonReviews, self).__init__(
            root, transform, pre_transform, force_reload
        )
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.split]

    @property
    def processed_file_names(self) -> str:
        return f"data_{self.split}.pt"

    def download(self) -> None:
        path = download_google_url(self.gdrive_id, self.root, self.gdrive_filename)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, "data")
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def _remap_ids(self, x):
        return x - 1

    def _temporal_boundaries(self):
        """Per-user contiguous (train, val, test) split sizes.

        Mirrors ``baselines.data.AmazonSequenceData``: validation/test sizes come
        from the global temporal split stored in ``review_splits.pkl`` (an
        80/10/10 split over review time), recovered as per-user interaction
        counts. Since ``sequential_data.txt`` is already time-ordered, the
        boundaries are ``train = items[:n_train]``,
        ``val = items[n_train : n_train + n_val]``, ``test = items[n_train + n_val:]``.
        Falls back to leave-one-out (``n_val = n_test = 1``) when the pickle is
        absent.

        Returns ``(sequences, user_ids, n_val, n_test)`` aligned by user row.
        """
        base = os.path.join(self.raw_dir, self.split)
        with open(os.path.join(base, DATAMAPS_FILE), "r") as f:
            user2id = json.load(f)["user2id"]

        user_ids, sequences, rawid_to_row = [], [], {}
        with open(os.path.join(base, SEQUENTIAL_DATA_FILE), "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parsed = list(map(int, line.split()))
                rawid_to_row[parsed[0]] = len(sequences)
                user_ids.append(parsed[0])
                sequences.append([self._remap_ids(i) for i in parsed[1:]])

        n_users = len(sequences)
        review_splits_path = os.path.join(base, REVIEW_SPLITS_FILE)
        if os.path.exists(review_splits_path):
            with open(review_splits_path, "rb") as f:
                review_splits = pickle.load(f)
            n_val = [0] * n_users
            n_test = [0] * n_users
            for split_name, counts in (("val", n_val), ("test", n_test)):
                for review in review_splits.get(split_name, []):
                    mapped = user2id.get(review["reviewerID"])
                    if mapped is None:
                        continue
                    row = rawid_to_row.get(int(mapped))
                    if row is not None:
                        counts[row] += 1
            # Keep the three segments non-negative and contiguous even if the
            # counts and the sequence file are ever slightly out of sync.
            for row, seq in enumerate(sequences):
                seq_len = len(seq)
                nt = min(n_test[row], seq_len)
                nv = min(n_val[row], seq_len - nt)
                n_test[row], n_val[row] = nt, nv
        else:
            # Leave-one-out fallback (classic P5 split).
            held = [1 if len(s) >= MIN_SEQUENCE_LENGTH else 0 for s in sequences]
            n_val, n_test = held, list(held)

        return sequences, user_ids, n_val, n_test

    def train_test_split(self, max_seq_len=20):
        sequences, user_ids, n_val, n_test = self._temporal_boundaries()
        splits = ["train", "eval", "test"]
        out = {sp: defaultdict(list) for sp in splits}

        def _add_eval(split, row, pos):
            # One example per held-out interaction: history is the full prefix
            # of items that occurred before it in time (capped at max_seq_len).
            history = sequences[row][max(0, pos - max_seq_len) : pos]
            out[split]["itemId"].append(history + [-1] * (max_seq_len - len(history)))
            out[split]["itemId_fut"].append(sequences[row][pos])
            out[split]["userId"].append(user_ids[row])

        for row, seq in enumerate(sequences):
            n_train = len(seq) - n_val[row] - n_test[row]

            # Train: next-item over the train-period prefix only (no leakage).
            # The whole prefix is kept unpadded for flexible training-time
            # subsampling.
            train_items = seq[:n_train]
            if train_items:
                out["train"]["itemId"].append(train_items[:-1])
                out["train"]["itemId_fut"].append(train_items[-1])
                out["train"]["userId"].append(user_ids[row])

            for pos in range(n_train, n_train + n_val[row]):
                _add_eval("eval", row, pos)
            for pos in range(n_train + n_val[row], len(seq)):
                _add_eval("test", row, pos)

        return {sp: pl.from_dict(dict(out[sp])) for sp in splits}

    def process(self, max_seq_len=20) -> None:
        data = HeteroData()

        with open(os.path.join(self.raw_dir, self.split, DATAMAPS_FILE), "r") as f:
            data_maps = json.load(f)

        # Construct user sequences
        sequences = self.train_test_split(max_seq_len=max_seq_len)
        data["user", "rated", "item"].history = {
            k: self._df_to_tensor_dict(v, ["itemId"]) for k, v in sequences.items()
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
                        path=os.path.join(self.raw_dir, self.split, META_FILE)
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


# if __name__ == "__main__":
#    AmazonReviews("dataset/amazon")
