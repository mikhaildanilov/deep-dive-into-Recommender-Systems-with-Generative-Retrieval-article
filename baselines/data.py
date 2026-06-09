"""Shared data loading for retrieval baselines.

Two datasets are exposed behind one common interface (:class:`SequenceData`):

* **Amazon Reviews** -- reads the data that already ships with this repository
  (``dataset/amazon/raw/<split>/{sequential_data.txt,datamaps.json,review_splits.pkl}``).
  The per-user sequence is split into three *contiguous, time-ordered* segments
  according to the global 80/10/10 temporal split stored in ``review_splits.pkl``
  (see ``data_preprocess_amazon_temporal.ipynb``).
* **MovieLens-1M** -- reads the raw ratings downloaded by the main pipeline
  (``dataset/ml-1m/raw/ratings.dat``) and applies a *global* temporal 80/10/10
  split over the rating timestamps: two global thresholds (the 80th and 90th
  timestamp percentiles) cut every user's chronological history into
  ``train`` / ``val`` / ``test``. This mirrors the temporal protocol used by the
  main RQ-VAE pipeline (``PreprocessingMixin._ordered_train_test_split``), which
  also thresholds on a global timestamp quantile rather than per user.

In both cases item ids are remapped to a dense 0-based range ``[0, num_items)``
and evaluation is **per-interaction**: every held-out validation/test
interaction becomes its own example whose history is the full prefix of items
that occurred before it in time. Users with no held-out interaction in a split
simply contribute no examples to it.

If Amazon's ``review_splits.pkl`` is absent the loader falls back to the classic
leave-one-out split (``val = items[-2]``, ``test = items[-1]``), the
``n_val = n_test = 1`` special case of the logic above.

The same object serves both non-sequential models (EASE, MF-BPR), which only
need the set of training interactions, and sequential models (SASRec, BERT4Rec,
TIGER), which consume the ordered ``train`` prefixes.
"""

import json
import os.path as osp
import pickle
from typing import Dict, List, NamedTuple, Optional

import torch
from torch import Tensor

DEFAULT_ROOT = "dataset/amazon"
DEFAULT_SPLIT = "beauty"
ML1M_DEFAULT_ROOT = "dataset/ml-1m"

SEQUENTIAL_DATA_FILE = "sequential_data.txt"
DATAMAPS_FILE = "datamaps.json"
REVIEW_SPLITS_FILE = "review_splits.pkl"

ML1M_RATINGS_FILE = "ratings.dat"
ML1M_RATING_HEADERS = ["userId", "movieId", "rating", "timestamp"]
# Global temporal split: items with timestamp <= the 80th percentile are train,
# the next decile (80-90th) is validation, and the top decile is test.
ML1M_TRAIN_QUANTILE = 0.8
ML1M_VAL_QUANTILE = 0.9
# k-core density filter applied to ml-1m (matches the >=5 occurrence filter used
# by the main pipeline's ``PreprocessingMixin._remove_low_occurrence``).
ML1M_MIN_INTERACTIONS = 5

# Minimum sequence length required to build a leave-one-out fallback split
# (train prefix + val target + test target). The Amazon data is pre-filtered to
# >= 5 interactions per user.
MIN_SEQUENCE_LENGTH = 3


class EvalExample(NamedTuple):
    """A single evaluation example.

    ``history`` is the ordered list of 0-based item ids the model may condition
    on (everything that happened before the target in time), ``target`` is the
    held-out next item, and ``seen`` is the set of items excluded from the
    ranking (everything in the history). ``user`` is the dense 0-based row index
    into :attr:`SequenceData.sequences` (and therefore into the EASE user-item
    matrix / MF-BPR user embedding).
    """

    user: int
    history: List[int]
    target: int
    seen: List[int]


class SequenceData:
    """Dataset-agnostic container around per-user chronological sequences.

    Subclasses populate, in ``__init__``, the following attributes:

    * ``num_items`` / ``num_users`` -- catalogue and user counts,
    * ``user_ids`` -- raw user id per row (unused by the ranking models, which
      address users by row index),
    * ``sequences`` -- per-user item ids in chronological order, with the
      ``n_train`` train items first, then ``n_val`` validation items, then
      ``n_test`` test items,
    * ``_n_val`` / ``_n_test`` -- per-row held-out counts.

    Every split/view method below is shared and assumes that contiguous,
    time-ordered layout.
    """

    num_items: int
    num_users: int
    user_ids: List[int]
    sequences: List[List[int]]
    _n_val: List[int]
    _n_test: List[int]

    # -- Split boundaries -------------------------------------------------

    def _n_train(self, row: int) -> int:
        return len(self.sequences[row]) - self._n_val[row] - self._n_test[row]

    def __len__(self) -> int:
        return len(self.sequences)

    # -- Sequential views -------------------------------------------------

    def train_sequences(self) -> List[List[int]]:
        """Per-user training prefixes (the train-period items only)."""
        return [seq[: self._n_train(row)] for row, seq in enumerate(self.sequences)]

    def eval_examples(self, split: str) -> List[EvalExample]:
        """Build per-interaction examples for ``split`` in {"val", "test"}.

        For every held-out interaction at position ``p`` in a user's sequence the
        example conditions on the full prefix ``items[:p]`` and predicts
        ``items[p]``. ``val`` positions are the ``n_val`` items right after the
        train prefix; ``test`` positions are the remaining tail.
        """
        if split not in ("val", "test"):
            raise ValueError(f"split must be 'val' or 'test', got {split!r}.")

        examples: List[EvalExample] = []
        for row, seq in enumerate(self.sequences):
            n_train = self._n_train(row)
            if split == "val":
                positions = range(n_train, n_train + self._n_val[row])
            else:
                positions = range(n_train + self._n_val[row], len(seq))
            for pos in positions:
                history = seq[:pos]
                examples.append(
                    EvalExample(
                        user=row,
                        history=history,
                        target=seq[pos],
                        seen=list(history),
                    )
                )
        return examples

    # -- Interaction (bag-of-items) views ---------------------------------

    def train_interactions(self) -> List[List[int]]:
        """Per-user *sets* of training items, used by EASE / MF-BPR."""
        return [
            sorted(set(seq[: self._n_train(row)]))
            for row, seq in enumerate(self.sequences)
        ]

    def user_item_matrix(self) -> Tensor:
        """Dense binary user-item matrix built from the training interactions.

        Shape ``[num_users_present, num_items]``. Rows align with
        :attr:`sequences` / the ``user`` field of :meth:`eval_examples`.
        """
        num_present = len(self.sequences)
        matrix = torch.zeros((num_present, self.num_items), dtype=torch.float32)
        for row, items in enumerate(self.train_interactions()):
            if items:
                matrix[row, torch.tensor(items, dtype=torch.long)] = 1.0
        return matrix


class AmazonSequenceData(SequenceData):
    """Amazon Reviews sequences with the 80/10/10 review-time temporal split."""

    def __init__(self, root: str = DEFAULT_ROOT, split: str = DEFAULT_SPLIT) -> None:
        self.root = root
        self.split = split

        raw_dir = osp.join(root, "raw", split)
        datamaps_path = osp.join(raw_dir, DATAMAPS_FILE)
        sequences_path = osp.join(raw_dir, SEQUENTIAL_DATA_FILE)
        review_splits_path = osp.join(raw_dir, REVIEW_SPLITS_FILE)
        if not osp.exists(sequences_path):
            raise FileNotFoundError(
                f"Could not find {sequences_path!r}. Expected the Amazon raw data "
                f"to be present under {raw_dir!r}."
            )

        with open(datamaps_path, "r") as f:
            datamaps = json.load(f)
        # Raw ids are 1-based and contiguous, so the number of items equals the
        # size of item2id and remapping is a simple -1 shift.
        user2id = datamaps["user2id"]
        self.num_items = len(datamaps["item2id"])
        self.num_users = len(user2id)

        # Per-user chronological sequences, in file (= raw user id) order. The
        # row index is what every downstream model uses to address a user.
        self.user_ids = []
        self.sequences = []
        rawid_to_row: Dict[int, int] = {}
        with open(sequences_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parsed = list(map(int, line.split()))
                raw_user, raw_items = parsed[0], parsed[1:]
                rawid_to_row[raw_user] = len(self.sequences)
                self.user_ids.append(raw_user - 1)
                self.sequences.append([item - 1 for item in raw_items])

        # Number of validation / test interactions held out per user.
        self._n_val, self._n_test = self._held_out_counts(
            review_splits_path, user2id, rawid_to_row
        )

    def _held_out_counts(
        self,
        review_splits_path: str,
        user2id: Dict[str, str],
        rawid_to_row: Dict[int, int],
    ) -> tuple:
        """Return ``(n_val, n_test)`` lists, one entry per user row.

        Counts are derived from ``review_splits.pkl`` (the temporal split). When
        that file is missing we fall back to a leave-one-out split.
        """
        n_users = len(self.sequences)

        if not osp.exists(review_splits_path):
            # Leave-one-out fallback: last item -> test, second-to-last -> val.
            n_val, n_test = [], []
            for seq in self.sequences:
                holds = 1 if len(seq) >= MIN_SEQUENCE_LENGTH else 0
                n_val.append(holds)
                n_test.append(holds)
            return n_val, n_test

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

        # Clamp so the three segments stay non-negative and contiguous even if
        # the counts and the sequence file are ever slightly out of sync.
        for row, seq in enumerate(self.sequences):
            seq_len = len(seq)
            nt = min(n_test[row], seq_len)
            nv = min(n_val[row], seq_len - nt)
            n_test[row] = nt
            n_val[row] = nv
        return n_val, n_test


class MovieLens1MSequenceData(SequenceData):
    """MovieLens-1M sequences with a global temporal 80/10/10 split.

    Reads ``<root>/raw/ratings.dat`` (downloaded by the main pipeline via
    ``torch_geometric``'s ``MovieLens1M``), applies a k-core density filter, then
    cuts each user's chronological history at two *global* timestamp thresholds.
    """

    def __init__(
        self,
        root: str = ML1M_DEFAULT_ROOT,
        split: Optional[str] = None,
        train_quantile: float = ML1M_TRAIN_QUANTILE,
        val_quantile: float = ML1M_VAL_QUANTILE,
        min_interactions: int = ML1M_MIN_INTERACTIONS,
    ) -> None:
        import pandas as pd

        self.root = root
        ratings_path = osp.join(root, "raw", ML1M_RATINGS_FILE)
        if not osp.exists(ratings_path):
            raise FileNotFoundError(
                f"Could not find {ratings_path!r}. Download MovieLens-1M first, "
                f"e.g. by running the main RQ-VAE pipeline once with a ml-1m gin "
                f"config (it fetches the raw data into {osp.join(root, 'raw')!r})."
            )

        df = pd.read_csv(
            ratings_path,
            sep="::",
            header=None,
            names=ML1M_RATING_HEADERS,
            engine="python",
            encoding="ISO-8859-1",
        )
        df = self._k_core_filter(df, min_interactions)

        # Dense, deterministic item remap (sorted by raw movieId). The inverse
        # map (dense id -> raw movieId) lets the LSH ablation build content
        # embeddings aligned to exactly this item universe.
        unique_items = sorted(int(m) for m in df["movieId"].unique())
        item2id = {movie_id: idx for idx, movie_id in enumerate(unique_items)}
        self.dense_id_to_movie: List[int] = unique_items
        self.num_items = len(unique_items)

        # Two global temporal thresholds define the contiguous train/val/test
        # boundaries shared by every user.
        t_train = df["timestamp"].quantile(train_quantile)
        t_val = df["timestamp"].quantile(val_quantile)

        df = df.sort_values(["userId", "timestamp", "movieId"])
        self.user_ids = []
        self.sequences = []
        n_val: List[int] = []
        n_test: List[int] = []
        for _, group in df.groupby("userId", sort=True):
            timestamps = group["timestamp"].to_numpy()
            items = [item2id[int(m)] for m in group["movieId"].to_numpy()]
            row = len(self.sequences)
            self.user_ids.append(row)
            self.sequences.append(items)
            n_val.append(int(((timestamps > t_train) & (timestamps <= t_val)).sum()))
            n_test.append(int((timestamps > t_val).sum()))

        self.num_users = len(self.sequences)
        self._n_val, self._n_test = n_val, n_test

    @staticmethod
    def _k_core_filter(df, k: int):
        """Iteratively drop users/items with fewer than ``k`` interactions."""
        while True:
            user_counts = df["userId"].value_counts()
            item_counts = df["movieId"].value_counts()
            keep_users = user_counts[user_counts >= k].index
            keep_items = item_counts[item_counts >= k].index
            filtered = df[df["userId"].isin(keep_users) & df["movieId"].isin(keep_items)]
            if len(filtered) == len(df):
                return filtered
            df = filtered


def make_sequence_data(
    dataset: str = "amazon",
    split: str = DEFAULT_SPLIT,
    root: Optional[str] = None,
) -> SequenceData:
    """Factory selecting the sequence container for ``dataset``.

    ``dataset`` is one of ``"amazon"`` or ``"ml-1m"`` (aliases accepted). ``split``
    is the Amazon sub-split (``beauty`` / ``sports`` / ``toys``) and is ignored
    for MovieLens-1M, which is a single dataset.
    """
    name = (dataset or "amazon").lower()
    if name in ("amazon", "amazon_reviews", "reviews"):
        return AmazonSequenceData(root=root or DEFAULT_ROOT, split=split)
    if name in ("ml-1m", "ml1m", "movielens1m", "movielens-1m"):
        return MovieLens1MSequenceData(root=root or ML1M_DEFAULT_ROOT)
    raise ValueError(
        f"Unknown dataset {dataset!r}; valid options: 'amazon', 'ml-1m'."
    )


def build_seen_mask(seen_batch: List[List[int]], num_items: int, device) -> Tensor:
    """Boolean ``[B, num_items]`` mask that is ``True`` for already-seen items."""
    mask = torch.zeros((len(seen_batch), num_items), dtype=torch.bool, device=device)
    for row, seen in enumerate(seen_batch):
        if seen:
            mask[row, torch.tensor(seen, dtype=torch.long, device=device)] = True
    return mask
