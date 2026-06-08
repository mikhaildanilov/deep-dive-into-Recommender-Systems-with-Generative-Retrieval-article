"""Shared data loading for retrieval baselines.

Reads the Amazon Reviews data that already ships with this repository
(``dataset/amazon/raw/<split>/{sequential_data.txt,datamaps.json,review_splits.pkl}``)
and exposes it in the format used across all baselines.

Item ids are remapped from the raw 1-based ids to a dense 0-based range
``[0, num_items)`` (identical to ``AmazonReviews._remap_ids``).

**Temporal split.** The per-user interaction sequence is split into three
*contiguous, time-ordered* segments according to the global temporal split
produced by ``data_preprocess_amazon_temporal.ipynb`` and stored in
``review_splits.pkl``. That file holds the train/val/test review lists from an
80/10/10 split over ``unixReviewTime`` (with a cold-start guarantee that every
user/item appears in train). Because ``sequential_data.txt`` already lists each
user's items in chronological order, we only need the *number* of validation and
test interactions per user to recover the boundaries::

    train = items[:n_train]
    val   = items[n_train : n_train + n_val]
    test  = items[n_train + n_val :]

Evaluation is **per-interaction**: every held-out validation/test interaction
becomes its own example whose history is the full prefix of items that occurred
before it in time. Users with no held-out interaction in a split simply
contribute no examples to it.

If ``review_splits.pkl`` is absent the loader falls back to the classic
leave-one-out split (``val = items[-2]``, ``test = items[-1]``), which is exactly
the ``n_val = n_test = 1`` special case of the logic above.

The same object serves both non-sequential models (EASE, MF-BPR), which only
need the set of training interactions, and sequential models (SASRec, BERT4Rec),
which consume the ordered ``train`` prefixes.
"""

import json
import os.path as osp
import pickle
from typing import Dict, List, NamedTuple

import torch
from torch import Tensor

DEFAULT_ROOT = "dataset/amazon"
DEFAULT_SPLIT = "beauty"

SEQUENTIAL_DATA_FILE = "sequential_data.txt"
DATAMAPS_FILE = "datamaps.json"
REVIEW_SPLITS_FILE = "review_splits.pkl"

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
    into :attr:`AmazonSequenceData.sequences` (and therefore into the EASE
    user-item matrix / MF-BPR user embedding).
    """

    user: int
    history: List[int]
    target: int
    seen: List[int]


class AmazonSequenceData:
    """Container around the raw Amazon sequential data with a temporal split."""

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
        self.num_items: int = len(datamaps["item2id"])
        self.num_users: int = len(user2id)

        # Per-user chronological sequences, in file (= raw user id) order. The
        # row index is what every downstream model uses to address a user.
        self.user_ids: List[int] = []
        self.sequences: List[List[int]] = []
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

    # -- Split boundaries -------------------------------------------------

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
        return [sorted(set(seq[: self._n_train(row)])) for row, seq in enumerate(self.sequences)]

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


def build_seen_mask(seen_batch: List[List[int]], num_items: int, device) -> Tensor:
    """Boolean ``[B, num_items]`` mask that is ``True`` for already-seen items."""
    mask = torch.zeros((len(seen_batch), num_items), dtype=torch.bool, device=device)
    for row, seen in enumerate(seen_batch):
        if seen:
            mask[row, torch.tensor(seen, dtype=torch.long, device=device)] = True
    return mask
