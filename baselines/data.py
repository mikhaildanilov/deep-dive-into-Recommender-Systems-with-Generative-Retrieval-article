"""Shared data loading for retrieval baselines.

Reads the Amazon Reviews data that already ships with this repository
(``dataset/amazon/raw/<split>/{sequential_data.txt,datamaps.json}``) and exposes
it in the leave-one-out format used across all baselines:

* item ids are remapped from the raw 1-based ids to a dense 0-based range
  ``[0, num_items)`` (identical to ``AmazonReviews._remap_ids``),
* for every user the full interaction sequence is split into
  ``train = items[:-2]``, ``val_target = items[-2]``, ``test_target = items[-1]``.

The same object serves both non-sequential models (EASE, MF-BPR), which only
need the set of training interactions, and sequential models (SASRec, BERT4Rec),
which consume the ordered ``train`` prefixes.
"""

import json
import os
import os.path as osp
from typing import List, NamedTuple

import torch
from torch import Tensor

DEFAULT_ROOT = "dataset/amazon"
DEFAULT_SPLIT = "beauty"

SEQUENTIAL_DATA_FILE = "sequential_data.txt"
DATAMAPS_FILE = "datamaps.json"

# Minimum sequence length required to build train prefix + val target + test
# target. The Amazon data is pre-filtered to >= 5 interactions per user.
MIN_SEQUENCE_LENGTH = 3


class EvalExample(NamedTuple):
    """A single leave-one-out evaluation example.

    ``history`` is the ordered list of 0-based item ids the model may condition
    on, ``target`` is the held-out next item, and ``seen`` is the set of items
    that must be excluded from the ranking (everything in the history).
    """

    user: int
    history: List[int]
    target: int
    seen: List[int]


class AmazonSequenceData:
    """Container around the raw Amazon sequential data."""

    def __init__(self, root: str = DEFAULT_ROOT, split: str = DEFAULT_SPLIT) -> None:
        self.root = root
        self.split = split

        raw_dir = osp.join(root, "raw", split)
        datamaps_path = osp.join(raw_dir, DATAMAPS_FILE)
        sequences_path = osp.join(raw_dir, SEQUENTIAL_DATA_FILE)
        if not osp.exists(sequences_path):
            raise FileNotFoundError(
                f"Could not find {sequences_path!r}. Expected the Amazon raw data "
                f"to be present under {raw_dir!r}."
            )

        with open(datamaps_path, "r") as f:
            datamaps = json.load(f)

        # Raw ids are 1-based and contiguous, so the number of items equals the
        # size of item2id and remapping is a simple -1 shift.
        self.num_items: int = len(datamaps["item2id"])
        self.num_users: int = len(datamaps["user2id"])

        self.user_ids: List[int] = []
        self.sequences: List[List[int]] = []
        with open(sequences_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parsed = list(map(int, line.split()))
                raw_user, raw_items = parsed[0], parsed[1:]
                if len(raw_items) < MIN_SEQUENCE_LENGTH:
                    continue
                self.user_ids.append(raw_user - 1)
                self.sequences.append([item - 1 for item in raw_items])

    def __len__(self) -> int:
        return len(self.sequences)

    # -- Sequential views -------------------------------------------------

    def train_sequences(self) -> List[List[int]]:
        """Per-user training prefixes (everything but the last two items)."""
        return [items[:-2] for items in self.sequences]

    def eval_examples(self, split: str) -> List[EvalExample]:
        """Build leave-one-out examples for ``split`` in {"val", "test"}.

        * ``val``  conditions on ``items[:-2]`` and predicts ``items[-2]``.
        * ``test`` conditions on ``items[:-1]`` and predicts ``items[-1]``.
        """
        if split not in ("val", "test"):
            raise ValueError(f"split must be 'val' or 'test', got {split!r}.")

        examples: List[EvalExample] = []
        for user, items in zip(self.user_ids, self.sequences):
            if split == "val":
                history = items[:-2]
                target = items[-2]
            else:
                history = items[:-1]
                target = items[-1]
            examples.append(
                EvalExample(user=user, history=history, target=target, seen=list(history))
            )
        return examples

    # -- Interaction (bag-of-items) views ---------------------------------

    def train_interactions(self) -> List[List[int]]:
        """Per-user *sets* of training items, used by EASE / MF-BPR."""
        return [sorted(set(items[:-2])) for items in self.sequences]

    def user_item_matrix(self) -> Tensor:
        """Dense binary user-item matrix built from the training interactions.

        Shape ``[num_users_present, num_items]``. Rows align with
        :attr:`user_ids` / :attr:`sequences` order, *not* with raw user ids.
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
