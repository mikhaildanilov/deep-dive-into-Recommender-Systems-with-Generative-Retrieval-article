import gin
import numpy as np
import pandas as pd
import polars as pl
import torch
from data.schemas import FUT_SUFFIX
from sentence_transformers import SentenceTransformer
from typing import List


T5_ENCODER_MODELS = {
    "base": "sentence-transformers/sentence-t5-base",
    "large": "sentence-transformers/sentence-t5-large",
    "xl": "sentence-transformers/sentence-t5-xl",
    "xxl": "sentence-transformers/sentence-t5-xxl",
}

DEFAULT_T5_ENCODER_SIZE = "base"


@gin.configurable
def build_text_encoder(model_size: str = DEFAULT_T5_ENCODER_SIZE) -> SentenceTransformer:
    if model_size not in T5_ENCODER_MODELS:
        raise ValueError(
            f"Unsupported T5 encoder size '{model_size}'. "
            f"Supported sizes: {sorted(T5_ENCODER_MODELS)}."
        )
    return SentenceTransformer(T5_ENCODER_MODELS[model_size])


class PreprocessingMixin:
    @staticmethod
    def _process_genres(genres, one_hot=True):
        if one_hot:
            return genres

        max_genres = genres.sum(axis=1).max()
        idx_list = []
        for i in range(genres.shape[0]):
            idxs = np.where(genres[i, :] == 1)[0] + 1
            missing = max_genres - len(idxs)
            if missing > 0:
                idxs = np.array(list(idxs) + missing * [0])
            idx_list.append(idxs)
        return np.stack(idx_list)

    @staticmethod
    def _remove_low_occurrence(source_df, target_df, index_col):
        if isinstance(index_col, str):
            index_col = [index_col]
        out = target_df.copy()
        for col in index_col:
            count = source_df.groupby(col).agg(ratingCnt=("rating", "count"))
            high_occ = count[count["ratingCnt"] >= 5]
            out = out.merge(high_occ, on=col).drop(columns=["ratingCnt"])
        return out

    @staticmethod
    def _encode_text_feature(text_feat, model=None):
        if model is None:
            model = build_text_encoder()
        sentences = [str(s) for s in text_feat]
        return model.encode(
            batch_size=2,
            sentences=sentences,
            show_progress_bar=True,
            convert_to_tensor=True,
        ).cpu()

    @staticmethod
    def _ordered_train_test_split(df, on, train_split=0.8):
        threshold = df.select(pl.quantile(on, train_split)).item()
        return df.with_columns(is_train=pl.col(on) <= threshold)

    @staticmethod
    def _generate_user_history(
        ratings_df,
        features: List[str] = ["movieId", "rating"],
        window_size: int = 200,
        stride: int = 1,
        train_split: float = 0.8,
    ) -> dict:
        """История пользователей через глобальный порог по timestamp.

        Для каждого пользователя:
          train — все события с timestamp <= t_train, fut = -1
          eval  — train-префикс как вход, первое событие после t_train как цель

        features-колонки возвращаются как списки списков (переменная длина) —
        processed.py добивает паддингом через pad_sequence самостоятельно.
        """
        if isinstance(ratings_df, pl.DataFrame):
            ratings_df = ratings_df.to_pandas()

        ratings_df = (
            ratings_df.sort_values(["userId", "timestamp"])
            .reset_index(drop=True)
        )

        t_train = ratings_df["timestamp"].quantile(train_split)

        train_rows, eval_rows = [], []

        for uid, group in ratings_df.groupby("userId", sort=True):
            group = group.sort_values("timestamp")
            mask_train = group["timestamp"] <= t_train

            train_feat = {f: group.loc[mask_train, f].tolist() for f in features}
            eval_feat  = {f: group.loc[~mask_train, f].tolist() for f in features}

            if not train_feat[features[0]]:
                continue

            train_rows.append({
                "userId": uid,
                **{f: train_feat[f] for f in features},
                **{f + FUT_SUFFIX: -1 for f in features},
            })

            if eval_feat[features[0]]:
                eval_rows.append({
                    "userId": uid,
                    **{f: train_feat[f] for f in features},
                    **{f + FUT_SUFFIX: eval_feat[f][0] for f in features},
                })

        if not train_rows:
            raise ValueError(
                "Train split is empty — проверьте данные или train_split."
            )
        if not eval_rows:
            raise ValueError(
                "Eval split is empty — нет пользователей с событиями после t_train."
            )

        def make_split(rows):
            result = {}
            for f in features:
                # список списков — pad_sequence в processed.py ждёт именно это
                result[f] = [r[f] for r in rows]
                result[f + FUT_SUFFIX] = torch.tensor(
                    [r[f + FUT_SUFFIX] for r in rows], dtype=torch.long
                )
            result["userId"] = torch.tensor(
                [r["userId"] for r in rows], dtype=torch.long
            )
            return result

        return {
            "train": make_split(train_rows),
            "eval":  make_split(eval_rows),
        }