"""Typed, side-effect-free application stages shared by CLI entry points."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features.aggregate import _ZORUNLU, add_group_statistics
from .features.categorical import FrequencyEncoder
from .features.temporal import add_lag_features, add_rolling_features
from .validation import assert_folds_align

__all__ = [
    "DatasetBundle",
    "FeatureBundle",
    "FoldPlan",
    "build_frequency_features",
    "build_paired_distribution_features",
    "build_paired_history_features",
    "runtime_recipe_fingerprint",
]


@dataclass(frozen=True)
class DatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame | None
    target_column: str
    id_column: str
    sample_submission: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if self.target_column not in self.train.columns:
            raise ValueError(f"train hedef kolonu icermiyor: {self.target_column}")
        if self.id_column not in self.train.columns:
            raise ValueError(f"train ID kolonu icermiyor: {self.id_column}")
        if self.test is not None:
            if self.target_column in self.test.columns:
                raise ValueError(f"test frame hedef kolonunu icermemeli: {self.target_column}")
            if self.id_column not in self.test.columns:
                raise ValueError(f"test ID kolonu icermiyor: {self.id_column}")


@dataclass(frozen=True)
class FeatureBundle:
    train: pd.DataFrame
    test: pd.DataFrame | None
    feature_columns: tuple[str, ...]


@dataclass(frozen=True)
class FoldPlan:
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]
    n_rows: int
    covered: np.ndarray
    fingerprint: str

    @classmethod
    def from_folds(
        cls,
        folds: Sequence[tuple[np.ndarray, np.ndarray]],
        *,
        n_rows: int,
    ) -> FoldPlan:
        normalized = tuple(
            (
                np.asarray(train_idx, dtype=np.int64),
                np.asarray(valid_idx, dtype=np.int64),
            )
            for train_idx, valid_idx in folds
        )
        assert_folds_align(n_rows, normalized)
        covered = np.zeros(n_rows, dtype=bool)
        digest = hashlib.sha256(f"rows:{n_rows}|folds:{len(normalized)}".encode())
        for train_idx, valid_idx in normalized:
            covered[valid_idx] = True
            digest.update(b"|train|")
            digest.update(train_idx.tobytes())
            digest.update(b"|valid|")
            digest.update(valid_idx.tobytes())
        return cls(
            folds=normalized,
            n_rows=n_rows,
            covered=covered,
            fingerprint=digest.hexdigest(),
        )


def build_frequency_features(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    *,
    columns: Sequence[str],
) -> FeatureBundle:
    """Fit frequency mappings once on train and apply them to both frames."""

    encoder = FrequencyEncoder(columns).fit(train)
    train_encoded = encoder.transform(train)
    test_encoded = encoder.transform(test) if test is not None else None
    added = tuple(f"{column}_frekans" for column in columns)
    return FeatureBundle(train_encoded, test_encoded, added)


def _added_columns(before: pd.DataFrame, after: pd.DataFrame) -> tuple[str, ...]:
    return tuple(column for column in after.columns if column not in before.columns)


def build_paired_distribution_features(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    *,
    group_columns: Sequence[str],
    value_columns: Sequence[str],
    frequency_columns: Sequence[str],
    aggregations: Sequence[str] = ("mean", "std"),
    target_column: str | None,
) -> FeatureBundle:
    """Fit group/frequency mappings on train once and reuse them on test."""
    train_grouped = add_group_statistics(
        train,
        group_columns,
        value_columns,
        aggregations=aggregations,
        reference=train,
        target_column=target_column,
    )
    test_grouped = (
        add_group_statistics(
            test,
            group_columns,
            value_columns,
            aggregations=aggregations,
            reference=train,
            target_column=target_column,
        )
        if test is not None
        else None
    )
    encoder = FrequencyEncoder(frequency_columns).fit(train)
    train_encoded = encoder.transform(train_grouped)
    test_encoded = encoder.transform(test_grouped) if test_grouped is not None else None
    return FeatureBundle(
        train_encoded,
        test_encoded,
        _added_columns(train, train_encoded),
    )


def build_paired_history_features(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    *,
    value_column: str,
    time_column: str,
    horizon: int,
    shifts: Sequence[int],
    group_columns: Sequence[str] | None = None,
    rolling_windows: Sequence[int] = (),
    rolling_aggregations: Sequence[str] = ("mean", "std"),
    target_column: str | None = _ZORUNLU,
) -> FeatureBundle:
    """Build train/test history on one chronological axis without target future.

    Test rows are appended only so their timestamps can address the train tail.
    If ``value_column`` is the target, test must not carry it; a NaN placeholder
    guarantees that no test target can enter a lag or rolling window.

    ``target_column`` ACIKCA verilmelidir (``_ZORUNLU`` nobetcisi). Varsayilan
    ``None`` ile birakilsaydi asagidaki koruma opt-in olurdu: cagiran parametreyi
    atlarsa ``target_column == value_column`` hicbir zaman dogru olmaz, test
    hedefleri ayni kronolojik eksene girer ve sonraki test satirlarinin
    lag/rolling penceresine sessizce yazilir. Bu, ``features.aggregate``te
    olculmus ve nobetciyle kapatilmis olan hatanin ta kendisidir; kardes
    fonksiyon ``build_paired_distribution_features`` zaten zorunlu tutuyor.
    """
    if target_column is _ZORUNLU:
        raise TypeError(
            "target_column ACIKCA verilmelidir.\n"
            "Varsayilan None olsaydi 'vermedim' ile 'hedef yok' ayirt edilemez ve "
            "test hedefi lag/rolling penceresine sessizce sizabilirdi.\n"
            f"  hedef bu seri ise : target_column={value_column!r}\n"
            "  hedef degilse     : target_column=None  (bilincli karar)"
        )
    if value_column not in train.columns:
        raise KeyError(f"Kolon '{value_column}' train icinde yok.")
    if time_column not in train.columns:
        raise KeyError(f"Zaman kolonu '{time_column}' train icinde yok.")

    original_train = train
    if test is None:
        combined = train.copy()
        n_train = len(train)
        inserted_target = False
    else:
        if time_column not in test.columns:
            raise KeyError(f"Zaman kolonu '{time_column}' test icinde yok.")
        train_times = pd.to_datetime(train[time_column], errors="raise")
        test_times = pd.to_datetime(test[time_column], errors="raise")
        if len(train_times) and len(test_times) and train_times.max() >= test_times.min():
            raise ValueError("Test zamani train sonrasinda olmali; history ekseni ortusuyor.")

        test_for_history = test.copy()
        inserted_target = target_column == value_column and value_column not in test.columns
        if target_column == value_column and value_column in test.columns:
            raise ValueError(
                f"Test hedef kolonu '{value_column}' iceriyor; hedef turevli history "
                "gelecek hedefi kabul etmez."
            )
        if value_column not in test_for_history.columns:
            if not inserted_target:
                raise KeyError(f"Kolon '{value_column}' test icinde yok.")
            test_for_history[value_column] = np.nan
        n_train = len(train)
        combined = pd.concat([train, test_for_history], ignore_index=True, sort=False)

    transformed = add_lag_features(
        combined,
        value_column,
        shifts=shifts,
        time_column=time_column,
        horizon=horizon,
        group_columns=group_columns,
    )
    if rolling_windows:
        transformed = add_rolling_features(
            transformed,
            value_column,
            rolling_windows,
            time_column=time_column,
            horizon=horizon,
            group_columns=group_columns,
            aggregations=rolling_aggregations,
        )

    train_result = transformed.iloc[:n_train].set_axis(train.index)
    test_result = None
    if test is not None:
        test_result = transformed.iloc[n_train:].set_axis(test.index)
        if inserted_target:
            test_result = test_result.drop(columns=[value_column])
    return FeatureBundle(
        train_result,
        test_result,
        _added_columns(original_train, train_result),
    )


def runtime_recipe_fingerprint(recipe: dict[str, object], **resolved_behavior: object) -> str:
    """Hash the declared recipe together with values resolved at runtime."""
    payload = {"recipe": recipe, "resolved_behavior": resolved_behavior}
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
