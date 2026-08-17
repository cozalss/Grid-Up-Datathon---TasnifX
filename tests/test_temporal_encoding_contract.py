"""Temporal target encoding must expose and enforce OOF coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.features.categorical import oof_target_encode


def _forward_folds() -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (np.arange(0, 4), np.arange(4, 7)),
        (np.arange(0, 7), np.arange(7, 10)),
    ]


def test_partial_temporal_coverage_fails_closed_by_default() -> None:
    train = pd.DataFrame({"kategori": ["a", "b"] * 5})
    target = pd.Series(np.arange(10, dtype=float))

    with pytest.raises(ValueError, match="kapsam"):
        oof_target_encode(train, target, ["kategori"], _forward_folds())


def test_nan_policy_exposes_coverage_without_breaking_two_value_unpacking() -> None:
    train = pd.DataFrame({"kategori": ["a", "b"] * 5})
    target = pd.Series(np.arange(10, dtype=float))

    result = oof_target_encode(
        train,
        target,
        ["kategori"],
        _forward_folds(),
        uncovered_policy="nan",
    )
    encoded_train, encoded_test = result

    assert encoded_test is None
    assert len(result) == 2
    np.testing.assert_array_equal(
        result.covered,
        np.array([False, False, False, False, True, True, True, True, True, True]),
    )
    assert encoded_train.loc[:3, "kategori_hedef_kod"].isna().all()
    assert encoded_train.loc[4:, "kategori_hedef_kod"].notna().all()


def test_uncovered_rows_do_not_change_when_future_targets_change() -> None:
    train = pd.DataFrame({"kategori": ["a", "b"] * 5})
    original = pd.Series(np.arange(10, dtype=float))
    changed = original.copy()
    changed.iloc[7:] = 1_000_000.0

    first = oof_target_encode(
        train, original, ["kategori"], _forward_folds(), uncovered_policy="nan"
    )
    second = oof_target_encode(
        train, changed, ["kategori"], _forward_folds(), uncovered_policy="nan"
    )

    assert first[0].loc[:3, "kategori_hedef_kod"].isna().all()
    assert second[0].loc[:3, "kategori_hedef_kod"].isna().all()
    np.testing.assert_array_equal(first.covered, second.covered)
