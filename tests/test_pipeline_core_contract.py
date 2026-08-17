from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.pipeline import (
    DatasetBundle,
    FoldPlan,
    build_frequency_features,
    build_paired_distribution_features,
    build_paired_history_features,
    runtime_recipe_fingerprint,
)


def test_dataset_bundle_rejects_target_in_test():
    train = pd.DataFrame({"id": [1], "target": [2.0]})
    test = pd.DataFrame({"id": [2], "target": [0.0]})

    with pytest.raises(ValueError, match="test.*target"):
        DatasetBundle(train=train, test=test, target_column="target", id_column="id")


def test_frequency_stage_fits_train_once_and_does_not_mutate_inputs():
    train = pd.DataFrame({"kategori": ["a"] * 9 + ["b"], "target": range(10)})
    test = pd.DataFrame({"kategori": ["a"] + ["b"] * 9})
    train_before, test_before = train.copy(deep=True), test.copy(deep=True)

    result = build_frequency_features(train, test, columns=["kategori"])

    assert result.train.loc[0, "kategori_frekans"] == pytest.approx(0.9)
    assert result.test.loc[0, "kategori_frekans"] == pytest.approx(0.9)
    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(test, test_before)


def test_fold_plan_fingerprint_is_deterministic_and_sensitive():
    first = FoldPlan.from_folds(
        [(np.array([0, 1]), np.array([2])), (np.array([0, 1, 2]), np.array([3]))],
        n_rows=4,
    )
    same = FoldPlan.from_folds(
        [(np.array([0, 1]), np.array([2])), (np.array([0, 1, 2]), np.array([3]))],
        n_rows=4,
    )
    changed = FoldPlan.from_folds(
        [(np.array([0]), np.array([1])), (np.array([0, 1, 2]), np.array([3]))],
        n_rows=4,
    )

    assert first.fingerprint == same.fingerprint
    assert first.fingerprint != changed.fingerprint
    assert first.covered.tolist() == [False, False, True, True]


def test_paired_distribution_features_fit_train_only() -> None:
    train = pd.DataFrame({"grup": ["a", "a", "b"], "x": [0.0, 2.0, 10.0], "kat": ["a", "a", "b"]})
    test = pd.DataFrame({"grup": ["a"], "x": [100.0], "kat": ["a"]})

    result = build_paired_distribution_features(
        train,
        test,
        group_columns=["grup"],
        value_columns=["x"],
        frequency_columns=["kat"],
        aggregations=("mean",),
        target_column="hedef",
    )

    assert result.test is not None
    assert result.test.loc[0, "x_bazinda_grup_mean"] == pytest.approx(1.0)
    assert result.test.loc[0, "kat_frekans"] == pytest.approx(2 / 3)


def test_paired_history_features_carry_train_tail_into_test() -> None:
    train = pd.DataFrame(
        {
            "tarih": pd.date_range("2025-01-01", periods=6),
            "grup": ["g"] * 6,
            "x": np.arange(6, dtype=float),
        }
    )
    test = pd.DataFrame(
        {
            "tarih": pd.date_range("2025-01-07", periods=2),
            "grup": ["g"] * 2,
            "x": [100.0, 101.0],
        }
    )

    result = build_paired_history_features(
        train,
        test,
        value_column="x",
        time_column="tarih",
        horizon=2,
        group_columns=["grup"],
        shifts=[2],
        rolling_windows=[2],
        rolling_aggregations=("mean",),
        target_column=None,
    )

    assert result.test is not None
    assert result.test["x_shift2"].tolist() == pytest.approx([4.0, 5.0])
    assert result.test["x_ufuk2_kayan2_mean"].tolist() == pytest.approx([3.5, 4.5])


def test_target_column_atlanirsa_acik_hata_verir() -> None:
    """``target_column`` sessiz varsayilana sahip OLMAMALIDIR.

    Varsayilan ``None`` olsaydi hedef korumasi opt-in olurdu: cagiran parametreyi
    atladiginda ``target_column == value_column`` asla dogru olmaz ve test
    hedefleri lag/rolling penceresine sessizce sizardi. Bu, ``features.aggregate``
    icinde OLCULMUS (hedefle 0.96 korelasyonlu kolonlar) ve nobetciyle kapatilmis
    hatanin aynisidir.
    """
    train = pd.DataFrame(
        {"tarih": pd.date_range("2025-01-01", periods=4), "x": [1.0, 2.0, 3.0, 4.0]}
    )

    with pytest.raises(TypeError, match="target_column"):
        build_paired_history_features(
            train,
            None,
            value_column="x",
            time_column="tarih",
            horizon=2,
            shifts=[2],
        )


def test_target_history_never_accepts_test_target_values() -> None:
    train = pd.DataFrame(
        {"tarih": pd.date_range("2025-01-01", periods=3), "target": [1.0, 2.0, 3.0]}
    )
    test = pd.DataFrame({"tarih": pd.date_range("2025-01-04", periods=2), "target": [999.0, 999.0]})

    with pytest.raises(ValueError, match="hedef|target"):
        build_paired_history_features(
            train,
            test,
            value_column="target",
            target_column="target",
            time_column="tarih",
            horizon=2,
            shifts=[2],
        )


def test_runtime_recipe_fingerprint_covers_resolved_behavior() -> None:
    base = {"model": "lightgbm", "metric": "rmsle"}
    first = runtime_recipe_fingerprint(base, target_transform="log1p", n_estimators=400)
    same = runtime_recipe_fingerprint(base, target_transform="log1p", n_estimators=400)
    changed = runtime_recipe_fingerprint(base, target_transform=None, n_estimators=400)

    assert first == same
    assert first != changed
