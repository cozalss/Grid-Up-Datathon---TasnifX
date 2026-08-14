"""Feature secimi testleri: null importance ve SHAP geri eleme."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.models import starter_params
from gridup.selection import (
    SelectionResult,
    SelectionStep,
    mean_absolute_shap,
    null_importance_filter,
    shap_backward_selection,
)


@pytest.fixture(scope="module")
def signal_and_noise():
    """3 gercek sinyal + 12 saf gurultu feature'i.

    Gurultu kolonlarindan bir kismi YUKSEK KARDINALITELI -- agac modelleri bunlara
    yapisal olarak yuksek onem verir, gercek sinyal tasimasalar bile. Null
    importance'in yakalamasi gereken tam olarak budur.
    """
    rng = np.random.default_rng(11)
    n = 1200

    real_a = rng.normal(size=n)
    real_b = rng.normal(size=n)
    real_c = rng.integers(0, 4, size=n).astype(float)

    frame = pd.DataFrame({"gercek_a": real_a, "gercek_b": real_b, "gercek_c": real_c})
    for index in range(8):
        frame[f"gurultu_{index}"] = rng.normal(size=n)
    for index in range(4):
        # Yuksek kardinalite: her satir neredeyse benzersiz
        frame[f"gurultu_yuksek_{index}"] = rng.random(n)

    target = 4 * real_a - 3 * real_b + 2 * real_c + rng.normal(0, 0.3, size=n)
    return frame, target.astype("float64")


@pytest.mark.slow
class TestNullImportance:
    def test_keeps_real_signal(self, signal_and_noise):
        frame, target = signal_and_noise
        params = starter_params("lightgbm", "regression")
        params["n_estimators"] = 150

        result = null_importance_filter(
            frame, target, params=params, n_runs=3, verbose=False
        )

        for column in ("gercek_a", "gercek_b", "gercek_c"):
            assert column in result["keep"], f"{column} gercek sinyal ama atildi"

    def test_real_features_rank_above_noise(self, signal_and_noise):
        frame, target = signal_and_noise
        params = starter_params("lightgbm", "regression")
        params["n_estimators"] = 150

        result = null_importance_filter(
            frame, target, params=params, n_runs=3, verbose=False
        )
        ranking = result["scores"].set_index("feature")["oran"]

        assert ranking["gercek_a"] > ranking["gurultu_0"]
        assert ranking["gercek_b"] > ranking["gurultu_yuksek_0"]

    def test_scores_frame_has_expected_columns(self, signal_and_noise):
        frame, target = signal_and_noise
        params = starter_params("lightgbm", "regression")
        params["n_estimators"] = 100

        result = null_importance_filter(
            frame, target, params=params, n_runs=2, verbose=False
        )

        assert set(result["scores"].columns) == {
            "feature", "gercek_onem", "null_esik", "oran",
        }
        assert len(result["keep"]) + len(result["drop"]) == frame.shape[1]


@pytest.mark.slow
class TestMeanAbsoluteShap:
    def test_ranks_real_features_first(self, signal_and_noise):
        import lightgbm as lgb

        frame, target = signal_and_noise
        model = lgb.LGBMRegressor(n_estimators=120, verbose=-1, random_state=0)
        model.fit(frame, target)

        scores = mean_absolute_shap(model, frame, sample_size=500)

        assert list(scores.index[:3]) == sorted(
            ["gercek_a", "gercek_b", "gercek_c"], key=lambda c: -scores[c]
        )
        assert scores["gercek_a"] > scores["gurultu_0"]

    def test_sampling_keeps_all_features_in_index(self, signal_and_noise):
        import lightgbm as lgb

        frame, target = signal_and_noise
        model = lgb.LGBMRegressor(n_estimators=60, verbose=-1, random_state=0)
        model.fit(frame, target)

        scores = mean_absolute_shap(model, frame, sample_size=200)

        assert set(scores.index) == set(frame.columns)


@pytest.mark.slow
class TestShapBackwardSelection:
    def test_drops_noise_and_keeps_score(self, signal_and_noise):
        frame, target = signal_and_noise
        folds = [
            (np.arange(0, 800), np.arange(800, 1000)),
            (np.arange(0, 1000), np.arange(1000, 1200)),
        ]
        params = starter_params("lightgbm", "regression")
        params["n_estimators"] = 200

        result = shap_backward_selection(
            frame, target, folds, metric="rmse", params=params,
            drop_per_step=4, min_features=3, max_steps=4, patience=2,
            shap_sample=400, progress=None,
        )

        assert len(result.best_features) <= frame.shape[1]
        # Gercek sinyaller hayatta kalmali
        for column in ("gercek_a", "gercek_b"):
            assert column in result.best_features

    def test_history_is_recorded_for_the_curve(self, signal_and_noise):
        frame, target = signal_and_noise
        folds = [(np.arange(0, 900), np.arange(900, 1200))]
        params = starter_params("lightgbm", "regression")
        params["n_estimators"] = 120

        result = shap_backward_selection(
            frame, target, folds, metric="rmse", params=params,
            drop_per_step=5, min_features=4, max_steps=3, patience=3,
            shap_sample=300, progress=None,
        )

        curve = result.curve()
        assert len(curve) >= 1
        assert set(curve.columns) == {"feature_sayisi", "skor"}
        assert curve["feature_sayisi"].is_monotonic_increasing


class TestSelectionResultShape:
    """Veri gerektirmeyen yapisal testler."""

    def test_curve_sorts_by_feature_count(self):
        result = SelectionResult(
            best_features=["a"],
            best_score=1.0,
            history=[
                SelectionStep(30, 1.5, ("x",), 1.0),
                SelectionStep(10, 1.2, ("y",), 1.0),
                SelectionStep(20, 1.3, ("z",), 1.0),
            ],
        )
        curve = result.curve()
        assert curve["feature_sayisi"].tolist() == [10, 20, 30]

    def test_summary_handles_empty_history(self):
        result = SelectionResult(best_features=[], best_score=float("nan"))
        assert "Adim yok" in result.summary()
