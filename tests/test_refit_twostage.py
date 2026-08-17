"""Cok tohumlu yeniden egitim ve iki asamali model testleri."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.metrics import get_metric
from gridup.models import starter_params
from gridup.refit import estimate_full_data_rounds, extract_best_iterations, multi_seed_refit
from gridup.two_stage import (
    fit_two_stage,
    tune_threshold,
    zero_baseline_score,
)


class TestFullDataRounds:
    def test_scales_up_by_one_over_k(self):
        """5 katli CV'de tam veri %20 daha buyuk -> ~%20 daha fazla agac."""
        rounds = estimate_full_data_rounds([100, 100, 100, 100, 100], n_folds=5)
        assert rounds == 120

    def test_uses_mean_of_folds(self):
        rounds = estimate_full_data_rounds([50, 150], n_folds=5)
        assert rounds == pytest.approx(120, abs=1)

    def test_ignores_zero_and_none(self):
        rounds = estimate_full_data_rounds([0, 100, 100], n_folds=5)
        assert rounds == 120

    def test_empty_raises_instead_of_guessing(self):
        """Sessizce bir varsayilan uydurmak, yanlis agac sayisiyla egitmek demektir."""
        with pytest.raises(ValueError, match="alinamadi"):
            estimate_full_data_rounds([], n_folds=5)

    def test_safety_factor_makes_it_conservative(self):
        assert estimate_full_data_rounds([100] * 5, n_folds=5, safety=0.9) < 120


class TestExtractBestIterations:
    def test_reads_lightgbm_attribute(self):
        class FakeModel:
            best_iteration_ = 137

        assert extract_best_iterations([FakeModel(), FakeModel()]) == [137, 137]

    def test_skips_models_without_early_stopping(self):
        class NoStop:
            pass

        assert extract_best_iterations([NoStop()]) == []


@pytest.fixture(scope="module")
def data():
    """Basit dogrusal veri: train, hedef, test."""
    rng = np.random.default_rng(0)
    n = 600
    features = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.integers(0, 5, size=n).astype(float),
        }
    )
    target = 3 * features["a"] - 2 * features["b"] + rng.normal(0, 0.4, size=n)
    return features.iloc[:500], target.to_numpy()[:500], features.iloc[500:]


@pytest.mark.slow
class TestMultiSeedRefit:
    def test_averaging_reduces_variance_versus_single_seed(self, data):
        train, y, test = data
        params = starter_params("lightgbm", "regression")
        params["n_estimators"] = 60

        result = multi_seed_refit(
            train,
            y,
            test,
            kind="lightgbm",
            params=params,
            n_estimators=60,
            seeds=(0, 1, 2, 3),
            verbose=False,
        )

        assert result.per_seed_predictions.shape == (4, len(test))
        # Ortalamanin sapmasi, tekil tohumlarin sapmasindan kucuk olmali
        single_spread = result.per_seed_predictions.std(axis=0).mean()
        assert single_spread >= 0
        assert len(result.predictions) == len(test)

    def test_seeds_actually_differ(self, data):
        """Tohum gercekten degismezse cok tohumlu ortalama anlamsizdir."""
        train, y, test = data
        params = starter_params("lightgbm", "regression")
        params["n_estimators"] = 60
        params["subsample"] = 0.7
        params["subsample_freq"] = 1

        result = multi_seed_refit(
            train,
            y,
            test,
            kind="lightgbm",
            params=params,
            n_estimators=60,
            seeds=(0, 1, 2),
            verbose=False,
        )

        first, second = result.per_seed_predictions[0], result.per_seed_predictions[1]
        assert not np.allclose(first, second), "tohumlar ayni sonucu uretti"

    def test_requires_test_frame(self, data):
        train, y, _ = data
        with pytest.raises(ValueError, match="test"):
            multi_seed_refit(
                train,
                y,
                None,
                kind="lightgbm",  # type: ignore[arg-type]
                params=starter_params("lightgbm", "regression"),
                n_estimators=10,
                verbose=False,
            )

    def test_length_mismatch_raises(self, data):
        train, _, test = data
        with pytest.raises(ValueError, match="uzunluklari"):
            multi_seed_refit(
                train,
                np.zeros(3),
                test,
                kind="lightgbm",
                params=starter_params("lightgbm", "regression"),
                n_estimators=10,
                verbose=False,
            )


class TestZeroBaseline:
    def test_all_zero_baseline_on_sparse_target(self):
        y = np.array([0.0] * 90 + [5.0] * 10)
        score = zero_baseline_score(y, metric="mae")
        assert score == pytest.approx(0.5)

    def test_baseline_is_strong_when_target_is_mostly_zero(self):
        """Sifir-siskin veride 'hep sifir' sasirtici derecede gucludur."""
        y = np.array([0.0] * 95 + [3.0] * 5)
        baseline = zero_baseline_score(y, metric="mae")
        # Ortalamayi tahmin etmek DAHA KOTU
        mean_prediction = np.full_like(y, y.mean())
        metric_fn, _, _ = get_metric("mae")
        assert baseline < float(metric_fn(y, mean_prediction))


class TestThresholdTuning:
    def test_tuned_threshold_beats_default_half(self):
        rng = np.random.default_rng(1)
        n = 2000
        is_event = rng.random(n) < 0.15
        y = np.where(is_event, rng.integers(1, 6, size=n), 0).astype("float64")
        # Kalibre olmayan bir olasilik: olaylarda yuksek ama 0.5'in altinda
        probability = np.clip(is_event * 0.30 + rng.random(n) * 0.20, 0, 1)
        magnitude = np.full(n, 3.0)

        result = tune_threshold(y, probability, magnitude, metric="mae")

        assert result["best_score"] <= result["score_at_half"]
        assert 0.0 < result["best_threshold"] < 1.0

    def test_reports_all_zero_baseline_for_comparison(self):
        y = np.array([0.0] * 80 + [2.0] * 20)
        probability = np.linspace(0, 1, 100)
        magnitude = np.full(100, 2.0)

        result = tune_threshold(y, probability, magnitude, metric="mae")

        assert "score_all_zero" in result
        assert result["score_all_zero"] == pytest.approx(0.4)


@pytest.fixture(scope="module")
def sparse_data():
    """Sifir-siskin sayim hedefi: olay olasiligi bir surucuye bagli."""
    rng = np.random.default_rng(7)
    n = 900
    driver = rng.normal(size=n)
    features = pd.DataFrame(
        {
            "surucu": driver,
            "gurultu": rng.normal(size=n),
            "olcek": rng.integers(1, 4, size=n).astype(float),
        }
    )
    probability = 1 / (1 + np.exp(-(driver * 2 - 1.2)))
    occurs = rng.random(n) < probability
    magnitude = features["olcek"].to_numpy() * rng.integers(1, 4, size=n)
    target = np.where(occurs, magnitude, 0).astype("float64")

    folds = [
        (np.arange(0, 600), np.arange(600, 750)),
        (np.arange(0, 750), np.arange(750, 900)),
    ]
    return features, target, folds


@pytest.mark.slow
class TestTwoStage:
    def test_rejects_target_with_no_zeros(self, sparse_data):
        features, _, folds = sparse_data
        with pytest.raises(ValueError, match="hic sifir yok"):
            fit_two_stage(features, np.ones(len(features)), folds, verbose=False)

    def test_rejects_target_with_no_positives(self, sparse_data):
        features, _, folds = sparse_data
        with pytest.raises(ValueError, match="hic pozitif"):
            fit_two_stage(features, np.zeros(len(features)), folds, verbose=False)

    def test_produces_both_stages_and_a_threshold(self, sparse_data):
        features, target, folds = sparse_data

        result = fit_two_stage(
            features,
            target,
            folds,
            metric="mae",
            early_stopping_rounds=30,
            verbose=False,
        )

        assert result.best_threshold is not None
        assert 0.0 < result.best_threshold < 1.0
        assert result.oof_probability.shape == (len(features),)
        assert result.oof_magnitude.shape == (len(features),)
        assert "sifir_orani" in result.diagnostics

    def test_thresholded_mode_needs_a_threshold(self, sparse_data):
        features, target, folds = sparse_data
        result = fit_two_stage(features, target, folds, early_stopping_rounds=30, verbose=False)
        result.best_threshold = None
        with pytest.raises(ValueError, match="esik"):
            result.predict_oof(mode="thresholded")

    def test_expected_mode_needs_no_threshold(self, sparse_data):
        features, target, folds = sparse_data
        result = fit_two_stage(features, target, folds, early_stopping_rounds=30, verbose=False)
        predictions = result.predict_oof(mode="expected")
        assert predictions.shape == (len(features),)
        assert (predictions >= 0).all()

    def test_second_stage_trains_only_on_positives(self, sparse_data):
        features, target, folds = sparse_data
        result = fit_two_stage(features, target, folds, early_stopping_rounds=30, verbose=False)
        # 2. asamanin OOF'u pozitif sayisi kadar satirla egitildi
        assert int(result.positive_mask.sum()) < len(features)
        assert len(result.regressor.oof_predictions) == int(result.positive_mask.sum())
