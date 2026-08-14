"""Ileri modul testleri: kuantil merdiveni, stacking, zoo, tuning, raporlama."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.ensemble import prune_by_correlation, stack_oof
from gridup.models import starter_params
from gridup.reporting import (
    business_impact,
    cv_fold_table,
    error_by_segment,
    feature_importance_table,
    model_footprint,
    prediction_vs_actual_table,
    worst_segments,
)
from gridup.two_stage import (
    conditional_quantile_from_hurdle,
    fit_quantile_ladder,
    mae_optimal_quantile,
)

# =============================================================================
# MAE-optimal kuantil: matematigin kendisi
# =============================================================================


class TestMaeOptimalQuantile:
    """q* = 1 - 0.5/p  --  karisim dagiliminin medyani."""

    def test_certain_event_gives_plain_median(self):
        """p=1 -> hedef her zaman pozitif -> klasik medyan (0.5)."""
        assert mae_optimal_quantile(np.array([1.0]))[0] == pytest.approx(0.5)

    def test_known_values_from_derivation(self):
        result = mae_optimal_quantile(np.array([0.6, 0.8, 1.0]))
        assert result[0] == pytest.approx(1 - 0.5 / 0.6)   # ~0.1667
        assert result[1] == pytest.approx(0.375)
        assert result[2] == pytest.approx(0.5)

    def test_probability_at_or_below_half_means_predict_zero(self):
        """p <= 0.5 ise karisimin medyani 0'dir -- NaN ile isaretlenir."""
        result = mae_optimal_quantile(np.array([0.5, 0.3, 0.1]))
        assert np.isnan(result).all()

    def test_quantile_is_always_below_half_when_uncertain(self):
        """p < 1 iken dogru kuantil HER ZAMAN 0.5'in altinda.

        Bu, 'thresholded' modun neden suboptimal oldugunu gosterir: o mod
        kosullu MEDYANI (q=0.5) kullanir.
        """
        probabilities = np.linspace(0.51, 0.99, 20)
        quantiles = mae_optimal_quantile(probabilities)
        assert (quantiles < 0.5).all()

    def test_monotonic_in_probability(self):
        result = mae_optimal_quantile(np.array([0.6, 0.7, 0.8, 0.9]))
        assert np.all(np.diff(result) > 0)


class TestConditionalQuantileDecoder:
    @pytest.fixture
    def ladder(self):
        # 4 satir; kuantil arttikca tahmin artiyor (dogru siralama)
        return {
            0.1: np.array([1.0, 2.0, 3.0, 4.0]),
            0.5: np.array([5.0, 6.0, 7.0, 8.0]),
            0.9: np.array([9.0, 10.0, 11.0, 12.0]),
        }

    def test_low_probability_rows_become_zero(self, ladder):
        probability = np.array([0.3, 0.4, 0.5, 0.45])
        result = conditional_quantile_from_hurdle(probability, ladder)
        assert (result == 0).all()

    def test_certain_rows_use_median_rung(self, ladder):
        """p=1 -> q*=0.5 -> merdivenin 0.5 basamagi."""
        probability = np.ones(4)
        result = conditional_quantile_from_hurdle(probability, ladder)
        np.testing.assert_allclose(result, ladder[0.5])

    def test_interpolates_between_rungs(self, ladder):
        # p=0.8 -> q*=0.375 -> 0.1 ile 0.5 arasinda
        probability = np.full(4, 0.8)
        result = conditional_quantile_from_hurdle(probability, ladder)
        assert (result > ladder[0.1]).all()
        assert (result < ladder[0.5]).all()

    def test_never_negative(self, ladder):
        negative_ladder = {level: values - 20 for level, values in ladder.items()}
        result = conditional_quantile_from_hurdle(np.ones(4), negative_ladder)
        assert (result >= 0).all()

    def test_empty_ladder_raises(self):
        with pytest.raises(ValueError, match="bos"):
            conditional_quantile_from_hurdle(np.array([0.8]), {})

    def test_length_mismatch_raises(self, ladder):
        with pytest.raises(ValueError, match="uzunluk"):
            conditional_quantile_from_hurdle(np.array([0.8, 0.9]), ladder)

    def test_beats_naive_median_on_zero_inflated_data(self):
        """Asil iddia: MAE'de q* cozucusu duz medyandan iyi olmali."""
        rng = np.random.default_rng(5)
        n = 4000
        probability = rng.uniform(0.3, 0.95, n)
        occurs = rng.random(n) < probability
        magnitude = rng.gamma(2.0, 3.0, n)
        y = np.where(occurs, magnitude, 0.0)

        # Kosullu dagilimin gercek kuantilleri (ayni gamma)
        from scipy.stats import gamma as gamma_dist

        ladder = {
            level: gamma_dist.ppf(level, a=2.0, scale=3.0) * np.ones(n)
            for level in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
        }

        optimal = conditional_quantile_from_hurdle(probability, ladder)
        naive_median = np.where(probability > 0.5, ladder[0.5], 0.0)

        mae_optimal = float(np.mean(np.abs(optimal - y)))
        mae_naive = float(np.mean(np.abs(naive_median - y)))
        assert mae_optimal < mae_naive


@pytest.mark.slow
class TestQuantileLadder:
    def test_trains_all_levels_and_orders_correctly(self):
        rng = np.random.default_rng(3)
        n = 800
        features = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
        target = 5 + 3 * features["x"].to_numpy() + rng.normal(0, 2, n)
        folds = [(np.arange(0, 600), np.arange(600, 800))]

        params = starter_params("lightgbm", "regression")
        params["n_estimators"] = 80

        ladder = fit_quantile_ladder(
            features, target, folds, quantiles=(0.2, 0.5, 0.8),
            params=params, early_stopping_rounds=20, verbose=False,
        )

        assert set(ladder) == {0.2, 0.5, 0.8}
        valid = np.arange(600, 800)
        low = ladder[0.2].oof_predictions[valid].mean()
        mid = ladder[0.5].oof_predictions[valid].mean()
        high = ladder[0.8].oof_predictions[valid].mean()
        assert low < mid < high, "kuantiller sirali degil"


# =============================================================================
# Stacking ve budama
# =============================================================================


class TestPruneByCorrelation:
    def test_drops_near_duplicate_models(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=500)
        y = base + rng.normal(0, 0.3, 500)
        oof = {
            "a": base,
            "a_kopya": base + rng.normal(0, 1e-4, 500),   # neredeyse ayni
            "b": base * 0.5 + rng.normal(0, 1.0, 500),    # farkli
        }

        kept = prune_by_correlation(oof, y, metric="rmse", max_correlation=0.99)

        assert "a_kopya" not in kept or "a" not in kept
        assert "b" in kept
        assert len(kept) == 2

    def test_respects_member_cap(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=300)
        oof = {f"m{i}": y + rng.normal(0, 1, 300) for i in range(10)}
        kept = prune_by_correlation(oof, y, max_members=3)
        assert len(kept) <= 3

    def test_best_model_is_kept_first(self):
        rng = np.random.default_rng(2)
        y = rng.normal(size=400)
        oof = {
            "kotu": y + rng.normal(0, 3, 400),
            "iyi": y + rng.normal(0, 0.2, 400),
        }
        kept = prune_by_correlation(oof, y, metric="rmse")
        assert kept[0] == "iyi"


class TestStacking:
    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(7)
        n = 1200
        y = rng.normal(10, 3, n)
        oof = {
            "m1": y + rng.normal(0, 1.0, n),
            "m2": y * 1.15 + rng.normal(0, 1.2, n),   # sistematik FAZLA tahmin
            "m3": y + rng.normal(0, 2.0, n),
        }
        folds = [
            (np.arange(0, 600), np.arange(600, 900)),
            (np.arange(0, 900), np.arange(900, 1200)),
        ]
        return oof, y, folds

    def test_produces_oof_and_score(self, setup):
        oof, y, folds = setup
        result = stack_oof(oof, y, folds, metric="rmse", verbose=False)
        assert result["oof"].shape == y.shape
        assert np.isfinite(result["score"])
        assert 0 < result["coverage"] <= 1

    def test_reports_hill_climbing_comparison(self, setup):
        """Karar verebilmek icin ikisi de raporlanmali."""
        oof, y, folds = setup
        result = stack_oof(oof, y, folds, metric="rmse", verbose=False)
        comparison = result["vs_hill_climbing"]
        assert "stacking" in comparison
        assert "hill_climbing" in comparison
        assert isinstance(comparison["stacking_wins"], bool)

    def test_can_assign_negative_weight_to_biased_model(self, setup):
        """Hill climbing'in yapamadigi sey: duzeltici negatif katsayi."""
        oof, y, folds = setup
        result = stack_oof(oof, y, folds, meta="ridge", metric="rmse", verbose=False)
        assert result["coefficients"], "ridge katsayilari bos"

    def test_produces_test_predictions(self, setup):
        oof, y, folds = setup
        test = {name: values[:100] for name, values in oof.items()}
        result = stack_oof(oof, y, folds, test_predictions=test, verbose=False)
        assert result["test"] is not None
        assert len(result["test"]) == 100

    def test_unknown_meta_raises(self, setup):
        oof, y, folds = setup
        with pytest.raises(ValueError, match="Bilinmeyen meta"):
            stack_oof(oof, y, folds, meta="olmayan", verbose=False)


# =============================================================================
# Raporlama
# =============================================================================


class _FakeResult:
    """CVResult benzeri hafif nesne."""

    def __init__(self):
        self.fold_scores = [1.20, 1.25, 1.18, 1.30]
        self.overall_score = 1.23
        self.metric_name = "rmse"
        self.is_stable = True
        self.elapsed_seconds = 42.0
        self.models = []
        self.feature_importance = pd.DataFrame(
            {
                "feature": ["tarih_ay", "tatil_mi", "tuketim_lag1", "ilce_frekans", "x"],
                "importance": [100.0, 50.0, 200.0, 30.0, 10.0],
            }
        )


class TestReporting:
    def test_fold_table_includes_summary_rows(self):
        table = cv_fold_table(_FakeResult())
        assert len(table) == 4 + 3
        assert "ortalama" in table["fold"].astype(str).tolist()
        assert "std" in table["fold"].astype(str).tolist()

    def test_error_by_segment_skips_small_groups(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=100)
        pred = y + rng.normal(0, 0.5, 100)
        segments = pd.Series(["buyuk"] * 95 + ["kucuk"] * 5)

        table = error_by_segment(y, pred, segments, min_count=20)

        assert "kucuk" not in table["segment"].tolist()
        assert "buyuk" in table["segment"].tolist()

    def test_error_by_segment_raises_when_nothing_qualifies(self):
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="esigini"):
            error_by_segment(y, y, pd.Series(["a", "b"]), min_count=50)

    def test_segment_table_reports_bias(self):
        y = np.zeros(60)
        pred = np.full(60, 3.0)   # sistematik fazla tahmin
        table = error_by_segment(y, pred, pd.Series(["a"] * 60), min_count=10)
        assert table["yanlilik"].iloc[0] == pytest.approx(3.0)

    def test_worst_segments_returns_requested_count(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=600)
        pred = y + rng.normal(0, 1, 600)
        segments = pd.Series(rng.choice(list("abcdef"), 600))
        assert len(worst_segments(y, pred, segments, top=3, min_count=20)) <= 3

    def test_calibration_table_uses_quantile_bins(self):
        rng = np.random.default_rng(2)
        y = rng.gamma(2, 2, 500)
        pred = y + rng.normal(0, 0.5, 500)
        table = prediction_vs_actual_table(y, pred, bins=5)
        assert len(table) <= 5
        assert "sapma" in table.columns

    def test_importance_grouping_sums_to_hundred(self):
        table = feature_importance_table(
            _FakeResult(), group_prefixes=("tarih_", "tatil_", "tuketim_")
        )
        assert table["pay_yuzde"].sum() == pytest.approx(100.0, abs=0.2)

    def test_importance_without_grouping_returns_top_n(self):
        table = feature_importance_table(_FakeResult(), top=3)
        assert len(table) == 3

    def test_model_footprint_handles_empty_list(self):
        report = model_footprint([])
        assert report["model_sayisi"] == 0

    def test_business_impact_translates_to_plain_language(self):
        y = np.array([10.0, 12.0, 8.0, 11.0] * 25)
        good = y + 0.5
        result = business_impact(y, good, unit_label="kesinti")
        assert result["iyilesme_yuzde"] > 0
        assert "kesinti" in result["ozet"]

    def test_business_impact_separates_over_and_under(self):
        """Fazla ve eksik tahmin FARKLI is maliyetleridir."""
        y = np.array([5.0, 5.0])
        pred = np.array([8.0, 2.0])   # biri fazla, biri eksik
        result = business_impact(y, pred)
        assert result["fazla_tahmin_toplam"] == pytest.approx(3.0)
        assert result["eksik_tahmin_toplam"] == pytest.approx(3.0)


# =============================================================================
# Zoo
# =============================================================================


@pytest.mark.slow
class TestModelZoo:
    def test_all_models_share_the_same_folds(self):
        """Ayni fold'lar SART -- farkli bolmelerle uretilmis OOF harmanlanamaz."""
        from gridup.zoo import ZooEntry, make_model_zoo

        rng = np.random.default_rng(4)
        n = 700
        features = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
        target = 3 * features["a"] - features["b"] + rng.normal(0, 0.5, n)
        folds = [(np.arange(0, 500), np.arange(500, 700))]

        lgb_params = starter_params("lightgbm", "regression")
        lgb_params["n_estimators"] = 60
        cat_params = starter_params("catboost", "regression")
        cat_params["iterations"] = 60

        result = make_model_zoo(
            features, target, folds,
            entries=[
                ZooEntry("lgb", "lightgbm", lgb_params),
                ZooEntry("cat", "catboost", cat_params),
            ],
            metric="rmse", early_stopping_rounds=20, verbose=False,
        )

        assert set(result.oof_matrix) == {"lgb", "cat"}
        for values in result.oof_matrix.values():
            assert len(values) == n
        assert len(result.leaderboard()) == 2
        assert result.correlation().shape == (2, 2)
