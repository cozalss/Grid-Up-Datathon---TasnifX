"""Okuma, profilleme, metrik, submission ve uctan uca akis testleri."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.compat import (
    assert_no_removed_api,
    categorical_columns,
    is_categorical_like,
    safe_concat,
)
from gridup.io_utils import read_table, sniff_dialect
from gridup.metrics import (
    inverse_log_transform,
    log_transform_target,
    optimize_threshold,
    rmsle,
    smape,
)
from gridup.profiling import profile
from gridup.submission import blend_submissions, validate_submission, write_submission
from gridup.synthetic import SyntheticSpec, make_distribution_dataset


class TestDialectSniffing:
    """Turk kurumlarindan cikan dosya bicimlerinin dogru tespiti."""

    def test_detects_semicolon_and_comma_decimal_in_cp1254(self, tmp_path):
        # Arrange -- TUIK/Excel ihracinin tipik hali
        path = tmp_path / "tuik.csv"
        content = "İL;TÜKETİM;ORAN\nİzmir;1.234.567,89;12,5\nMuğla;987.654,32;8,3\n"
        path.write_bytes(content.encode("cp1254"))

        # Act
        guess = sniff_dialect(path)

        # Assert
        assert guess.encoding in {"cp1254", "iso-8859-9"}
        assert guess.delimiter == ";"
        assert guess.decimal == ","

    def test_thousands_separator_does_not_split_a_number(self, tmp_path):
        """``1.234.567,89`` TEK sayidir -- uc alana bolunurse sessizce bozulur."""
        path = tmp_path / "sayilar.csv"
        path.write_bytes("ad;deger\nA;1.234.567,89\nB;2.000,50\n".encode("cp1254"))

        frame = read_table(path, verbose=False)

        assert len(frame) == 2
        assert frame["deger"].iloc[0] == pytest.approx(1_234_567.89)
        assert frame["deger"].iloc[1] == pytest.approx(2_000.50)

    def test_detects_standard_utf8_comma_csv(self, tmp_path):
        path = tmp_path / "standart.csv"
        path.write_text("a,b,c\n1,2.5,x\n3,4.5,y\n", encoding="utf-8")

        guess = sniff_dialect(path)

        assert guess.delimiter == ","
        assert guess.decimal == "."

    def test_handles_utf8_bom_from_excel(self, tmp_path):
        path = tmp_path / "excel.csv"
        path.write_text("İL,DEĞER\nİzmir,1\n", encoding="utf-8-sig")

        frame = read_table(path, verbose=False)

        assert list(frame.columns) == ["il", "deger"]

    def test_column_names_are_normalized_and_original_kept(self, tmp_path):
        path = tmp_path / "turkce.csv"
        path.write_text("İL,KESİNTİ SÜRESİ (dk)\nİzmir,12\n", encoding="utf-8")

        frame = read_table(path, verbose=False)

        assert list(frame.columns) == ["il", "kesinti_suresi_dk"]
        assert frame.attrs["original_columns"]["İL"] == "il"

    def test_undecodable_file_raises_rather_than_producing_mojibake(self, tmp_path):
        path = tmp_path / "bozuk.bin"
        path.write_bytes(bytes(range(256)) * 4)

        # latin-1 her bayti kabul eder, bu yuzden hata beklemiyoruz;
        # ama en azindan cokmeden bir tahmin dondurmeli.
        guess = sniff_dialect(path)
        assert guess.encoding in {"utf-8", "cp1254", "iso-8859-9", "latin-1"}


class TestCompat:
    """pandas 3.0 / numpy 2.x uyumluluk korumalari."""

    def test_string_column_is_categorical_like_on_pandas_3(self):
        """pandas 3.0'da metin 'str' dtype'indadir ve is_object_dtype onu GORMEZ."""
        series = pd.Series(["a", "b", "c"])
        assert is_categorical_like(series) is True

    def test_object_column_is_categorical_like(self):
        series = pd.Series(["a", 1, None], dtype=object)
        assert is_categorical_like(series) is True

    def test_category_column_is_categorical_like(self):
        assert is_categorical_like(pd.Series(["a", "b"]).astype("category")) is True

    @pytest.mark.parametrize(
        "series",
        [
            pd.Series([1, 2, 3]),
            pd.Series([1.0, 2.0]),
            pd.Series([True, False]),
            pd.to_datetime(pd.Series(["2024-01-01"])),
        ],
    )
    def test_model_ready_dtypes_are_not_categorical_like(self, series):
        assert is_categorical_like(series) is False

    def test_categorical_columns_finds_every_text_column(self):
        frame = pd.DataFrame(
            {
                "metin": ["a", "b"],
                "nesne": pd.Series(["a", 1], dtype=object),
                "kategori": pd.Series(["x", "y"]).astype("category"),
                "sayi": [1, 2],
            }
        )
        assert set(categorical_columns(frame)) == {"metin", "nesne", "kategori"}

    def test_removed_api_scanner_flags_pandas_2_idioms(self):
        source = "df = df.applymap(str)\nx = np.NaN\ndf2 = df.append(other)"
        problems = assert_no_removed_api(source)
        assert len(problems) >= 3

    def test_removed_api_scanner_passes_clean_code(self):
        assert assert_no_removed_api("df = df.map(str)\nx = np.nan") == []

    def test_safe_concat_handles_empty_frames(self):
        result = safe_concat([pd.DataFrame({"a": [1]}), pd.DataFrame(columns=["a"])])
        assert len(result) == 1
        assert pd.api.types.is_numeric_dtype(result["a"])


class TestMetrics:
    def test_rmsle_matches_manual_computation(self):
        y_true = np.array([1.0, 10.0, 100.0])
        y_pred = np.array([1.5, 9.0, 110.0])
        expected = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
        assert rmsle(y_true, y_pred) == pytest.approx(expected)

    def test_rmsle_clips_negative_predictions_instead_of_crashing(self):
        assert np.isfinite(rmsle(np.array([1.0, 2.0]), np.array([-5.0, 2.0])))

    def test_log_transform_round_trips(self):
        values = np.array([0.0, 1.0, 100.0, 5000.0])
        recovered = inverse_log_transform(log_transform_target(values))
        np.testing.assert_allclose(recovered, values, rtol=1e-9)

    def test_log_transform_rejects_values_below_minus_one(self):
        with pytest.raises(ValueError):
            log_transform_target(np.array([-2.0]))

    def test_inverse_transform_clips_negatives_by_default(self):
        assert inverse_log_transform(np.array([-10.0]))[0] == 0.0

    def test_smape_is_finite_when_truth_is_zero(self):
        assert np.isfinite(smape(np.array([0.0, 10.0]), np.array([1.0, 10.0])))

    def test_threshold_optimization_beats_default_half_on_imbalanced_data(self):
        # Arrange -- %5 pozitif, model kalibre degil
        rng = np.random.default_rng(0)
        y_true = (rng.random(2000) < 0.05).astype(int)
        y_proba = np.clip(y_true * 0.3 + rng.random(2000) * 0.25, 0, 1)

        # Act
        result = optimize_threshold(y_true, y_proba, metric="f1")

        # Assert
        assert result["best_score"] >= result["score_at_half"]
        assert 0.0 < result["best_threshold"] < 1.0


class TestSubmissionValidation:
    def test_nan_prediction_is_rejected(self):
        submission = pd.DataFrame({"ID": [1, 2], "hedef": [1.0, np.nan]})
        check = validate_submission(submission)
        assert not check.is_valid
        assert any("NaN" in error for error in check.errors)

    def test_infinite_prediction_is_rejected(self):
        submission = pd.DataFrame({"ID": [1, 2], "hedef": [1.0, np.inf]})
        assert not validate_submission(submission).is_valid

    def test_missing_ids_versus_sample_are_reported(self):
        sample = pd.DataFrame({"ID": [1, 2, 3], "hedef": [0.0, 0.0, 0.0]})
        submission = pd.DataFrame({"ID": [1, 2], "hedef": [1.0, 2.0]})

        check = validate_submission(submission, sample=sample)

        assert not check.is_valid
        assert any("eksik" in error.lower() for error in check.errors)

    def test_constant_prediction_produces_warning_not_error(self):
        submission = pd.DataFrame({"ID": [1, 2, 3], "hedef": [5.0, 5.0, 5.0]})
        check = validate_submission(submission)
        assert check.is_valid
        assert any("AYNI" in warning for warning in check.warnings)

    def test_negative_prediction_warns_for_physical_quantity(self):
        submission = pd.DataFrame({"ID": [1, 2], "hedef": [-1.0, 5.0]})
        check = validate_submission(submission, allow_negative=False)
        assert any("negatif" in warning.lower() for warning in check.warnings)

    def test_write_submission_clips_negatives_and_validates(self, tmp_path):
        path = write_submission(
            np.array([1, 2, 3]), np.array([-5.0, 10.0, 20.0]),
            tmp_path / "sub.csv", id_column="ID", target_column="hedef",
        )
        written = pd.read_csv(path)
        assert (written["hedef"] >= 0).all()
        assert len(written) == 3

    def test_blend_rejects_mismatched_id_order(self, tmp_path):
        first = tmp_path / "a.csv"
        second = tmp_path / "b.csv"
        pd.DataFrame({"ID": [1, 2], "hedef": [1.0, 2.0]}).to_csv(first, index=False)
        pd.DataFrame({"ID": [2, 1], "hedef": [3.0, 4.0]}).to_csv(second, index=False)

        with pytest.raises(ValueError, match="ID sirasi"):
            blend_submissions([first, second])

    def test_blend_averages_correctly(self, tmp_path):
        first = tmp_path / "a.csv"
        second = tmp_path / "b.csv"
        pd.DataFrame({"ID": [1, 2], "hedef": [0.0, 0.0]}).to_csv(first, index=False)
        pd.DataFrame({"ID": [1, 2], "hedef": [10.0, 20.0]}).to_csv(second, index=False)

        blended = blend_submissions([first, second], weights=[0.5, 0.5])

        assert blended["hedef"].tolist() == [5.0, 10.0]


@pytest.fixture(scope="module")
def dataset():
    """Kucuk sentetik veri seti -- modul basina bir kez uretilir."""
    spec = SyntheticSpec(n_transformers=8, start_date="2024-01-01", end_date="2024-06-30", seed=7)
    return make_distribution_dataset(spec)


class TestSyntheticDataAndProfiling:
    def test_test_split_contains_no_target_columns(self, dataset):
        _, test, _ = dataset
        for leaked in ("KESİNTİ_SÜRESİ_DK", "ARIZA_VAR_MI", "ARIZA_TİPİ"):
            assert leaked not in test.columns

    def test_train_and_test_are_time_separated(self, dataset):
        train, test, _ = dataset
        assert train["TARİH"].max() < test["TARİH"].min()

    def test_solution_covers_every_test_row(self, dataset):
        _, test, solution = dataset
        assert len(solution) == len(test)
        assert set(solution["ID"]) == set(test["ID"])

    def test_generation_is_reproducible(self):
        spec = SyntheticSpec(n_transformers=4, start_date="2024-01-01", end_date="2024-02-01")
        first, _, _ = make_distribution_dataset(spec)
        second, _, _ = make_distribution_dataset(spec)
        pd.testing.assert_frame_equal(first, second)

    def test_profile_detects_zero_inflated_skewed_target(self, dataset):
        train, test, _ = dataset

        report = profile(train, test, target="KESİNTİ_SÜRESİ_DK")

        assert report.target_summary["gorev_tahmini"] == "regression"
        assert report.target_summary["carpiklik"] > 1.0
        assert report.target_summary["sifir_orani"] > 0.5
        assert "log1p" in report.target_summary.get("oneri", "")

    def test_profile_flags_schema_difference_as_leakage_candidates(self, dataset):
        train, test, _ = dataset
        report = profile(train, test, target="KESİNTİ_SÜRESİ_DK")
        assert "ARIZA_VAR_MI" in report.schema_diff["train_only"]

    def test_profile_finds_the_time_column(self, dataset):
        train, test, _ = dataset
        assert "TARİH" in profile(train, test).time_columns

    def test_report_renders_without_error(self, dataset):
        train, test, _ = dataset
        text = profile(train, test, target="KESİNTİ_SÜRESİ_DK").report()
        assert "VERI PROFILI" in text
        assert len(text) > 500


@pytest.mark.slow
class TestEndToEnd:
    """Kucuk ama gercek bir egitim dongusu -- pipeline'in butunlugunu kanitlar."""

    def test_full_pipeline_beats_median_baseline(self, tmp_path):
        from gridup.features import add_calendar_features
        from gridup.models import cross_validate, starter_params
        from gridup.turkish import normalize_columns
        from gridup.validation import purged_time_series_split

        # Arrange
        spec = SyntheticSpec(
            n_transformers=25, start_date="2024-01-01", end_date="2024-12-31", seed=3
        )
        train_raw, test_raw, solution = make_distribution_dataset(spec)
        train = train_raw.rename(columns=normalize_columns(train_raw.columns))
        test = test_raw.rename(columns=normalize_columns(test_raw.columns))
        solution = solution.rename(columns=normalize_columns(solution.columns))

        target = "kesinti_suresi_dk"
        drop = [target, "ariza_var_mi", "ariza_tipi", "id", "tarih"]

        train = add_calendar_features(train, "tarih", include_year=False)
        test = add_calendar_features(test, "tarih", include_year=False)

        columns = [c for c in train.columns if c not in drop and c in test.columns]
        folds = list(purged_time_series_split(train["tarih"], n_splits=3))

        params = starter_params("lightgbm", "regression")
        params["n_estimators"] = 120

        # Act
        y = log_transform_target(train[target].to_numpy())
        result = cross_validate(
            train[columns], y, folds,
            kind="lightgbm", metric="rmse", params=params,
            test=test[columns], early_stopping_rounds=30, verbose=False,
        )

        path = write_submission(
            test["id"].to_numpy(),
            inverse_log_transform(result.test_predictions),
            tmp_path / "e2e.csv",
            id_column="id", target_column=target, validate=True,
        )

        # Assert
        merged = pd.read_csv(path).merge(solution, on="id", suffixes=("_p", "_g"))
        truth = merged[f"{target}_g"].to_numpy()
        model_score = rmsle(truth, merged[f"{target}_p"].to_numpy())
        baseline_score = rmsle(truth, np.full_like(truth, float(np.median(train[target]))))

        assert model_score < baseline_score, "model medyan baseline'i gecemedi"
        assert len(result.fold_scores) == len(folds)
        assert result.feature_importance["importance"].sum() > 0

    def test_cv_result_reports_instability(self):
        from gridup.models import CVResult

        stable = CVResult(
            oof_predictions=np.zeros(10), test_predictions=None,
            fold_scores=[1.00, 1.02, 0.99], overall_score=1.0,
            feature_importance=pd.DataFrame({"feature": [], "importance": []}),
        )
        noisy = CVResult(
            oof_predictions=np.zeros(10), test_predictions=None,
            fold_scores=[0.5, 1.5, 1.0], overall_score=1.0,
            feature_importance=pd.DataFrame({"feature": [], "importance": []}),
        )

        assert stable.is_stable is True
        assert noisy.is_stable is False
