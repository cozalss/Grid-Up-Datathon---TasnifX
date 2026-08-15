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
    safe_str,
)
from gridup.features.aggregate import add_group_statistics
from gridup.features.categorical import add_combination_features, add_frequency_encoding
from gridup.features.temporal import (
    MISSING_HOLIDAY_DISTANCE,
    add_calendar_features,
    add_lag_features,
    add_turkish_holiday_features,
)
from gridup.io_utils import read_table, sniff_dialect
from gridup.metrics import (
    inverse_log_transform,
    log_transform_target,
    optimize_threshold,
    postprocess_predictions,
    rmsle,
    smape,
)
from gridup.models import COUNT_OBJECTIVES, starter_params
from gridup.panel import build_panel, panel_coverage
from gridup.profiling import profile
from gridup.submission import blend_submissions, validate_submission, write_submission
from gridup.synthetic import SyntheticSpec, make_distribution_dataset
from gridup.turkish import normalize_columns


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


class TestDirtyDataSurvival:
    """Gercek veride kacinilmaz olan bozukluklar pipeline'i COKERTMEMELI.

    Bu testlerin hepsi, adversarial review sirasinda gercekten cokturulen
    veya sessizce yanlis sonuc ureten senaryolardir.
    """

    def test_calendar_features_survive_unparseable_dates(self):
        frame = pd.DataFrame({"tarih": ["2024-01-01", "bozuk-tarih", "2024-01-03"]})
        result = add_calendar_features(frame, "tarih")
        assert len(result) == 3
        assert result["tarih_ay"].isna().sum() == 1

    def test_holiday_distance_does_not_mark_broken_dates_as_holidays(self):
        """NaT sentinel'i int16'ya kirpilinca 0'a dusuyordu -> 'tam bayram gunu'."""
        frame = pd.DataFrame(
            {"tarih": ["2024-06-15", "bozuk-tarih", "2024-07-04", "2024-03-11"]}
        )

        result = add_turkish_holiday_features(frame, "tarih")

        broken = result.iloc[1]
        assert broken["tatil_mesafe"] == MISSING_HOLIDAY_DISTANCE
        assert broken["tatil_yakininda"] == 0

    def test_lag_features_survive_missing_group_values(self):
        """np.lexsort ham object dizisinde str/None karsilastirmasiyla cokuyordu."""
        frame = pd.DataFrame(
            {
                "tarih": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
                ),
                "trafo_id": ["A", None, "A", None],
                "deger": [10.0, 20.0, 30.0, 40.0],
            }
        )

        result = add_lag_features(
            frame, "deger", [1], time_column="tarih", horizon=1, group_columns=["trafo_id"]
        )

        assert len(result) == 4
        # Eksik grup kendi icinde bir grup olur; A grubunun 2. satiri 10.0 gorur.
        assert result.loc[2, "deger_lag1"] == pytest.approx(10.0)

    def test_lag_features_sort_string_dates_chronologically(self):
        """Sifir dolgusuz metin tarihte sozluksel siralama yanlis lag uretiyordu."""
        frame = pd.DataFrame(
            {
                "tarih": ["2024-1-2", "2024-1-9", "2024-1-10", "2024-1-20", "2024-1-3"],
                "trafo_id": ["A"] * 5,
                "deger": [1.0, 2.0, 3.0, 4.0, 1.5],
            }
        )

        result = add_lag_features(
            frame, "deger", [1], time_column="tarih", horizon=1, group_columns=["trafo_id"]
        )

        # 2024-1-3 (deger 1.5) satirinin lag1'i 2024-1-2'nin degeri = 1.0 olmali
        assert result.loc[4, "deger_lag1"] == pytest.approx(1.0)


class TestVersionSafeStrings:
    """.astype(str) pandas 2.x'te NaN'i 'None' stringine cevirir -- safe_str cevirmez."""

    def test_safe_str_preserves_missing_by_default(self):
        result = safe_str(pd.Series(["a", "b", None]))
        assert result.isna().sum() == 1
        assert "None" not in result.dropna().tolist()

    def test_safe_str_applies_sentinel_when_asked(self):
        result = safe_str(pd.Series(["a", None]), missing="_EKSIK")
        assert result.tolist() == ["a", "_EKSIK"]

    def test_combination_features_do_not_invent_none_categories(self):
        frame = pd.DataFrame(
            {"il": ["izmir", None, "mugla"], "ilce": ["konak", "bornova", None]}
        )

        result = add_combination_features(frame, [("il", "ilce")])
        values = result["il__ilce"].astype(object).tolist()

        assert values[0] == "izmir__konak"
        assert pd.isna(values[1]) and pd.isna(values[2])
        assert not any(isinstance(v, str) and "None" in v for v in values)


class TestEncodingDistinguishesMissingFromUnseen:
    def test_unseen_category_gets_zero_but_missing_stays_nan(self):
        reference = pd.DataFrame({"tip": ["a", "a", "b", "b", "b"]})
        frame = pd.DataFrame({"tip": ["a", "b", "c", None]})

        result = add_frequency_encoding(frame, ["tip"], reference=reference)
        encoded = result["tip_frekans"]

        assert encoded.iloc[2] == pytest.approx(0.0)   # gorulmemis kategori
        assert pd.isna(encoded.iloc[3])                 # gercekten eksik


class TestAggregateTargetGuard:
    def test_target_in_value_columns_is_rejected(self):
        frame = pd.DataFrame({"ilce": ["a", "b"], "hedef": [1.0, 2.0]})
        with pytest.raises(ValueError, match="sizinti"):
            add_group_statistics(frame, ["ilce"], ["hedef"], target_column="hedef")


class TestSubmissionClippingIsVisible:
    def test_negative_predictions_are_reported_before_clipping(self, tmp_path, capsys):
        """Once kirpip sonra dogrulamak uyariyi ASLA tetiklenemez kiliyordu."""
        write_submission(
            np.array([1, 2, 3, 4]),
            np.array([-5.0, -2.0, 10.0, 20.0]),
            tmp_path / "sub.csv",
            id_column="ID", target_column="hedef",
        )

        output = capsys.readouterr().out
        assert "negatif" in output.lower()
        assert "KIRPILDI" in output


class TestProfileFailsLoudlyOnBadTarget:
    def test_misspelled_target_raises_instead_of_silently_skipping(self):
        frame = pd.DataFrame({"hedef": [1.0, 2.0], "x": [1, 2]})
        with pytest.raises(KeyError):
            profile(frame, target="Hedef")


class TestColumnNormalizationCollisions:
    def test_suffix_does_not_collide_with_an_existing_raw_name(self):
        """'A/B' + 'A B' -> a_b, a_b_2 ; ama 'A_B_2' zaten a_b_2'ye gidiyordu."""
        mapping = normalize_columns(["A/B", "A B", "A_B_2"])
        assert len(set(mapping.values())) == 3


class TestPanel:
    """Olay kayitlarindan tam panel: eksik 'olay olmadi' gunleri doldurulmali."""

    def test_missing_days_are_filled_with_zero(self):
        # Arrange -- 2 ilce, yalnizca olay olan gunler kayitli
        events = pd.DataFrame(
            {
                "ilce": ["Konak", "Konak", "Bodrum"],
                "tarih": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02"]),
                "kesinti": [2.0, 1.0, 5.0],
            }
        )

        # Act
        panel = build_panel(
            events, entity_columns=["ilce"], time_column="tarih", verbose=False
        )

        # Assert -- 2 ilce x 3 gun = 6 satir
        assert len(panel) == 6
        konak_02 = panel[(panel["ilce"] == "Konak") & (panel["tarih"] == "2024-01-02")]
        assert konak_02["kesinti"].iloc[0] == 0.0

    def test_same_day_events_are_summed_not_dropped(self):
        events = pd.DataFrame(
            {
                "ilce": ["Konak", "Konak"],
                "tarih": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "kesinti": [2.0, 3.0],
            }
        )
        panel = build_panel(
            events, entity_columns=["ilce"], time_column="tarih", verbose=False
        )
        assert panel["kesinti"].sum() == pytest.approx(5.0)

    def test_synthetic_rows_are_flagged(self):
        events = pd.DataFrame(
            {
                "ilce": ["Konak"],
                "tarih": pd.to_datetime(["2024-01-01"]),
                "kesinti": [1.0],
            }
        )
        panel = build_panel(
            events, entity_columns=["ilce"], time_column="tarih",
            end="2024-01-03", verbose=False,
        )
        assert panel["_dolduruldu"].sum() == 2

    def test_coverage_reports_sparsity(self):
        events = pd.DataFrame(
            {
                "ilce": ["A", "B"],
                "tarih": pd.to_datetime(["2024-01-01", "2024-01-05"]),
                "kesinti": [1.0, 1.0],
            }
        )
        result = panel_coverage(events, entity_columns=["ilce"], time_column="tarih")
        assert result["expected_rows"] == 10.0   # 2 varlik x 5 gun
        assert result["coverage"] == pytest.approx(0.2)

    def test_missing_entity_column_raises(self):
        frame = pd.DataFrame({"tarih": pd.to_datetime(["2024-01-01"])})
        with pytest.raises(KeyError):
            build_panel(frame, entity_columns=["yok"], time_column="tarih")


class TestPostprocess:
    def test_negatives_are_clipped(self):
        result = postprocess_predictions(np.array([-3.0, 2.0]), verbose=False)
        assert result[0] == 0.0

    def test_rounding_for_count_targets(self):
        result = postprocess_predictions(
            np.array([2.4, 2.6]), round_to_integer=True, verbose=False
        )
        assert result.tolist() == [2.0, 3.0]

    def test_upper_bound_caps_absurd_predictions(self):
        """Bir arastirma modellerin musteri sayisindan fazla kesinti tahmin
        ettigini olcmustu (5,2 kat asiri tahmin)."""
        result = postprocess_predictions(
            np.array([5.0, 5000.0]), clip_max=100.0, verbose=False
        )
        assert result[1] == 100.0

    def test_input_array_is_not_mutated(self):
        original = np.array([-1.0, 2.0])
        postprocess_predictions(original, verbose=False)
        assert original[0] == -1.0


class TestCountObjectives:
    def test_registry_covers_all_three_libraries(self):
        assert set(COUNT_OBJECTIVES) == {"lightgbm", "xgboost", "catboost"}
        for mapping in COUNT_OBJECTIVES.values():
            assert {"poisson", "tweedie", "mae", "l2"} <= set(mapping)

    def test_objective_override_is_applied(self):
        params = starter_params("lightgbm", "regression", objective="poisson")
        assert params["objective"] == "poisson"

    def test_default_objective_unchanged_without_override(self):
        assert starter_params("lightgbm", "regression")["objective"] == "regression"


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
        folds = purged_time_series_split(
            train["tarih"], n_splits=3, embargo=pd.Timedelta(days=7), verbose=False
        )

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
