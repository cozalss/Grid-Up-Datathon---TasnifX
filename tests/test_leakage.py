"""Sizinti korumalari testi.

Bu dosyadaki her test, CV skorunu yapay olarak yukseltip leaderboard'da coke
gitmeye yol acan bir hatayi engeller. Bunlar "guzel olsa iyi olur" testleri
degil; yarismanin kaybedildigi yerlerdir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.features.categorical import oof_target_encode
from gridup.features.temporal import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    shared_origin,
)
from gridup.validation import (
    assert_folds_align,
    build_splitter,
    check_train_test_overlap,
    leakage_report,
    purged_time_series_split,
    suggest_scheme,
)


@pytest.fixture
def time_series_frame() -> pd.DataFrame:
    """Iki varlik, 200 gunluk gunluk seri."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    frames = [
        pd.DataFrame(
            {
                "tarih": dates,
                "trafo_id": entity,
                "tuketim": rng.normal(100, 10, size=len(dates)).cumsum(),
                "hedef": rng.normal(50, 5, size=len(dates)),
            }
        )
        for entity in ("TR001", "TR002")
    ]
    return pd.concat(frames, ignore_index=True)


class TestOofTargetEncoding:
    """Hedef kodlama fold-disi olmak ZORUNDA."""

    def test_refuses_to_run_without_folds(self):
        """Fold'suz cagri sessizce sizintili kodlama uretmek yerine HATA vermeli."""
        train = pd.DataFrame({"kategori": ["a", "b", "a", "b"]})
        target = pd.Series([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="fold"):
            oof_target_encode(train, target, ["kategori"], folds=[])

    def test_encoding_of_a_row_excludes_that_rows_own_target(self):
        """Sizintinin kanit testi.

        Her kategoriden bir satir kendi fold'unda: naif kodlama o satira KENDI
        hedefini verir. Fold-disi kodlama veremez.
        """
        # Arrange -- 4 satir, 2 kategori, her kategoride cok farkli iki deger
        train = pd.DataFrame({"kategori": ["a", "a", "b", "b"]})
        target = pd.Series([0.0, 100.0, 0.0, 100.0])
        folds = [
            (np.array([1, 3]), np.array([0, 2])),
            (np.array([0, 2]), np.array([1, 3])),
        ]

        # Act
        encoded, _ = oof_target_encode(train, target, ["kategori"], folds, smoothing=0.0)
        values = encoded["kategori_hedef_kod"].to_numpy()

        # Assert -- 0. satir yalnizca 1. satiri (100.0) gorebilir
        assert values[0] == pytest.approx(100.0)
        assert values[1] == pytest.approx(0.0)

    def test_unseen_category_falls_back_to_prior_not_nan(self):
        train = pd.DataFrame({"kategori": ["a", "a", "b", "b"]})
        target = pd.Series([1.0, 2.0, 3.0, 4.0])
        folds = [(np.array([0, 1]), np.array([2, 3]))]

        encoded, _ = oof_target_encode(train, target, ["kategori"], folds)

        assert not encoded["kategori_hedef_kod"].isna().any()

    def test_test_encoding_uses_full_train(self):
        train = pd.DataFrame({"kategori": ["a", "a", "b", "b"]})
        test = pd.DataFrame({"kategori": ["a", "b"]})
        target = pd.Series([10.0, 10.0, 20.0, 20.0])
        folds = [(np.array([0, 2]), np.array([1, 3])), (np.array([1, 3]), np.array([0, 2]))]

        _, encoded_test = oof_target_encode(
            train, target, ["kategori"], folds, test=test, smoothing=0.0
        )

        assert encoded_test is not None
        values = encoded_test["kategori_hedef_kod"].to_numpy()
        assert values[0] == pytest.approx(10.0)
        assert values[1] == pytest.approx(20.0)


class TestRollingWindowLeakage:
    """Kayan pencere mevcut satiri DISLAMALI."""

    def test_rolling_mean_excludes_current_row(self, time_series_frame):
        # Arrange -- bilinen degerlerle tek varlik
        frame = pd.DataFrame(
            {
                "tarih": pd.date_range("2024-01-01", periods=5, freq="D"),
                "trafo_id": "TR001",
                "deger": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

        # Act
        result = add_rolling_features(
            frame, "deger", [2], time_column="tarih",
            group_columns=["trafo_id"], aggregations=("mean",),
        )
        rolled = result["deger_kayan2_mean"].to_numpy()

        # Assert -- 2. satirin penceresi {10, 20}: kendi degeri (30) DAHIL DEGIL
        assert np.isnan(rolled[0])
        assert rolled[1] == pytest.approx(10.0)
        assert rolled[2] == pytest.approx(15.0)

    def test_lag_features_do_not_leak_future(self):
        frame = pd.DataFrame(
            {
                "tarih": pd.date_range("2024-01-01", periods=4, freq="D"),
                "trafo_id": "TR001",
                "deger": [1.0, 2.0, 3.0, 4.0],
            }
        )

        result = add_lag_features(
            frame, "deger", [1], time_column="tarih", group_columns=["trafo_id"]
        )
        lagged = result["deger_lag1"].to_numpy()

        assert np.isnan(lagged[0])
        assert lagged[1] == pytest.approx(1.0)
        assert lagged[3] == pytest.approx(3.0)

    def test_lag_respects_group_boundaries(self, time_series_frame):
        """Bir varligin ilk satiri, ONCEKI VARLIGIN son degerini almamali."""
        result = add_lag_features(
            time_series_frame, "tuketim", [1],
            time_column="tarih", group_columns=["trafo_id"],
        )
        first_rows = result.groupby("trafo_id", observed=True).head(1)
        assert first_rows["tuketim_lag1"].isna().all()

    def test_input_frame_is_not_mutated(self, time_series_frame):
        before = time_series_frame.copy()
        add_lag_features(
            time_series_frame, "tuketim", [1, 7],
            time_column="tarih", group_columns=["trafo_id"],
        )
        pd.testing.assert_frame_equal(time_series_frame, before)

    def test_output_preserves_original_row_order(self):
        """Fonksiyon icinde siralama yapiliyor -- cikti girdi sirasinda donmeli."""
        frame = pd.DataFrame(
            {
                "tarih": pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]),
                "trafo_id": "TR001",
                "deger": [30.0, 10.0, 20.0],
            }
        )
        result = add_lag_features(
            frame, "deger", [1], time_column="tarih", group_columns=["trafo_id"]
        )
        # Girdi sirasi korunmus: 0. satir hala 2024-01-03
        assert result.loc[0, "tarih"] == pd.Timestamp("2024-01-03")
        # 2024-01-03'un lag1'i 2024-01-02'nin degeri = 20.0
        assert result.loc[0, "deger_lag1"] == pytest.approx(20.0)


class TestPurgedSplit:
    def test_embargo_creates_gap_between_train_and_valid(self, time_series_frame):
        times = time_series_frame["tarih"]
        embargo = pd.Timedelta(days=10)

        for train_idx, valid_idx in purged_time_series_split(
            times, n_splits=3, embargo=embargo
        ):
            train_max = times.iloc[train_idx].max()
            valid_min = times.iloc[valid_idx].min()
            assert valid_min - train_max >= embargo

    def test_train_always_precedes_validation(self, time_series_frame):
        times = time_series_frame["tarih"]
        for train_idx, valid_idx in purged_time_series_split(
            times, n_splits=3, embargo=pd.Timedelta(days=1)
        ):
            assert times.iloc[train_idx].max() < times.iloc[valid_idx].min()

    def test_empty_series_raises(self):
        with pytest.raises(ValueError):
            purged_time_series_split(
                pd.Series([], dtype="datetime64[ns]"), embargo=pd.Timedelta(0)
            )

    def test_embargo_is_required(self):
        """Ambargo verilmezse fonksiyon CALISMAZ.

        Onceki varsayilan (`total_span / (n_splits+1) * 0.01`) 3 yillik gunluk
        veride ~2 gun uretiyordu -- 7/14/30 gunluk kayan pencerelerin neredeyse
        tamami fold sinirini asiyordu. Sessizce yetersiz bir varsayilan yerine
        acik bir secim istiyoruz.
        """
        times = pd.Series(pd.date_range("2024-01-01", periods=50, freq="D"))
        with pytest.raises(TypeError):
            purged_time_series_split(times, n_splits=3)  # type: ignore[call-arg]

    def test_oversized_embargo_raises_instead_of_returning_nothing(self):
        times = pd.Series(pd.date_range("2024-01-01", periods=40, freq="D"))
        with pytest.raises(ValueError, match="Hicbir fold"):
            purged_time_series_split(times, n_splits=3, embargo=pd.Timedelta(days=500))

    def test_string_dates_are_sorted_chronologically_not_lexically(self):
        """Sifir dolgusuz metin tarih: '2024-1-10' < '2024-1-2' sozluksel olarak."""
        times = pd.Series(["2024-1-2", "2024-1-20", "2024-1-3", "2024-1-9", "2024-1-10"] * 6)
        folds = purged_time_series_split(times, n_splits=2, embargo=pd.Timedelta(0), verbose=False)
        parsed = pd.to_datetime(times)
        for train_idx, valid_idx in folds:
            assert parsed.iloc[train_idx].max() <= parsed.iloc[valid_idx].min()


class TestFoldAlignment:
    """Fold indeksleri frame ile hizali degilse GURULTULU hata."""

    def test_out_of_bounds_index_raises(self):
        folds = [(np.array([0, 1]), np.array([2, 99]))]
        with pytest.raises(ValueError, match="indeksi"):
            assert_folds_align(10, folds)

    def test_overlapping_train_and_valid_raises(self):
        folds = [(np.array([0, 1, 2]), np.array([2, 3]))]
        with pytest.raises(ValueError, match="hem train hem valid"):
            assert_folds_align(10, folds)

    def test_empty_fold_side_raises(self):
        with pytest.raises(ValueError, match="bos"):
            assert_folds_align(10, [(np.array([]), np.array([1]))])

    def test_valid_folds_pass(self):
        assert_folds_align(10, [(np.array([0, 1, 2]), np.array([3, 4]))])

    def test_cross_validate_rejects_misaligned_folds(self):
        """Feature asamasinda satir sayisi degistiyse CV sessizce devam ETMEMELI."""
        from gridup.models import cross_validate

        frame = pd.DataFrame({"a": range(20), "b": range(20)})
        folds = [(np.array([0, 1]), np.array([500, 501]))]  # baska bir frame'den
        with pytest.raises(ValueError, match="indeksi"):
            cross_validate(frame, np.arange(20), folds, verbose=False)


class TestLeakageReport:
    def test_detects_target_derived_column(self):
        rng = np.random.default_rng(0)
        target = rng.normal(50, 10, size=200)
        frame = pd.DataFrame(
            {
                "hedef": target,
                "hedefin_kopyasi": target * 2 + 1,  # mukemmel korelasyon
                "gurultu": rng.normal(0, 1, size=200),
            }
        )

        findings = leakage_report(frame, "hedef")

        assert any("hedefin_kopyasi" in message for message in findings["critical"])

    def test_detects_time_overlap_between_train_and_test(self):
        train = pd.DataFrame({"tarih": pd.date_range("2024-01-01", periods=100), "hedef": 1.0})
        test = pd.DataFrame({"tarih": pd.date_range("2024-02-01", periods=50)})

        findings = leakage_report(train, "hedef", test=test, time_column="tarih")

        assert any("ortusme" in message.lower() for message in findings["critical"])

    def test_clean_time_split_produces_no_critical_finding(self):
        train = pd.DataFrame({"tarih": pd.date_range("2024-01-01", periods=100), "hedef": 1.0})
        test = pd.DataFrame({"tarih": pd.date_range("2024-05-01", periods=50)})

        findings = leakage_report(train, "hedef", test=test, time_column="tarih")

        assert not findings["critical"]

    def test_flags_columns_missing_from_test(self):
        train = pd.DataFrame({"a": [1, 2], "sadece_train": [3, 4], "hedef": [0.0, 1.0]})
        test = pd.DataFrame({"a": [5, 6]})

        findings = leakage_report(train, "hedef", test=test)

        assert any("sadece_train" in message for message in findings["warning"])

    def test_missing_target_raises(self):
        with pytest.raises(ValueError, match="yok"):
            leakage_report(pd.DataFrame({"a": [1]}), "olmayan_hedef")

    def test_non_numeric_target_reports_that_the_check_was_skipped(self):
        """En guclu kontrol atlaniyorsa bunu SOYLEMELI -- '0 kritik' yaniltir."""
        frame = pd.DataFrame({"durum": ["ARIZALI", "NORMAL"] * 50, "x": range(100)})

        findings = leakage_report(frame, "durum")

        assert any("ATLANDI" in message for message in findings["info"])


class TestHorizonAwareFeatures:
    """Tahmin ufku: test ILERIDEKI bir blok ise lag'ler o kadar kaydirilmali."""

    def test_horizon_shifts_lag_beyond_the_prediction_gap(self):
        frame = pd.DataFrame(
            {
                "tarih": pd.date_range("2024-01-01", periods=10, freq="D"),
                "trafo_id": "TR001",
                "deger": [float(i) for i in range(10)],
            }
        )

        result = add_lag_features(
            frame, "deger", [1], time_column="tarih",
            group_columns=["trafo_id"], horizon=5,
        )
        lagged = result["deger_ufuk5_lag1"].to_numpy()

        # Ufuk 5 -> 5. satirin en taze mevcut degeri 0. satirinki (5 adim geride)
        assert np.isnan(lagged[:5]).all()
        assert lagged[5] == pytest.approx(0.0)
        assert lagged[9] == pytest.approx(4.0)

    def test_horizon_one_matches_previous_behaviour(self):
        frame = pd.DataFrame(
            {
                "tarih": pd.date_range("2024-01-01", periods=5, freq="D"),
                "trafo_id": "TR001",
                "deger": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        result = add_lag_features(
            frame, "deger", [1], time_column="tarih", group_columns=["trafo_id"]
        )
        assert result["deger_lag1"].iloc[1] == pytest.approx(1.0)

    def test_rolling_respects_horizon(self):
        frame = pd.DataFrame(
            {
                "tarih": pd.date_range("2024-01-01", periods=8, freq="D"),
                "trafo_id": "TR001",
                "deger": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            }
        )
        result = add_rolling_features(
            frame, "deger", [2], time_column="tarih",
            group_columns=["trafo_id"], horizon=3, aggregations=("mean",),
        )
        rolled = result["deger_ufuk3_kayan2_mean"].to_numpy()

        # shift(3) -> [NaN, NaN, NaN, 10, 20, 30, 40, 50]
        # 4. satirin 2'lik penceresi = {10, 20} (ham indeks 0 ve 1) -> 15
        # Yani model, 3 adim once bile bilmedigi hicbir degeri gormuyor.
        assert rolled[4] == pytest.approx(15.0)
        assert np.isnan(rolled[:3]).all()

    def test_zero_horizon_rejected(self):
        frame = pd.DataFrame(
            {"tarih": pd.date_range("2024-01-01", periods=3), "deger": [1.0, 2.0, 3.0]}
        )
        with pytest.raises(ValueError, match="horizon"):
            add_lag_features(frame, "deger", [1], time_column="tarih", horizon=0)


class TestCalendarOriginDrift:
    """Gun sayaci: ortak origin verilmezse URETILMEZ."""

    def test_day_counter_is_absent_without_explicit_origin(self):
        frame = pd.DataFrame({"tarih": pd.date_range("2024-01-01", periods=5)})
        result = add_calendar_features(frame, "tarih")
        assert "tarih_gun_sayaci" not in result.columns

    def test_shared_origin_keeps_train_and_test_on_one_axis(self):
        """Bulunan HIGH hatanin regresyon testi.

        Ayri ayri hesaplanan origin ile test sayaci 0'dan basliyordu ve model
        test'in en guncel satirlarini train'in en eski donemi saniyordu.
        """
        train = pd.DataFrame({"tarih": pd.date_range("2024-01-01", "2025-09-30", freq="D")})
        test = pd.DataFrame({"tarih": pd.date_range("2025-10-01", "2025-12-31", freq="D")})

        origin = shared_origin(train, test, time_column="tarih")
        train_out = add_calendar_features(train, "tarih", origin=origin)
        test_out = add_calendar_features(test, "tarih", origin=origin)

        assert test_out["tarih_gun_sayaci"].min() > train_out["tarih_gun_sayaci"].max()

    def test_shared_origin_requires_a_valid_date(self):
        frame = pd.DataFrame({"tarih": ["bozuk", "yine bozuk"]})
        with pytest.raises(ValueError):
            shared_origin(frame, time_column="tarih")


class TestSchemeSuggestion:
    def test_time_column_yields_time_series_scheme(self, time_series_frame):
        suggestion = suggest_scheme(time_series_frame, target="hedef", known_time="tarih")
        assert suggestion.scheme == "TimeSeriesSplit"

    def test_repeated_entity_yields_group_scheme(self):
        frame = pd.DataFrame({"trafo_id": ["A"] * 50 + ["B"] * 50, "hedef": range(100)})
        suggestion = suggest_scheme(frame, target="hedef", known_group="trafo_id")
        assert "Group" in suggestion.scheme

    def test_warns_when_both_time_and_group_present(self, time_series_frame):
        suggestion = suggest_scheme(
            time_series_frame, target="hedef", known_time="tarih", known_group="trafo_id"
        )
        assert any("purged" in warning.lower() for warning in suggestion.warnings)

    def test_builder_returns_usable_splitter(self):
        splitter = build_splitter("KFold", n_splits=3, seed=1)
        frame = pd.DataFrame({"a": range(30)})
        assert len(list(splitter.split(frame))) == 3

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="Bilinmeyen"):
            build_splitter("OlmayanSema")


class TestTrainTestOverlap:
    def test_detects_shared_entities(self):
        train = pd.DataFrame({"trafo_id": ["A", "B", "C"]})
        test = pd.DataFrame({"trafo_id": ["B", "C", "D"]})

        result = check_train_test_overlap(train, test, ["trafo_id"])

        assert result["overlap"] == 2
        assert "GroupKFold" in result["note"]

    def test_reports_no_overlap(self):
        train = pd.DataFrame({"trafo_id": ["A", "B"]})
        test = pd.DataFrame({"trafo_id": ["C", "D"]})

        result = check_train_test_overlap(train, test, ["trafo_id"])

        assert result["overlap"] == 0
