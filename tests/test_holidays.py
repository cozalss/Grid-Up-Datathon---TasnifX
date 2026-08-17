"""TR tatil feature testleri.

Her test, Faz 3 arastirmasinda bulunup bu makinede DOGRULANMIS bir bosluga
karsilik gelir. Tarihler holidays kutuphanesinden teyit edildi.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gridup.features.temporal import (
    ADMINISTRATIVE_LEAVE,
    HOLIDAY_CODES,
    add_turkish_holiday_features,
)


def _frame(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"tarih": pd.to_datetime(dates)})


class TestHalfDayHolidays:
    """Varsayilan holidays cagrisi arifeleri ATLIYOR -- yilda 3 gun."""

    def test_ramazan_arifesi_is_detected_as_half_day(self):
        # 2026-03-19: "Ramazan Bayrami (saat 13.00'ten)" -- kutuphaneden dogrulandi
        result = add_turkish_holiday_features(_frame(["2026-03-19"]), "tarih")
        row = result.iloc[0]
        assert row["tatil_yarim_gun"] == 1
        assert row["tatil_mi"] == 0  # tam gun tatil DEGIL
        assert row["tatil_agirligi"] == pytest.approx(0.5)

    def test_kurban_arifesi_is_detected(self):
        result = add_turkish_holiday_features(_frame(["2026-05-26"]), "tarih")
        assert result["tatil_yarim_gun"].iloc[0] == 1

    def test_cumhuriyet_arifesi_is_detected(self):
        result = add_turkish_holiday_features(_frame(["2026-10-28"]), "tarih")
        assert result["tatil_yarim_gun"].iloc[0] == 1

    def test_full_day_holiday_is_not_marked_half(self):
        # 29 Ekim tam gun tatildir
        result = add_turkish_holiday_features(_frame(["2026-10-29"]), "tarih")
        row = result.iloc[0]
        assert row["tatil_mi"] == 1
        assert row["tatil_yarim_gun"] == 0
        assert row["tatil_agirligi"] == pytest.approx(1.0)

    def test_disabling_half_days_loses_the_arife(self):
        """Kapatma secenegi calisiyor mu -- ve kapatinca gercekten kaybediyor muyuz."""
        with_half = add_turkish_holiday_features(
            _frame(["2026-03-19"]), "tarih", include_half_days=True
        )
        without = add_turkish_holiday_features(
            _frame(["2026-03-19"]), "tarih", include_half_days=False
        )
        assert with_half["tatil_agirligi"].iloc[0] == pytest.approx(0.5)
        assert without["tatil_agirligi"].iloc[0] == pytest.approx(0.0)

    def test_ordinary_day_has_zero_weight(self):
        result = add_turkish_holiday_features(_frame(["2026-03-10"]), "tarih")
        row = result.iloc[0]
        assert row["tatil_mi"] == 0
        assert row["tatil_yarim_gun"] == 0
        assert row["tatil_agirligi"] == pytest.approx(0.0)


class TestHolidayCodes:
    """Serbest metin yerine sabit kod: ';' cakismasi ve ASCII disi karakter icin."""

    def test_religious_holiday_gets_stable_code(self):
        result = add_turkish_holiday_features(_frame(["2026-03-20"]), "tarih")
        assert result["tatil_kod"].iloc[0] == HOLIDAY_CODES["ramazan"]

    def test_national_holiday_gets_stable_code(self):
        result = add_turkish_holiday_features(_frame(["2026-10-29"]), "tarih")
        assert result["tatil_kod"].iloc[0] == HOLIDAY_CODES["cumhuriyet"]

    def test_collision_is_flagged_and_religious_wins(self):
        """2023-04-23: 'Ramazan Bayrami; Ulusal Egemenlik ve Cocuk Bayrami'.

        Dini bayram kazanir -- uc gunluk tatil, seyahat, sanayinin durmasi
        acisindan baskin olan odur.
        """
        result = add_turkish_holiday_features(_frame(["2023-04-23"]), "tarih")
        row = result.iloc[0]
        assert row["tatil_cakisma"] == 1
        assert row["tatil_kod"] == HOLIDAY_CODES["ramazan"]

    def test_non_holiday_gets_zero_code(self):
        result = add_turkish_holiday_features(_frame(["2026-03-10"]), "tarih")
        assert result["tatil_kod"].iloc[0] == 0

    def test_code_column_is_integer_not_text(self):
        """Metin kolonu Turkce CSV'de ';' ile sutun kaydirabilir."""
        result = add_turkish_holiday_features(
            _frame(["2023-04-23", "2026-10-29", "2026-03-10"]), "tarih"
        )
        assert pd.api.types.is_integer_dtype(result["tatil_kod"])

    def test_no_generated_column_contains_semicolon(self):
        """Uretilen hicbir deger ';' icermemeli -- TR CSV'de alan ayiricisidir."""
        result = add_turkish_holiday_features(
            _frame(["2022-05-01", "2023-04-23", "2026-03-19"]), "tarih"
        )
        new_columns = [c for c in result.columns if c.startswith("tatil")]
        for column in new_columns:
            values = result[column].astype(str)
            assert not values.str.contains(";").any(), f"{column} ';' iceriyor"


class TestAdministrativeLeave:
    def test_known_leave_day_is_flagged(self):
        # 2024-06-20: Kurban cevresi idari izin (arastirmadan, dogrulanmadi)
        result = add_turkish_holiday_features(_frame(["2024-06-20"]), "tarih")
        assert result["tatil_idari_izin"].iloc[0] == pytest.approx(1.0)

    def test_half_day_leave_has_half_weight(self):
        result = add_turkish_holiday_features(_frame(["2024-04-09"]), "tarih")
        assert result["tatil_idari_izin"].iloc[0] == pytest.approx(0.5)

    def test_leave_table_can_be_replaced(self):
        result = add_turkish_holiday_features(
            _frame(["2030-01-15"]), "tarih", administrative_leave={"2030-01-15": 1.0}
        )
        assert result["tatil_idari_izin"].iloc[0] == pytest.approx(1.0)

    def test_leave_can_be_disabled(self):
        result = add_turkish_holiday_features(
            _frame(["2024-06-20"]), "tarih", administrative_leave={}
        )
        assert result["tatil_idari_izin"].iloc[0] == pytest.approx(0.0)

    def test_default_table_is_not_empty(self):
        assert len(ADMINISTRATIVE_LEAVE) > 5


class TestWorkforceLoss:
    """Sebeke tarafinda anlamli soru: bugun saha ekibi calisiyor mu?"""

    def test_full_holiday_is_total_loss(self):
        result = add_turkish_holiday_features(_frame(["2026-10-29"]), "tarih")
        assert result["tatil_isgucu_kaybi"].iloc[0] == pytest.approx(1.0)

    def test_arife_is_half_loss(self):
        # 2026-03-19 Persembe -- hafta ici, yalnizca arife etkisi
        result = add_turkish_holiday_features(_frame(["2026-03-19"]), "tarih")
        assert result["tatil_isgucu_kaybi"].iloc[0] == pytest.approx(0.5)

    def test_weekend_counts_as_loss(self):
        # 2026-03-14 Cumartesi
        result = add_turkish_holiday_features(_frame(["2026-03-14"]), "tarih")
        assert result["tatil_isgucu_kaybi"].iloc[0] == pytest.approx(1.0)

    def test_ordinary_weekday_is_zero(self):
        result = add_turkish_holiday_features(_frame(["2026-03-10"]), "tarih")
        assert result["tatil_isgucu_kaybi"].iloc[0] == pytest.approx(0.0)


class TestRobustness:
    def test_broken_date_does_not_become_a_holiday(self):
        result = add_turkish_holiday_features(
            pd.DataFrame({"tarih": ["2026-10-29", "bozuk-tarih"]}), "tarih"
        )
        broken = result.iloc[1]
        assert broken["tatil_mi"] == 0
        assert broken["tatil_yakininda"] == 0
        assert broken["tatil_kod"] == 0

    def test_input_frame_is_not_mutated(self):
        frame = _frame(["2026-10-29"])
        before = frame.copy()
        add_turkish_holiday_features(frame, "tarih")
        pd.testing.assert_frame_equal(frame, before)

    def test_all_generated_columns_are_numeric(self):
        """Model dogrudan besleyebilmeli -- kategorik kodlama gerekmesin."""
        result = add_turkish_holiday_features(
            _frame(["2026-03-19", "2026-10-29", "2026-03-10"]), "tarih"
        )
        for column in [c for c in result.columns if c.startswith("tatil")]:
            assert pd.api.types.is_numeric_dtype(result[column]), column

    def test_moving_religious_holidays_shift_across_years(self):
        """Dini bayramlar hicri takvimle kayar -- ay+gun kolonlari yakalayamaz."""
        dates = ["2023-04-21", "2024-04-10", "2025-03-30", "2026-03-20"]
        result = add_turkish_holiday_features(_frame(dates), "tarih")
        # Hepsi Ramazan Bayrami ilk gunu, ama farkli ay/gun kombinasyonlari
        assert (result["tatil_kod"] == HOLIDAY_CODES["ramazan"]).all()
        months = pd.to_datetime(dates).month
        assert len(set(months)) > 1, "tarihler ayni ayda -- test anlamsiz"


class TestDistanceStillWorks:
    def test_distance_is_zero_on_a_holiday(self):
        result = add_turkish_holiday_features(_frame(["2026-10-29"]), "tarih")
        assert result["tatil_mesafe"].iloc[0] == 0

    def test_distance_grows_away_from_holidays(self):
        result = add_turkish_holiday_features(
            _frame(["2026-10-29", "2026-10-31", "2026-11-10"]), "tarih"
        )
        distances = result["tatil_mesafe"].to_numpy()
        assert distances[0] < distances[1] < distances[2]

    def test_near_flag_respects_window(self):
        result = add_turkish_holiday_features(_frame(["2026-10-31"]), "tarih", window_days=3)
        assert result["tatil_yakininda"].iloc[0] == 1
        far = add_turkish_holiday_features(_frame(["2026-10-31"]), "tarih", window_days=1)
        assert far["tatil_yakininda"].iloc[0] == 0


def test_no_nan_in_generated_columns():
    dates = pd.date_range("2024-01-01", "2026-12-31", freq="D").strftime("%Y-%m-%d").tolist()
    result = add_turkish_holiday_features(_frame(dates), "tarih")
    generated = [c for c in result.columns if c.startswith("tatil")]
    assert not result[generated].isna().to_numpy().any()
