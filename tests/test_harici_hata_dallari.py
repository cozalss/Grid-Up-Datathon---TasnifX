"""Harici veri feature'larinin HATA ve KENAR dallari.

test_harici_sizinti.py mutlu yolu ve sizintiyi olcer; burada girdi sozlesmesi
zorlanir: eksik kolon KeyError, gecersiz parametre ValueError, bos secim
girdiyi degistirmeden dondurur, nufus normalizasyonu 0 nufusta NaN birakir.
Bunlar CI'da veri dosyasi olmadan da kosar -- yani kapsam raporunda gorunen
tek dogrulama katmani budur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.features.national import (
    add_annual_district_attribute,
    add_national_series,
    add_seasonal_district_profile,
    daily_from_hourly,
)
from gridup.features.tourism import add_monthly_attribute, district_monthly_estimate


def _panel() -> pd.DataFrame:
    gunler = pd.date_range("2026-01-01", periods=10, freq="D")
    return pd.DataFrame([("bornova", g) for g in gunler], columns=["ilce_key", "tarih"])


def _yillik() -> pd.DataFrame:
    return pd.DataFrame({"ilce_key": ["bornova"], "yil": [2025], "geceleme": [1000.0]})


# --- daily_from_hourly / add_national_series -----------------------------


def test_daily_from_hourly_eksik_kolonlar_keyerror() -> None:
    saatlik = pd.DataFrame(
        {"zaman": pd.date_range("2026-01-01", periods=3, freq="h"), "x": [1, 2, 3]}
    )
    with pytest.raises(KeyError, match="'yok' kolonu yok"):
        daily_from_hourly(saatlik, time_column="yok", value_columns=["x"])
    with pytest.raises(KeyError, match="su kolonlar yok"):
        daily_from_hourly(saatlik, time_column="zaman", value_columns=["x", "y"])


def test_add_national_series_sozlesme_dallari() -> None:
    gunluk = pd.DataFrame({"tarih": pd.date_range("2026-01-01", periods=10, freq="D"), "tr": 1.0})
    with pytest.raises(KeyError, match="frame icinde 'yok'"):
        add_national_series(_panel(), gunluk, time_column="yok", horizon=1)
    with pytest.raises(KeyError, match="daily_national icinde"):
        add_national_series(
            _panel(), gunluk, time_column="tarih", national_time_column="yok", horizon=1
        )
    # Deger kolonu secilmemis ve tabloda zaman disinda kolon yoksa: kopya doner
    sadece_zaman = gunluk[["tarih"]]
    sonuc = add_national_series(_panel(), sadece_zaman, time_column="tarih", horizon=1)
    pd.testing.assert_frame_equal(sonuc, _panel())


# --- add_annual_district_attribute -----------------------------------------


def test_annual_attribute_negatif_lag_ve_eksik_kolonlar() -> None:
    with pytest.raises(ValueError, match="year_lag negatif"):
        add_annual_district_attribute(
            _panel(), _yillik(), key_column="ilce_key", time_column="tarih",
            value_columns=["geceleme"], year_lag=-1,
        )  # fmt: skip
    with pytest.raises(KeyError, match="frame icinde 'yok'"):
        add_annual_district_attribute(
            _panel(), _yillik(), key_column="yok", time_column="tarih", value_columns=["geceleme"]
        )
    with pytest.raises(KeyError, match="annual icinde 'yok'"):
        add_annual_district_attribute(
            _panel(), _yillik(), key_column="ilce_key", time_column="tarih", value_columns=["yok"]
        )


def test_annual_attribute_nufus_normalizasyonu() -> None:
    """Kisi basi kolon uretilir; nufus 0 ise NaN (sonsuz degil); nufus kolonu frame'e sizmaz."""
    yillik = pd.DataFrame(
        {"ilce_key": ["bornova", "sifir"], "yil": [2025, 2025], "geceleme": [1000.0, 500.0]}
    )
    nufus = pd.DataFrame({"ilce_key": ["bornova", "sifir"], "nufus": [200, 0]})
    panel = pd.DataFrame(
        {"ilce_key": ["bornova", "sifir"], "tarih": pd.to_datetime(["2026-03-01", "2026-03-01"])}
    )
    sonuc = add_annual_district_attribute(
        panel, yillik, key_column="ilce_key", time_column="tarih", value_columns=["geceleme"],
        population=nufus, prefix="t",
    )  # fmt: skip
    assert "t_geceleme_kisi_basi" in sonuc.columns
    assert "nufus" not in sonuc.columns
    assert np.isclose(sonuc.loc[0, "t_geceleme_kisi_basi"], 5.0)
    assert np.isnan(sonuc.loc[1, "t_geceleme_kisi_basi"])
    assert not np.isinf(sonuc["t_geceleme_kisi_basi"].fillna(0)).any()


def test_annual_attribute_nufus_kolonu_eksikse_keyerror() -> None:
    with pytest.raises(KeyError, match="population icinde"):
        add_annual_district_attribute(
            _panel(), _yillik(), key_column="ilce_key", time_column="tarih",
            value_columns=["geceleme"], population=pd.DataFrame({"x": [1]}),
        )  # fmt: skip


# --- add_seasonal_district_profile -----------------------------------------


def test_seasonal_profile_sozlesme_dallari() -> None:
    profil = pd.DataFrame({"ilce_key": ["bornova"] * 12, "ay": range(1, 13), "su": range(12)})
    with pytest.raises(KeyError, match="frame icinde 'yok'"):
        add_seasonal_district_profile(_panel(), profil, key_column="yok", time_column="tarih")
    with pytest.raises(KeyError, match="profile icinde 'yok'"):
        add_seasonal_district_profile(
            _panel(), profil, key_column="ilce_key", time_column="tarih", month_column="yok"
        )
    # Deger kolonu yoksa kopya doner
    sonuc = add_seasonal_district_profile(
        _panel(), profil[["ilce_key", "ay"]], key_column="ilce_key", time_column="tarih"
    )
    pd.testing.assert_frame_equal(sonuc, _panel())
    # Tekrarlanan (anahtar, ay) satir cogaltir -> reddedilir
    tekrar = pd.concat([profil, profil.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="tekrarlanan"):
        add_seasonal_district_profile(_panel(), tekrar, key_column="ilce_key", time_column="tarih")
    # Mutlu yol: Ocak satirlari ay=1 degerini alir, girdi degismez
    panel = _panel()
    kopya = panel.copy()
    sonuc = add_seasonal_district_profile(panel, profil, key_column="ilce_key", time_column="tarih")
    pd.testing.assert_frame_equal(panel, kopya)
    assert (sonuc["mevsim_su"] == 0).all()


# --- tourism -----------------------------------------------------------------


def test_monthly_attribute_eksik_kolonlar_keyerror() -> None:
    aylik = pd.DataFrame({"il_key": ["mugla"], "yil": [2025], "ay": [7], "geceleme": [1.0]})
    panel = pd.DataFrame({"il_key": ["mugla"], "tarih": pd.to_datetime(["2026-07-15"])})
    with pytest.raises(KeyError, match="frame icinde 'yok'"):
        add_monthly_attribute(
            panel, aylik, key_column="yok", time_column="tarih", value_columns=["geceleme"]
        )
    with pytest.raises(KeyError, match="monthly icinde 'yok'"):
        add_monthly_attribute(
            panel, aylik, key_column="il_key", time_column="tarih", value_columns=["yok"]
        )


def test_district_estimate_eksik_kolonlar_keyerror() -> None:
    yillik = pd.DataFrame(
        {"ilce_key": ["bodrum"], "il_key": ["mugla"], "yil": [2025], "geceleme": [1.0]}
    )
    aylik = pd.DataFrame({"il_key": ["mugla"], "yil": [2025], "ay": [7], "geceleme": [1.0]})
    with pytest.raises(KeyError, match="annual icinde"):
        district_monthly_estimate(yillik.drop(columns=["il_key"]), aylik)
    with pytest.raises(KeyError, match="monthly icinde"):
        district_monthly_estimate(yillik, aylik.drop(columns=["ay"]))
