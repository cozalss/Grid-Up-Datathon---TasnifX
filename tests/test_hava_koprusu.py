"""Hava forecast koprusu (scripts/fetch_weather_bridge.py) -- cevrimdisi sozlesme.

Ag cagrisi yok: ``kopru_kur`` ve ``dikis_kontrolu`` sentetik arsiv/forecast
tablolariyla dogrulanir. P0-10 (2026-08-18 denetimi): arsiv 2026-08-09'da
bitiyordu, test blogu asarsa hava kolonlari yalnizca testte NaN oluyordu.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

KOK = Path(__file__).resolve().parents[1]


def _kopru():
    yol = KOK / "scripts" / "fetch_weather_bridge.py"
    spec = importlib.util.spec_from_file_location("fetch_weather_bridge", yol)
    modul = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(modul)
    return modul


def _tablo(bas: str, gun: int, sicaklik: float, *, konum: str = "İzmir-Bornova") -> pd.DataFrame:
    m = _kopru()
    tarih = pd.date_range(bas, periods=gun, freq="D")
    frame = pd.DataFrame(
        {
            "konum": konum,
            "konum_key": "izmir-bornova",
            "il_key": "izmir",
            "ilce_key": "bornova",
            "tarih": tarih,
            "sicaklik_max": sicaklik + 5,
            "sicaklik_min": sicaklik - 5,
            "sicaklik_ort": sicaklik,
            "hissedilen_max": sicaklik + 4,
            "yagis_toplam": 0.0,
            "yagmur_toplam": 0.0,
            "kar_toplam": 0.0,
            "yagis_saati": 0.0,
            "ruzgar_max": 10.0,
            "firtina_max": 20.0,
            "gunes_radyasyon": 20.0,
        }
    )
    del m
    return frame


def test_kopru_yalnizca_arsiv_disi_gunleri_bayrakla_ekler() -> None:
    m = _kopru()
    arsiv = m.fw.add_derived_features(_tablo("2026-08-01", 9, 30.0))  # ..08-09
    tahmin = _tablo("2026-08-08", 10, 31.0)  # 08-08..08-17 (2 gun ortusme)
    fark = m.dikis_kontrolu(arsiv, tahmin)
    assert np.isclose(fark, 1.0)
    birlesik = m.kopru_kur(arsiv, tahmin)
    assert birlesik["tarih"].max() == pd.Timestamp("2026-08-17")
    assert birlesik["tarih"].is_unique
    bayrak = birlesik.set_index("tarih")[m.KAYNAK_KOLONU]
    assert (bayrak.loc[:"2026-08-09"] == 0).all()
    assert (bayrak.loc["2026-08-10":] == 1).all()
    assert int(bayrak.sum()) == 8
    assert set(arsiv.columns) | {m.KAYNAK_KOLONU} == set(birlesik.columns)
    # Turetilmis kolonlar forecast satirlarinda da dolu (ayni fonksiyon)
    assert birlesik["sogutma_derece_gun"].notna().all()


def test_dikis_esigi_asilirsa_reddedilir() -> None:
    m = _kopru()
    arsiv = _tablo("2026-08-01", 9, 30.0)
    tahmin = _tablo("2026-08-08", 10, 36.0)  # 6 C fark
    with pytest.raises(ValueError, match="Dikis basarisiz"):
        m.dikis_kontrolu(arsiv, tahmin)
    with pytest.raises(ValueError, match="ortusen gun yok"):
        m.dikis_kontrolu(arsiv, _tablo("2026-09-01", 3, 30.0))


def test_eski_tahmin_satirlari_yeni_arsivle_degistirilir() -> None:
    """Ikinci kosu: arsiv 08-12'ye uzadiysa eski tahmin 08-10..08-12 arsivle EZILIR."""
    m = _kopru()
    ilk = m.kopru_kur(
        m.fw.add_derived_features(_tablo("2026-08-01", 9, 30.0)), _tablo("2026-08-08", 10, 31.0)
    )
    yeni_arsiv = m.fw.add_derived_features(_tablo("2026-08-01", 12, 30.0))  # ..08-12
    yeni_arsiv[m.KAYNAK_KOLONU] = 0
    ikinci = m.kopru_kur(
        pd.concat([yeni_arsiv, ilk[ilk[m.KAYNAK_KOLONU] == 1]], ignore_index=True),
        _tablo("2026-08-11", 10, 31.5),
    )
    bayrak = ikinci.set_index("tarih")[m.KAYNAK_KOLONU]
    assert (bayrak.loc[:"2026-08-12"] == 0).all()
    assert (bayrak.loc["2026-08-13":] == 1).all()
    assert ikinci["tarih"].is_unique
