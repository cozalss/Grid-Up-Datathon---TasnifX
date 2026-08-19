"""Ufuk/ambargo geometrisi (P1-3, 2026-08-18 denetimi).

Eski formul ``HORIZON = (test.max - test.min) + 1`` train ile test arasindaki
BOSLUGU yok sayiyordu. Olculdu: 100 gun train + 10 gun bosluk + 20 gun test,
HORIZON=20 -> CV satirlarinin lag'i 20 gun, TEST satirlarinin lag'i 30 gun
bayat. Ayni kolon iki farkli sey demekti; CV iyimserdi. Ambargo da yanlis
gerekcelenmisti (``max(horizon, 30)``): gecmis-hedef feature'lari zaten
horizon kadar kaydirilmis oldugu icin fazladan ambargo yalnizca her fold'un
egitimini bayatlatir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.features import add_lag_features
from gridup.pipeline import build_paired_history_features
from gridup.validation import forecast_geometry


def _seri(bas: str, gun: int) -> pd.Series:
    return pd.Series(pd.date_range(bas, periods=gun, freq="D"))


def test_bitisik_test_blogunda_ufuk_blok_boyudur() -> None:
    geo = forecast_geometry(_seri("2025-01-01", 100), _seri("2025-04-11", 20))
    assert geo.horizon_days == 20 and geo.gap_days == 0
    assert geo.embargo == pd.Timedelta(0) and not geo.interleaved


def test_bosluklu_testte_ufuk_boslugu_icerir_ambargo_bosluktur() -> None:
    train, test = _seri("2025-01-01", 100), _seri("2025-04-21", 20)
    geo = forecast_geometry(train, test)
    assert geo.gap_days == 10
    assert geo.horizon_days == 30  # 10 gun bosluk + 20 gun blok
    assert geo.embargo == pd.Timedelta(days=10)
    assert "bosluk 10 gun" in geo.summary()


def test_ic_ice_bolme_isaretlenir_ve_ambargo_sifirlanir() -> None:
    train, test = _seri("2025-01-01", 100), _seri("2025-02-01", 20)
    geo = forecast_geometry(train, test)
    assert geo.interleaved and geo.gap_days == 0
    assert "IC ICE" in geo.summary()
    with pytest.raises(ValueError, match="bos/NaT"):
        forecast_geometry(pd.Series([], dtype="datetime64[ns]"), test)


def test_geometrik_ufuk_lagi_gercekten_test_kadar_bayatlatir() -> None:
    """Denetimin kanit deneyi: bosluklu kurulumda CV ve test lag'i ayni olmali."""
    gunler_train = pd.date_range("2025-01-01", periods=100, freq="D")
    gunler_test = pd.date_range("2025-04-21", periods=20, freq="D")
    train = pd.DataFrame({"ilce": "a", "tarih": gunler_train, "y": np.arange(100.0)})
    test = pd.DataFrame({"ilce": "a", "tarih": gunler_test})

    geo = forecast_geometry(train["tarih"], test["tarih"])
    paket = build_paired_history_features(
        train, test, value_column="y", target_column="y", time_column="tarih",
        shifts=[geo.horizon_days], horizon=geo.horizon_days, group_columns=["ilce"],
    )  # fmt: skip
    kolon = f"y_shift{geo.horizon_days}"
    test_ozellik = paket.test
    assert test_ozellik is not None
    assert len(test_ozellik) == len(test), "kopru satirlari ciktiya sizmamali"
    # TARIH bazli esitlik: her test satirinin lag'i TAM ufuk kadar eski olmali.
    # Bosluk gunleri gecici kopru satirlariyla dolduruldugu icin satir
    # kaydirmasi = gun kaydirmasi (yoksa ilk test satiri 40 gun oncesini
    # goruyordu -- CV 30 gun ogrenip test 40 gun ile tahmin ediyordu).
    for konum in range(len(test_ozellik)):
        gun = gunler_test[konum]
        kaynak_gun = gun - pd.Timedelta(days=geo.horizon_days)
        beklenen = train.loc[train["tarih"] == kaynak_gun, "y"]
        deger = test_ozellik[kolon].iloc[konum]
        if beklenen.empty:  # kaynak gun bosluga/test'e dusuyorsa hedef yok
            assert pd.isna(deger)
        else:
            assert float(deger) == float(beklenen.iloc[0]), (gun, kaynak_gun)

    # Eski formul (blok boyu) 20 derdi ve test lag'i 10 gun daha taze GORUNURDU
    eski_ufuk = int((gunler_test.max() - gunler_test.min()).days) + 1
    assert eski_ufuk == 20 and geo.horizon_days == 30

    # CV tarafinda ayni kaydirma train satirlarinda da gecerli
    egitim = add_lag_features(
        train, "y", shifts=[geo.horizon_days], time_column="tarih",
        horizon=geo.horizon_days, group_columns=["ilce"],
    )  # fmt: skip
    assert egitim[kolon].iloc[50] == train["y"].iloc[50 - geo.horizon_days]
