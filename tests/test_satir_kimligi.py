"""UCTAN UCA SATIR KIMLIGI: girdi karistirilirsa sonuc DEGISMEMELI.

Bu dosyadaki tek test, bu depoda tekrar tekrar ortaya cikan hata sinifini
kokunden yakalar: **sessiz satir kaymasi.**

Bugune kadar bulunan ornekler -- hepsi hatasiz, ayni satir sayisiyla,
tamamen sessizce yanlis sonuc uretiyordu:

  * add_rolling_features / add_expanding_features  -> pencere BASKA gruptan
  * add_solar_features (tekrarli index)            -> ilceler TAKAS
  * add_weather_accumulators                       -> satir SIRASI degisti

Ortak noktalari: hicbiri bir birim testinde gorunmuyordu, cunku birim
testleri tek bir fonksiyonu tek bir sirali girdiyle sinar.

Buradaki sinav farkli: **ayni veriyi iki farkli SIRADA** feature zincirinden
gecirip, varlik-zaman anahtarina gore hizaladiktan sonra sonuclarin
BIREBIR ayni olmasini bekliyoruz. Bir fonksiyon satir kimligini kaybederse
bu test kirilir -- hangi fonksiyon oldugunu bilmeye gerek kalmadan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.features.spatial import add_neighbour_feature_mean
from gridup.features.temporal import (
    add_calendar_features,
    add_expanding_features,
    add_lag_features,
    add_rolling_features,
)
from gridup.features.weather import add_physical_derivatives, add_weather_accumulators

#: Gorunum sirasi KASITLI olarak alfabetik degil -- grup kaymasi ancak
#: boyle yakalanir.
YERLER = ["zeytinburnu", "aliaga", "menemen", "bornova"]

ANAHTAR = ["yer", "tarih"]


def _panel(n_gun: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    kayit = []
    for i, yer in enumerate(YERLER):
        for gun, tarih in enumerate(pd.date_range("2026-02-10", periods=n_gun)):
            kayit.append(
                {
                    "yer": yer,
                    "tarih": tarih,
                    # Degerler yerler arasi 1000 kat ayrisik: kayma olursa
                    # kucuk sayisal fark degil, aninda gorunur bir fark cikar.
                    "hedef": float((i + 1) * 1000 + gun),
                    "sicaklik_ort": float((i + 1) * 100 + gun),
                    "yagis_toplam": float(rng.integers(0, 20)),
                    "ruzgar_max": float(rng.integers(0, 60)),
                }
            )
    return pd.DataFrame(kayit)


def _komsuluk() -> pd.DataFrame:
    satir = [
        {"yer": a, "komsu": b, "mesafe_km": 40.0}
        for a in YERLER
        for b in YERLER
        if a != b
    ]
    return pd.DataFrame(satir)


def _zincir(frame: pd.DataFrame) -> pd.DataFrame:
    """Gercek bir feature boru hatti -- yarisma gunu kurulacak sirayla."""
    cikti = add_calendar_features(frame, "tarih")
    cikti = add_lag_features(
        cikti, "hedef", [1, 7], time_column="tarih", group_columns=["yer"]
    )
    cikti = add_rolling_features(
        cikti, "hedef", [3, 7], time_column="tarih", group_columns=["yer"]
    )
    cikti = add_expanding_features(
        cikti, "hedef", time_column="tarih", group_columns=["yer"]
    )
    cikti = add_weather_accumulators(
        cikti, group_columns=["yer"], time_column="tarih", value_columns=["yagis_toplam"]
    )
    cikti = add_physical_derivatives(
        cikti, group_columns=["yer"], time_column="tarih",
        temperature_max=None, temperature_min=None, gust_max=None,
    )
    return add_neighbour_feature_mean(
        cikti, _komsuluk(), key_column="yer", time_column="tarih",
        value_columns=["sicaklik_ort"],
    )


def _anahtara_gore_hizala(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(ANAHTAR).reset_index(drop=True)


@pytest.mark.parametrize("tohum", [0, 1, 2, 3, 4])
def test_girdi_sirasi_ciktiyi_degistirmiyor(tohum: int):
    """Ayni veri, farkli satir sirasi -> anahtara gore hizalandiginda AYNI cikti.

    Bir feature fonksiyonu satir kimligini kaybederse (grup kaydirma, index
    hizalamasi, sessiz yeniden siralama) bu test kirilir.
    """
    duz = _panel()
    karisik = duz.sample(frac=1.0, random_state=tohum).reset_index(drop=True)

    a = _anahtara_gore_hizala(_zincir(duz))
    b = _anahtara_gore_hizala(_zincir(karisik))

    assert list(a.columns) == list(b.columns)
    pd.testing.assert_frame_equal(
        a, b, check_like=False,
        obj=f"tohum={tohum}: girdi sirasi ciktiyi DEGISTIRDI -- satir kimligi kayboldu",
    )


def test_zincir_satir_sayisini_ve_anahtari_koruyor():
    duz = _panel()
    cikti = _zincir(duz)

    assert len(cikti) == len(duz)
    # Anahtar cifti birebir ayni kume olmali.
    a = set(map(tuple, duz[ANAHTAR].astype(str).to_numpy()))
    b = set(map(tuple, cikti[ANAHTAR].astype(str).to_numpy()))
    assert a == b


def test_zincir_dogru_gruptan_hesapliyor():
    """Bagimsiz pandas gercegiyle karsilastir -- zincir ne uretmeli?"""
    duz = _panel()
    cikti = _anahtara_gore_hizala(_zincir(duz))

    for yer in YERLER:
        grup = duz[duz.yer == yer].sort_values("tarih")
        beklenen = grup["hedef"].shift(1).rolling(3, min_periods=1).mean().to_numpy()
        gercek = cikti[cikti.yer == yer].sort_values("tarih")["hedef_kayan3_mean"].to_numpy()
        assert np.allclose(beklenen, gercek, equal_nan=True), (
            f"'{yer}' kayan ortalamasi BASKA gruptan geliyor"
        )
