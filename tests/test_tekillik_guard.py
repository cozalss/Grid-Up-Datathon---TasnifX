"""(grup, gun) tekillik guard'i -- ufuk duvarinin SESSIZ delinmesine karsi.

2026-08-18 denetimi olctu: olay-duzeyi bir kayitta (ilce basina gunde 3
satir) ``add_lag_features(..., horizon=7)`` satir kaydirdigi icin "7 satir
once" 2-3 gun oncesine denk geliyordu -- CV 7 gunluk ufuk sanirken model
neredeyse dunku hedefi goruyordu. Guard ``spatial.py``de vardi, temporal
ailenin tamamina tasindi; burada davranis olarak kanitlanir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.features.temporal import (
    add_days_since_event_features,
    add_event_decay_features,
    add_expanding_features,
    add_lag_features,
    add_rolling_features,
)
from gridup.pipeline import build_paired_history_features


def _olay_kaydi(olay_per_gun: int) -> pd.DataFrame:
    gunler = pd.date_range("2026-01-01", periods=40, freq="D")
    satirlar = [
        {"ilce": "bornova", "tarih": g, "y": float(i + 1)}
        for i, g in enumerate(gunler)
        for _ in range(olay_per_gun)
    ]
    return pd.DataFrame(satirlar)


def test_olay_duzeyi_kayit_reddedilir_gunluk_panel_gecer() -> None:
    tekrarli = _olay_kaydi(3)
    gunluk = _olay_kaydi(1)
    with pytest.raises(ValueError, match="tekrarliyor"):
        add_lag_features(
            tekrarli, "y", shifts=[7], time_column="tarih", horizon=7, group_columns=["ilce"]
        )
    sonuc = add_lag_features(
        gunluk, "y", shifts=[7], time_column="tarih", horizon=7, group_columns=["ilce"]
    )
    # Gunluk panelde 7 satir = 7 gun: feature tam 7 gun oncesinin degeri
    assert np.isclose(sonuc.loc[10, "y_shift7"], gunluk.loc[3, "y"])


@pytest.mark.parametrize(
    "fonksiyon, ek",
    [
        (add_rolling_features, {"windows": [3]}),
        (add_expanding_features, {}),
        (add_event_decay_features, {"half_lives": [3]}),
        (add_days_since_event_features, {}),
    ],
)
def test_tum_gecmis_hedef_aileleri_tekillik_ister(fonksiyon, ek) -> None:
    tekrarli = _olay_kaydi(2)
    with pytest.raises(ValueError, match="tekrarliyor"):
        fonksiyon(tekrarli, "y", time_column="tarih", horizon=3, group_columns=["ilce"], **ek)


def test_build_paired_history_features_tekrarli_traini_reddeder() -> None:
    train = _olay_kaydi(2)
    test = pd.DataFrame(
        {"ilce": ["bornova"] * 3, "tarih": pd.date_range("2026-02-20", periods=3, freq="D")}
    )
    with pytest.raises(ValueError, match="tekrarliyor"):
        build_paired_history_features(
            train, test, value_column="y", target_column="y", time_column="tarih",
            shifts=[7], horizon=7, group_columns=["ilce"],
        )  # fmt: skip
