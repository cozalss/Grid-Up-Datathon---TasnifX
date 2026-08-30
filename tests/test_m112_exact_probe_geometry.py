from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

MODEL29 = Path(__file__).resolve().parents[1] / "experiments" / "model29"
sys.path.insert(0, str(MODEL29))

from m112_kalibre import (  # noqa: E402
    gonderim_olcumlerini_ekle,
    gonderim_olcumu,
    onsele_dayali_duzeltme,
)


def test_gonderim_olcumu_leaderboard_skorunu_birebir_yeniden_kurar() -> None:
    taban = np.array([1.0, 1.4, 2.1, 0.3], dtype=np.float64)
    tahmin = np.array([1.2, 1.1, 2.4, 0.4], dtype=np.float64)
    m0 = 1.37
    skor = 1.12345

    yon, ic_carpim = gonderim_olcumu(taban, tahmin, skor, m0=m0)

    yeniden_kurulan_mse = m0 + np.mean(yon * yon) - 2.0 * ic_carpim
    assert np.sqrt(yeniden_kurulan_mse) == pytest.approx(skor, abs=1e-12)


def test_gonderim_olcumu_satir_sayisi_uyusmazligini_reddeder() -> None:
    with pytest.raises(ValueError, match="satir sayisi"):
        gonderim_olcumu(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            1.0,
            m0=1.0,
        )


def test_kayitli_gonderimler_soyut_yon_yerine_dosyanin_kendisini_kullanir() -> None:
    taban = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    gonderilen = np.array([0.5, 0.1, 0.9], dtype=np.float64)
    yonler: list[np.ndarray] = []
    ic_carpimlar: list[float] = []
    olcumler = [{"aday": "seviye", "dosya": "probe.csv", "skor": 0.91}]

    gonderim_olcumlerini_ekle(
        taban,
        yonler,
        ic_carpimlar,
        olcumler,
        okuyucu=lambda _: gonderilen,
        m0=1.2,
    )

    assert len(yonler) == 1
    assert yonler[0] == pytest.approx(gonderilen - taban)
    yeniden_kurulan = 1.2 + np.mean(yonler[0] ** 2) - 2.0 * ic_carpimlar[0]
    assert np.sqrt(yeniden_kurulan) == pytest.approx(0.91, abs=1e-12)


def test_onsel_duzeltmesi_yeni_yonleri_sirayla_diklestirir() -> None:
    bilinen = np.array([[1.0], [-1.0], [1.0], [-1.0]])
    gram = bilinen.T @ bilinen / len(bilinen)
    adaylar = {
        "ay": np.array([1.0, 1.0, -1.0, -1.0]),
        "haftasonu": np.array([1.0, -1.0, -1.0, 1.0]),
    }

    duzeltme, bilgi = onsele_dayali_duzeltme(
        adaylar,
        bilinen,
        gram,
        [("ay", 0.03), ("haftasonu", -0.02)],
        len(bilinen),
    )

    assert [satir["aday"] for satir in bilgi] == ["ay", "haftasonu"]
    assert np.mean(duzeltme * bilinen[:, 0]) == pytest.approx(0.0, abs=1e-12)
    assert np.mean(duzeltme**2) == pytest.approx(0.03**2 + 0.02**2, abs=1e-12)
