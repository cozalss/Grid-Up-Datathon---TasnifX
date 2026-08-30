from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

MODEL29 = Path(__file__).resolve().parents[1] / "experiments" / "model29"
sys.path.insert(0, str(MODEL29))

from m112_kalibre import gonderim_olcumu  # noqa: E402


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
