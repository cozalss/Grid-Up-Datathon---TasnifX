"""SICAK ADAYLAR -- 2. tur: TASIMAYA dayanmayan tek-parametreli adaylar.

1. tur (``adaylar.py``) blok-disi ogrenilen her YAPI'nin TERS tasindigini
gosterdi. Bu tur yalnizca tek kuresel parametreli, tasima varsayimi
gerektirmeyen adaylari olcer: harman agirligi, harman uzayi, kirpma
tabani/tavani, ve tohum sayisi.

Kapi ayni: uc blokta ayni isaret + test dMSE <= -0,002.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ortak import (  # noqa: E402
    AGIRLIK,
    BLOKLAR,
    C_GUN,
    KUYRUK_DELTA,
    SICAK_PAY,
    bloklari_kur,
    gun_etkisi,
    kuresel_delta,
    mse,
    tablo_yaz,
)


def zincir(b, log_tahmin: np.ndarray, *, c=C_GUN, ku=KUYRUK_DELTA) -> np.ndarray:
    """v83 sicak zinciri + blok kuresel seviye kalibrasyonu."""
    r = log_tahmin - b.lgc
    be = gun_etkisi(b.cerceve["tanim"].to_numpy(), b.cerceve["tarih"].to_numpy(), r)
    e = pd.Series(b.cerceve["tarih"].to_numpy()).map(be).to_numpy(dtype="float64")
    r = r + (c - 1.0) * (e - e.mean())
    r = r + ku * b.cerceve["kuyruk"].to_numpy(dtype="float64")
    return r + kuresel_delta(b, r)


def harman(b, agirlik: dict[str, float]) -> np.ndarray:
    pay = sum(agirlik.values())
    return sum(w * b.ham[a] for a, w in agirlik.items()) / pay


def satir(bl, ad: str, taban_mse: dict[str, float], yeni: dict[str, float]) -> dict:
    s: dict = {"aday": ad}
    tn = td = 0.0
    for k in BLOKLAR:
        d = yeni[k] - taban_mse[k]
        s[k] = d
        tn += bl[k].n
        td += d * bl[k].n
    s["GENEL"] = td / tn
    s["testMSE"] = s["GENEL"] * SICAK_PAY
    s["ayni_isaret"] = all(s[k] < 0 for k in BLOKLAR) or all(s[k] > 0 for k in BLOKLAR)
    return s


def main() -> int:
    bl = bloklari_kur()
    taban = {k: zincir(bl[k], harman(bl[k], AGIRLIK)) for k in BLOKLAR}
    tm = {k: mse(bl[k], taban[k]) for k in BLOKLAR}
    print("TABAN sicak MSE:", {k: round(v, 5) for k, v in tm.items()})
    print("TABAN sicak RMSLE:", {k: round(np.sqrt(v), 5) for k, v in tm.items()})

    satirlar = []

    # --- B1: aile agirliklari (tek aile + varyasyonlar) --------------------
    izgara = {
        "B1 saf cat": {"cat": 1.0},
        "B1 saf xgb": {"xgb": 1.0},
        "B1 saf lgbm": {"lgbm": 1.0},
        "B1 saf sinir_agi": {"sinir_agi": 1.0},
        "B2 ag YOK (3/1/1/0)": {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 0.0},
        "B2 ag 0,7 (3/1/1/0,7)": {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 0.7},
        "B2 ag 2,1 (3/1/1/2,1)": {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 2.1},
        "B2 ag 3,0 (3/1/1/3)": {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 3.0},
        "B3 esit (1/1/1/1)": {"cat": 1.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 1.0},
        "B3 xgb agir (2/3/1/1,4)": {"cat": 2.0, "xgb": 3.0, "lgbm": 1.0, "sinir_agi": 1.4},
        "B3 cat agir (5/1/1/1,4)": {"cat": 5.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 1.4},
        "B3 (2/2/1/1,4)": {"cat": 2.0, "xgb": 2.0, "lgbm": 1.0, "sinir_agi": 1.4},
        "B3 (3/2/2/1,4)": {"cat": 3.0, "xgb": 2.0, "lgbm": 2.0, "sinir_agi": 1.4},
    }
    for ad, ag in izgara.items():
        ag = {a: w for a, w in ag.items() if w > 0}
        yeni = {k: mse(bl[k], zincir(bl[k], harman(bl[k], ag))) for k in BLOKLAR}
        satirlar.append(satir(bl, ad, tm, yeni))

    # --- B4: harman UZAYI -- ham uzayda ortalama ---------------------------
    def ham_uzay(b):
        pay = sum(AGIRLIK.values())
        t = sum(w * np.expm1(b.ham[a]) for a, w in AGIRLIK.items()) / pay
        return np.log1p(np.clip(t, 0.0, None))

    yeni = {k: mse(bl[k], zincir(bl[k], ham_uzay(bl[k]))) for k in BLOKLAR}
    satirlar.append(satir(bl, "B4 ham uzayda harman", tm, yeni))

    # --- B5: KIRPMA tabani (tahmin alt siniri, kWh) ------------------------
    for taban_kwh in (0.5, 1.0, 2.0, 5.0):
        yeni = {}
        for k in BLOKLAR:
            b = bl[k]
            t = np.clip(np.expm1(taban[k] + b.lgc), taban_kwh, None)
            e = b.lgy - np.log1p(t)
            yeni[k] = float((e * e).mean())
        satirlar.append(satir(bl, f"B5 kirpma tabani {taban_kwh} kWh", tm, yeni))

    # --- B6: KIRPMA tavani (kapasite katı) ---------------------------------
    for kat in (24.0, 18.0, 12.0):
        yeni = {}
        for k in BLOKLAR:
            b = bl[k]
            t = np.minimum(np.expm1(taban[k] + b.lgc), kat * b.cerceve["guc"].to_numpy())
            e = b.lgy - np.log1p(np.clip(t, 0.0, None))
            yeni[k] = float((e * e).mean())
        satirlar.append(satir(bl, f"B6 kirpma tavani {kat:.0f}x guc", tm, yeni))

    # --- B7: TOHUM SAYISI (varyans azaltma tavani) -------------------------
    for n in (1, 2):
        yeni = {}
        for k in BLOKLAR:
            b = bl[k]
            yeni[k] = mse(bl[k], zincir(b, np.mean(b.tohum_harman[:n], axis=0)))
        satirlar.append(satir(bl, f"B7 tohum sayisi {n} (taban 3)", tm, yeni))

    tablo_yaz(satirlar)
    yol = Path(__file__).resolve().parent / "adaylar2.jsonl"
    with yol.open("w", encoding="utf-8") as f:
        for s in satirlar:
            f.write(pd.Series(s).to_json() + "\n")
    print(f"\nyazildi: {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
