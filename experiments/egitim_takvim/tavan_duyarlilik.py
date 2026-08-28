"""Tavan hesabinin taban secimine duyarliligi.

15 gunluk merkezli pencere bayram sicramalarini kismen SOGURUR (4 gunluk
Kurban 15 gunluk pencerede ~%27 zayiflar). Taban tatil gunlerini DISLAYARAK
yeniden kurulur ve tavan yeniden olculur -- bu, sogurmeyi kaldirir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "src"))
ONB = KOK / "data" / "interim" / "deney"

df = pd.read_parquet(ONB / "egitim.parquet", columns=["tarih", "tanim_num", "tuketim"])
df["tarih"] = pd.to_datetime(df["tarih"])
df["y"] = np.log1p(df["tuketim"].clip(lower=0.0))
df["r"] = df["y"] - df.groupby("tanim_num")["y"].transform("mean")
g = df.groupby("tarih").agg(d_ham=("r", "mean"), n=("r", "size")).sort_index()

import holidays  # noqa: E402

tr = holidays.country_holidays("TR", years=[2025, 2026])
tatil_mi = pd.Series([t.date() in tr for t in g.index], index=g.index)


def tasarim(idx):
    X = pd.get_dummies(idx.dayofweek, prefix="hg").astype(float)
    X.index = idx
    adlar = pd.Series([str(tr.get(t.date(), "") or "") for t in idx], index=idx).str.lower()
    X["tatil"] = adlar.ne("").astype(float).to_numpy()
    X["kurban"] = adlar.str.contains("kurban").astype(float).to_numpy()
    X["ramazan_b"] = adlar.str.contains("ramazan").astype(float).to_numpy()
    return X.to_numpy(dtype="float64")


def olc(alt, etiket):
    idx = pd.DatetimeIndex(alt.index)
    X, w, d = tasarim(idx), alt["n"].to_numpy(float), alt["d"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], d * np.sqrt(w), rcond=None)
    fit = X @ beta
    ac = float(np.average(fit**2, weights=w))
    top = float(np.average(d**2, weights=w))
    print(
        f"  {etiket:34} gun etkisi std={np.sqrt(top):.4f}  takvim std={np.sqrt(ac):.4f}  "
        f"TAVAN dMSE={-ac:.5f}  R2={ac / top:.3f}"
    )


for ad, pencere, tatil_disla in (
    ("15 gun, tatiller dahil", 15, False),
    ("15 gun, TABAN tatilsiz", 15, True),
    ("29 gun, TABAN tatilsiz", 29, True),
    ("45 gun, TABAN tatilsiz", 45, True),
):
    s = g["d_ham"].where(~tatil_mi) if tatil_disla else g["d_ham"]
    taban = s.rolling(pencere, center=True, min_periods=5).mean()
    gg = g.assign(d=g["d_ham"] - taban).dropna(subset=["d"])
    print(f"\n[{ad}]")
    olc(gg, "TUM EGITIM")
    olc(gg[(gg.index >= "2025-04-01") & (gg.index <= "2025-07-31")], "yaz25 (TESTIN IKIZI)")
    olc(gg[(gg.index >= "2025-12-01") & (gg.index <= "2026-03-31")], "kis26")

print("\nKAPI ESIGI: kabul icin gereken toplam test dMSE <= -0,00200")
