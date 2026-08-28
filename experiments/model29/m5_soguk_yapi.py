"""Soguk trafo seviyesinin YAPISI: bimodal mi? varlik deseni tahmin ediyor mu?"""

import numpy as np
import pandas as pd
from m1_geriteste import kes, yukle

tr = yukle()
kesim = "2025-10-31"
gec, hed = kes(tr, kesim)
hed = hed.copy()
hed["ly"] = np.log1p(hed.tuketim)
son = pd.Timestamp(kesim) + pd.DateOffset(months=4)

s = (
    hed[hed.soguk]
    .groupby("tanim")
    .agg(
        y=("ly", "mean"),
        n=("ly", "size"),
        guc=("guc", "first"),
        ilce=("ilce", "first"),
        bolge=("bolge", "first"),
        ilk=("tarih", "min"),
        sonn=("tarih", "max"),
        sifir=("tuketim", lambda v: (v == 0).mean()),
    )
)
s["gun_araligi"] = (s.sonn - s.ilk).dt.days + 1
s["yogunluk"] = s.n / s.gun_araligi
s["ilk_gun"] = (s.ilk - pd.Timestamp(kesim)).dt.days
s["kuyruk"] = (son - s.sonn).dt.days

print("SOGUK trafo seviye (y) dagilimi:")
print(s.y.describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_string())
print("\nhistogram (log1p tuketim ortalamasi):")
h, e = np.histogram(s.y, bins=20)
for i in range(20):
    print(f"  {e[i]:6.2f}-{e[i + 1]:6.2f}: {'#' * int(60 * h[i] / h.max())} {h[i]}")

print("\n--- varlik deseni sinyalleri (seviye ile korelasyon, Pearson) ---")
for c in ["n", "gun_araligi", "yogunluk", "ilk_gun", "kuyruk"]:
    print(f"  {c:14s} r = {np.corrcoef(s[c].astype(float), s.y)[0, 1]:+.4f}")
print(f"  log(guc)       r = {np.corrcoef(np.log(s.guc.clip(lower=1)), s.y)[0, 1]:+.4f}")

print("\n--- 'olu' (cok dusuk) siniflandirmasi ---")
for esik in [0.5, 2.0, 4.0]:
    m = s.y < esik
    print(
        f"  y < {esik}: {m.sum():4d} trafo (%{100 * m.mean():.1f}), satir payi %{100 * s.n[m].sum() / s.n.sum():.1f}"
    )
print("\n  n (gun sayisi) esikli olu orani:")
for lo, hi in [(1, 10), (11, 30), (31, 60), (61, 90), (91, 123)]:
    m = (s.n >= lo) & (s.n <= hi)
    if m.sum():
        print(
            f"   n {lo:3d}-{hi:3d}: {m.sum():4d} trafo, ort y {s.y[m].mean():.2f}, y<2 orani %{100 * (s.y[m] < 2).mean():.1f}"
        )
