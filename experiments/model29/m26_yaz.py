"""EKSEN 2d: gercek hedef penceresi icin (Nis-Tem) mevsim carpani; grup bazli mi global mi?"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *

tr = yukle()
tr["ly"] = np.log1p(tr.tuketim)
tr["ym"] = tr.tarih.dt.to_period("M")
# 2025 Mart sonu -> Nis-Tem 2025 gecisi: gercek test ile ayni takvim yapisi
MART = (tr.tarih >= pd.Timestamp("2025-03-25")) & (tr.tarih <= pd.Timestamp("2025-03-31"))
sub = tr[tr.groupby("tanim").tuketim.transform("max") >= 1]
b = sub[MART.reindex(sub.index, fill_value=False)].groupby("tanim").ly.mean()
res = {}
print("HEDEF AYLARIN 2025 MART-SONU SEVIYESINE GORE KAYMASI (trafo bazli, sonra ortalama)")
for m in [4, 5, 6, 7]:
    t = (
        sub[sub.tarih.dt.month == m][sub.tarih.dt.year == 2025].groupby("tanim").ly.mean()
        if False
        else sub[(sub.tarih.dt.month == m) & (sub.tarih.dt.year == 2025)].groupby("tanim").ly.mean()
    )
    d = (t - b).dropna()
    res[f"ay{m}"] = {
        "n": int(len(d)),
        "ort": float(d.mean()),
        "medyan": float(d.median()),
        "std": float(d.std()),
    }
    print(
        f"  {m:02d}: n={len(d):5d} ort {d.mean():+.4f} medyan {d.median():+.4f} std {d.std():.3f}"
    )
tum = []
for m in [4, 5, 6, 7]:
    t = sub[(sub.tarih.dt.month == m) & (sub.tarih.dt.year == 2025)].groupby("tanim").ly.mean()
    tum.append((t - b).dropna())
ort = pd.concat(tum, axis=1).mean(axis=1)
print(f"  Nis-Tem ortalama kaymasi: {ort.mean():+.4f} (trafo std {ort.std():.3f}, n={len(ort)})")
res["nistem_ort"] = {"ort": float(ort.mean()), "trafo_std": float(ort.std()), "n": int(len(ort))}

# --- grup bazli mi? ilce / guc kirilimi ve YARI-ORNEK guvenilirligi ---
info = sub.groupby("tanim").agg(ilce=("ilce", "first"), guc=("guc", "first"), il=("il", "first"))
df = pd.DataFrame({"d": ort}).join(info).dropna()
rng = np.random.default_rng(0)
df["yari"] = rng.integers(0, 2, len(df))
print("\nGRUP KIRILIMI (Nis-Tem eksi Mart-sonu):")
for anah in ["ilce", "il"]:
    gm = df.groupby(anah).d.agg(["mean", "size"])
    gm = gm[gm["size"] >= 15]
    a = df[df.yari == 0].groupby(anah).d.mean()
    bq = df[df.yari == 1].groupby(anah).d.mean()
    ort2 = pd.concat([a, bq], axis=1).dropna()
    print(
        f"  {anah}: grup sayisi(n>=15) {len(gm)} grup-ort std {gm['mean'].std():.3f} | yari-ornek korelasyon {ort2.iloc[:, 0].corr(ort2.iloc[:, 1]):.3f}"
    )
    res[f"grup_{anah}"] = {
        "n_grup": int(len(gm)),
        "std": float(gm["mean"].std()),
        "yari_kor": float(ort2.iloc[:, 0].corr(ort2.iloc[:, 1])),
    }
gk = pd.cut(df.guc, [0, 100, 250, 400, 630, 1000, 1e9])
print("  guc kovalari:")
for i, r in df.groupby(gk, observed=True).d.agg(["mean", "size"]).iterrows():
    print(f"    {i} n={int(r['size'])} ort {r['mean']:+.3f}")
# trafo bazli kayma tahmin edilebilir mi? (yari-ornek degil, ayni trafo icin 2. bir yil yok)
print(
    f"\n  trafo bazli kayma std {ort.std():.3f} -> global sabit kullanmanin kalan hatasi bu kadar"
)
json_yaz("eksen2d_yaz", res)
