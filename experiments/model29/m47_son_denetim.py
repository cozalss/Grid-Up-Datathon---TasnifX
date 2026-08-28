"""Gonderim oncesi son akil saglamasi: dagilimlar, uc degerler, bilinen trafolar."""

import os

import numpy as np
import pandas as pd
from m30_ozellik import KOK

te = pd.read_csv(
    os.path.join(KOK, "data/raw/test.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
tr = pd.read_csv(
    os.path.join(KOK, "data/raw/train.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
tr["ly"] = np.log1p(tr.tuketim)
A = pd.read_csv(os.path.join(KOK, "submissions/tuketim_v102_kappa_optimum.csv")).tuketim.values
B = pd.read_csv(os.path.join(KOK, "submissions/tuketim_m3_hl1_capali.csv")).tuketim.values
soguk = (~te.tanim.isin(set(tr.tanim))).values
print(
    f"{'dilim':12s} {'v102 log-ort':>13s} {'YENI log-ort':>13s} {'v102 std':>9s} {'YENI std':>9s}"
)
for ad, m in [("TUMU", np.ones(len(te), bool)), ("soguk", soguk), ("sicak", ~soguk)]:
    print(
        f"{ad:12s} {np.log1p(A[m]).mean():13.4f} {np.log1p(B[m]).mean():13.4f} "
        f"{np.log1p(A[m]).std():9.4f} {np.log1p(B[m]).std():9.4f}"
    )
print(
    f"\nuc degerler: v102 maks {A.max():,.0f}  YENI maks {B.max():,.0f}  (train maks {tr.tuketim.max():,.0f})"
)
print(
    f"sifira yakin: v102 <1kWh %{100 * (A < 1).mean():.2f}   YENI <1kWh %{100 * (B < 1).mean():.2f}"
)
print(f"train'de sifir orani %{100 * (tr.tuketim == 0).mean():.2f}")
# sicak trafolarda: tahminin kendi son-28-gun seviyesine uzakligi
son28 = tr[tr.tarih > pd.Timestamp("2026-03-31") - pd.Timedelta(days=28)].groupby("tanim").ly.mean()
k = te.tanim.map(son28)
m = (~soguk) & k.notna().values
print("\nsicak trafolarda tahmin - son28 seviyesi (log):")
print(f"  v102 ort {(np.log1p(A[m]) - k[m]).mean():+.4f} std {(np.log1p(A[m]) - k[m]).std():.4f}")
print(f"  YENI ort {(np.log1p(B[m]) - k[m]).mean():+.4f} std {(np.log1p(B[m]) - k[m]).std():.4f}")
# 2025 gercegi: ayni olcum
b25 = (
    tr[(tr.tarih > pd.Timestamp("2025-03-31") - pd.Timedelta(days=28)) & (tr.tarih <= "2025-03-31")]
    .groupby("tanim")
    .ly.mean()
)
h = tr[(tr.tarih > "2025-03-31") & (tr.tarih <= "2025-07-31")].copy()
h["k"] = h.tanim.map(b25)
h = h[h.k.notna()]
print(f"  2025 GERCEK ort {(h.ly - h.k).mean():+.4f} std {(h.ly - h.k).std():.4f}")
# tam sifir train gecmisi olan trafolar
olu = set(tr.groupby("tanim").tuketim.max().pipe(lambda s: s[s < 1]).index)
mo = te.tanim.isin(olu).values
print(f"\ntrain'de HIC tuketmemis trafolar: {mo.sum():,} test satiri")
print(f"  v102 log-ort {np.log1p(A[mo]).mean():.4f}   YENI log-ort {np.log1p(B[mo]).mean():.4f}")
