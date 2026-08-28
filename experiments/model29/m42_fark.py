"""Yeni model ile v102 arasindaki YON: enerji Q, kapsam, ve harman cebiri."""

import json
import os

import numpy as np
import pandas as pd
from m30_ozellik import KOK

te = pd.read_csv(
    os.path.join(KOK, "data/raw/test.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
tr = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"), dtype={"tanim": str})
a = np.log1p(
    pd.read_csv(os.path.join(KOK, "submissions/tuketim_v102_kappa_optimum.csv")).tuketim.values
)
b = np.log1p(
    pd.read_csv(os.path.join(KOK, "submissions/tuketim_m1_ileri_huber.csv")).tuketim.values
)
d = b - a
N = len(d)
sog = (~te.tanim.isin(set(tr.tanim))).values
Q = float((d**2).mean())
print(
    f"N {N:,}   Q = ||yeni - v102||^2/N = {Q:.6f}   |d| ort {np.abs(d).mean():.4f}  medyan {np.median(np.abs(d)):.4f}"
)
print(f"kapsam (|d|>1e-9): %{100 * (np.abs(d) > 1e-9).mean():.2f}")
print(
    f"  soguk satirlarda Q {float((d[sog] ** 2).mean()):.4f} (kutle %{100 * (d[sog] ** 2).sum() / (d**2).sum():.1f}, satir %{100 * sog.mean():.1f})"
)
print(f"  sicak satirlarda Q {float((d[~sog] ** 2).mean()):.4f}")
print(f"seviye: v102 log-ort {a.mean():.4f}  yeni {b.mean():.4f}  fark {b.mean() - a.mean():+.4f}")
print(f"yayilim: v102 std {a.std():.4f}  yeni std {b.std():.4f}")
print(f"korelasyon(v102, yeni) = {np.corrcoef(a, b)[0, 1]:.5f}")
print()
print("=== HARMAN CEBIRI: yeni'nin skoru S olcuulunce ===")
m0 = 1.00553**2
print(f"  v102 MSE m0 = {m0:.6f}")
print("  yeni MSE  m1 = S^2 ;  L = (m0 + Q - m1)/2 ;  kappa* = L/Q ;  optimum MSE = m0 - L^2/Q")
print(f"  {'S (yeni skor)':>14s} {'L':>10s} {'kappa*':>8s} {'optimum RMSLE':>14s}")
for S in [0.96, 0.98, 0.99, 1.00, 1.00553, 1.01, 1.02, 1.05, 1.10]:
    L = (m0 + Q - S * S) / 2
    k = L / Q
    opt = np.sqrt(max(m0 - L * L / Q, 0))
    print(f"  {S:14.5f} {L:+10.5f} {k:+8.4f} {opt:14.5f}")
json.dump(
    dict(Q=Q, m0=m0, kapsam=float((np.abs(d) > 1e-9).mean())), open("m42_fark.json", "w"), indent=1
)
