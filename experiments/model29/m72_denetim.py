"""m4 (havali) son denetim: m3 ve v102 ile karsilastir."""

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
S = lambda f: np.log1p(pd.read_csv(os.path.join(KOK, "submissions", f)).tuketim.values)
A = S("tuketim_v102_kappa_optimum.csv")
B = S("tuketim_m3_hl1_capali.csv")
C = S("tuketim_m4_hava_capali.csv")
sog = (~te.tanim.isin(set(tr.tanim))).values
print(f"{'':10s} {'log-ort':>9s} {'std':>7s} {'soguk std':>10s} {'sicak std':>10s}")
for ad, p in [("v102", A), ("m3", B), ("m4 HAVA", C)]:
    print(f"{ad:10s} {p.mean():9.4f} {p.std():7.4f} {p[sog].std():10.4f} {p[~sog].std():10.4f}")
print(
    f"\nQ(m4,v102)={((C - A) ** 2).mean():.6f}  Q(m3,v102)={((B - A) ** 2).mean():.6f}  Q(m4,m3)={((C - B) ** 2).mean():.6f}"
)
print(f"korelasyon m3-m4 {np.corrcoef(B, C)[0, 1]:.5f}")
# mevsim denetimi
son7 = tr[tr.tarih > pd.Timestamp("2026-03-31") - pd.Timedelta(days=7)].groupby("tanim").ly.mean()
k = te.tanim.map(son7)
m = (~sog) & k.notna().values
b25 = (
    tr[(tr.tarih > pd.Timestamp("2025-03-31") - pd.Timedelta(days=7)) & (tr.tarih <= "2025-03-31")]
    .groupby("tanim")
    .ly.mean()
)
h = tr[(tr.tarih > "2025-03-31") & (tr.tarih <= "2025-07-31")].copy()
h["k"] = h.tanim.map(b25)
h = h[h.k.notna()]
print("\nson7'ye gore AY KAYMASI (sicak trafolar):")
print(f"  {'ay':>3s} {'v102':>8s} {'m3':>8s} {'m4 HAVA':>8s} {'2025 GERCEK':>12s}")
for ay in (4, 5, 6, 7):
    s = m & (te.tarih.dt.month == ay).values
    g = h[h.tarih.dt.month == ay]
    print(
        f"  {ay:3d} {(A[s] - k[s]).mean():+8.4f} {(B[s] - k[s]).mean():+8.4f} {(C[s] - k[s]).mean():+8.4f} {(g.ly - g.k).mean():+12.4f}"
    )
print(f"\nuc degerler: v102 maks {np.expm1(A).max():,.0f}  m4 maks {np.expm1(C).max():,.0f}")
print(
    f"<1kWh: v102 %{100 * (np.expm1(A) < 1).mean():.2f}  m4 %{100 * (np.expm1(C) < 1).mean():.2f}  (train sifir %{100 * (tr.tuketim == 0).mean():.2f})"
)
