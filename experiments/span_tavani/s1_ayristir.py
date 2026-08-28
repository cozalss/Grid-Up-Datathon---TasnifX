"""d108 ve dy1'i, olculmus 23 gonderimin span'ini kullanarak AYRISTIRMAYI dene.

Fikir: r = log1p(p0) - log1p(t) bilinmiyor, ama her olculmus yon d_i icin
    L_i = -<r,d_i>/n = (m0 + G_ii - m_i)/2
tam olarak biliniyor (yalniz 5 hane yuvarlamasi belirsizligi).
Yeni bir d yonu span{d_i} icindeyse L(d) = sum c_j L_j ile COZULUR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SKOR = {
    # "gun1_baseline.csv": 1.22670,  # diskteki dosya 40 satirlik baska bir artefakt
    "tuketim_v2.csv": 1.16143,
    "tuketim_v7.csv": 1.16922,
    "tuketim_v15.csv": 1.03910,
    "tuketim_v16.csv": 1.06605,
    "tuketim_v18.csv": 1.03370,
    "tuketim_v25_hedge.csv": 1.04820,
    "tuketim_v27_v18hedge.csv": 1.03362,
    "tuketim_v30_buzme.csv": 1.02639,
    "tuketim_v44_v27yeni.csv": 1.03053,
    "tuketim_v46_gun.csv": 1.02448,
    "tuketim_v47_eskison.csv": 1.01750,
    "tuketim_v50_nihai30.csv": 1.01686,
    "tuketim_v55_gunolcek.csv": 1.01591,
    "tuketim_v67_c1335_olay.csv": 1.01548,
    "tuketim_v73_soguk_gun160.csv": 1.01538,
    "tuketim_v79_S3.csv": 1.01556,
    "tuketim_v80_optimum.csv": 1.01341,
    "tuketim_v81_sicak08.csv": 1.01429,
    "tuketim_v83_sicak_optimum.csv": 1.01318,
    "tuketim_v101_hepsi.csv": 1.01614,
    "tuketim_v102_kappa_optimum.csv": 1.00553,
    "tuketim_v109_birlesik.csv": 1.01818,
}

sample = pd.read_csv("data/raw/sample_submission.csv", dtype={"id": str})
idx = sample.id.to_numpy()
n = len(idx)


def yukle(ad: str) -> np.ndarray:
    d = pd.read_csv(f"submissions/{ad}", dtype={"id": str})
    d = d.set_index("id").reindex(idx)
    v = d.iloc[:, 0].to_numpy(dtype=float)
    assert not np.isnan(v).any(), f"{ad}: NaN"
    return np.log1p(np.clip(v, 0, None))


adlar = list(SKOR)
print(f"n = {n:,}   olculmus gonderim = {len(adlar)}")
Y = np.stack([yukle(a) for a in adlar])  # (k, n) log uzayinda

TABAN = "tuketim_v102_kappa_optimum.csv"
i0 = adlar.index(TABAN)
y0 = Y[i0]
m0 = SKOR[TABAN] ** 2

# Olculmus yonler (tabanin kendisi haric)
ad_d = [a for a in adlar if a != TABAN]
D = np.stack([Y[adlar.index(a)] - y0 for a in ad_d])  # (k-1, n)
G = D @ D.T / n
L = np.array([(m0 + G[j, j] - SKOR[a] ** 2) / 2.0 for j, a in enumerate(ad_d)])

print(f"\ntaban {TABAN}  m0 = {m0:.6f}")
print(f"olculmus yon sayisi = {len(ad_d)}")

# --- Yeni yonler ------------------------------------------------------------
y108 = yukle("tuketim_v108_sicak_onarim.csv")
yy1 = yukle("tuketim_y1_sicak_klasik.csv")
y109 = yukle("tuketim_v109_birlesik.csv")

d108 = y108 - y0
dy1 = yy1 - y0
d109 = y109 - y0

print("\n=== v109 = v102 + d108 + dy1 DOGRULAMASI ===")
fark = d109 - (d108 + dy1)
print(f"  ||d109 - (d108+dy1)|| / ||d109|| = {np.linalg.norm(fark) / np.linalg.norm(d109):.3e}")
for ad, d in (("d108", d108), ("dy1", dy1), ("d109", d109)):
    print(f"  Q({ad}) = {d @ d / n:.6f}")
print(
    f"  L(d109) olculmus = {(m0 + d109 @ d109 / n - SKOR['tuketim_v109_birlesik.csv'] ** 2) / 2:.6f}"
)


# --- Span projeksiyonu ------------------------------------------------------
def coz(d: np.ndarray, ad: str) -> None:
    b = D @ d / n
    # En kucuk kareler: G c = b  (ridge ile sayisal null'u bastir)
    w, V = np.linalg.eigh(G)
    print(f"\n=== {ad} span analizi ===")
    print(f"  Q = {d @ d / n:.6f}")
    print(f"  G ozdegerleri: max {w.max():.3e}  min {w.min():.3e}")
    for kesme in (1e-6, 1e-8, 1e-10):
        sec = w > kesme * w.max()
        Vs, ws = V[:, sec], w[sec]
        c = Vs @ ((Vs.T @ b) / ws)
        icinde = c @ G @ c
        artik = d @ d / n - icinde
        Ld = float(c @ L)
        pay = icinde / (d @ d / n)
        print(
            f"  kesme {kesme:.0e}  rank {sec.sum():2d}  "
            f"span-ici pay {pay:6.2%}  artik Q {artik:.6f}  L_span = {Ld:+.6f}"
        )


coz(d108, "d108 (sicak seviye desili, onarilmis olcut)")
coz(dy1, "dy1 (y1 yeni model ailesi)")
