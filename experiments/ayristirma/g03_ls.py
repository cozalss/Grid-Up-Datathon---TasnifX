from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
G = KOK / "submissions"


def lg(a):
    d = pd.read_csv(G / a)
    return np.log1p(d["tuketim"].to_numpy("float64")), d["id"]


names = {
    "v83": "tuketim_v83_sicak_optimum.csv",
    "v93": "tuketim_v93_gram_optimum.csv",
    "P1": "tuketim_p1_sicak_ilce.csv",
    "P3": "tuketim_p3_soguk_seviye.csv",
    "B": "tuketim_v96_grupb_optimum.csv",
    "bos": "tuketim_v94_bosluk_oncesi.csv",
    "v101": "tuketim_v101_hepsi.csv",
    "v85": "tuketim_v85_gram_rank2.csv",
    "P2": "tuketim_p2_sicak_seviye.csv",
    "P4": "tuketim_p4_sicak_ay.csv",
    "P5": "tuketim_p5_soguk_kva.csv",
    "v90": "tuketim_v90_temiz_sota.csv",
    "v99": "tuketim_v99_mimari_sekil.csv",
    "v82": "tuketim_v82_ayirici.csv",
    "yas": "tuketim_prob_yas790.csv",
    "v98": "tuketim_v98_mimari_prob.csv",
    "v91": "tuketim_v91_grupb_kaldirma.csv",
}
F = {}
ids = None
for k, v in names.items():
    F[k], i = lg(v)
    if ids is None:
        ids = i
n = F["v83"].size
d = F["v101"] - F["v83"]
C = {
    "v93": F["v93"] - F["v83"],
    "P1": F["P1"] - F["v93"],
    "P3": F["P3"] - F["v93"],
    "B": F["B"] - F["v93"],
    "bos": F["bos"] - F["v93"],
}
ad = list(C)
A = np.array([C[k] for k in ad])
coef, _, _, _ = np.linalg.lstsq(A.T, d, rcond=None)
res = d - coef @ A
print("EN KUCUK KARELER katsayilari (d = sum a_i c_i):")
for k, a in zip(ad, coef):
    print(f"   a[{k:5s}] = {a:+.6f}")
print(f"kalinti Q = {float(res @ res) / n:.7f}  (a=1 iken 9.795e-3)")
# kalintinin anatomisi
print(
    f"\nKALINTI r: Q={float(res @ res) / n:.7f}  sifirdan farkli satir={int((np.abs(res) > 1e-9).sum()):,}"
)
nz = np.abs(res) > 1e-9
print(
    f"   r[nz] ort={res[nz].mean():+.5f} med={np.median(res[nz]):+.5f} min={res.min():+.4f} max={res.max():+.4f}"
)
# hangi satirlar?
tanim = ids.str.rsplit("_", n=1).str[0]
tr = pd.read_csv(KOK / "data/raw/train.csv", usecols=["tanim"], dtype={"tanim": str})
sicak = tanim.isin(set(tr["tanim"].unique())).to_numpy()
print(f"   nz satirlarin %{100 * sicak[nz].mean():.1f}'i SICAK  (genel %{100 * sicak.mean():.1f})")
print(f"   nz trafo sayisi = {tanim[nz].nunique():,} / {tanim.nunique():,}")
# a=1 kalintisi
res1 = d - A.sum(0)
nz1 = np.abs(res1) > 1e-9
print(
    f"\na=1 KALINTISI: Q={float(res1 @ res1) / n:.7f} nz={int(nz1.sum()):,} trafo={tanim[nz1].nunique():,}"
)
print(f"   ort={res1[nz1].mean():+.5f} min={res1.min():+.4f} max={res1.max():+.4f}")
print(f"   nz satirlarin %{100 * sicak[nz1].mean():.1f}'i SICAK")
# kalinti diger dosyalarla ortusuyor mu
for k in ["v85", "v90", "v99", "v98", "v82", "yas", "P2", "v91"]:
    u = F[k] - F["v83"]
    c = float(res1 @ u) / np.sqrt(float(res1 @ res1) * float(u @ u))
    print(f"   cos(r1, {k:5s}-v83) = {c:+.4f}")
np.save(Path(__file__).parent / "r1.npy", res1)
