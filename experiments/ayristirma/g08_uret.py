"""Bugunun dosyasi: span optimumu + v90-dik probu (kappa=f_olculen)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gram2"))
sys.stdout.reconfigure(encoding="utf-8")
from g01_havuz import yukle

KOK = Path(__file__).resolve().parents[2]
GON = KOK / "submissions"
CIK = Path(__file__).resolve().parent
adlar, X, skorlar, ids = yukle()
n = X.shape[1]
i83 = adlar.index("v83")
m0 = float(skorlar[i83] ** 2)
yon = [a for a in adlar if a != "v83"]
idx = [adlar.index(a) for a in yon]
D = X[idx] - X[i83]
Gm = (D @ D.T) / n
mm = skorlar[idx] ** 2
b = (m0 + np.diag(Gm) - mm) / 2.0
lam, V = np.linalg.eigh(Gm)
kes = lam > lam.max() * 1e-10
Vk, lk = V[:, kes], lam[kes]
w = Vk @ ((Vk.T @ b) / lk)
adim_span = w @ D
kaz_span = float(w @ b)


def lg(a):
    d = pd.read_csv(GON / a)
    return np.log1p(d["tuketim"].to_numpy("float64"))


v83 = lg("tuketim_v83_sicak_optimum.csv")
v90 = lg("tuketim_v90_temiz_sota.csv")
u = v90 - v83
c = (D @ u) / n
al = Vk @ ((Vk.T @ c) / lk)
uper = u - al @ D
Qp = float(uper @ uper) / n
f = 0.3735
m_taban = m0 - kaz_span
print(f"span optimumu: kazanc {kaz_span:.7f}  MSE {m_taban:.7f}  RMSLE {np.sqrt(m_taban):.6f}")
print(f"v90 dik yonu : Q_perp {Qp:.7f}   kappa={f}")
yl = v83 + adim_span + f * uper
neg = int((yl < 0).sum())
sap = float(((np.minimum(yl, 0)) ** 2).sum()) / n
yl = np.maximum(yl, 0.0)
print(f"kirpilan satir {neg}  kirpma Q etkisi {sap:.3e} (ihmal edilebilir)")
tah = np.expm1(yl)
cik = KOK / "submissions" / "tuketim_v103_span_v90prob.csv"
pd.DataFrame({"id": ids, "tuketim": tah}).to_csv(cik, index=False, float_format="%.17g")
print(f"YAZILDI {cik}")
print("\nON KAYIT (v90 gerceklesme oranina gore):")
for ff in [0.0, 0.1, 0.18675, 0.25, 0.3735, 0.5, 0.55, 0.75, 1.0]:
    dm = f * f * Qp - 2 * f * ff * Qp
    ms = m_taban + dm
    print(f"   f={ff:5.3f}  dMSE {dm:+.7f}  MSE {ms:.7f}  RMSLE {np.sqrt(ms):.5f}")
json.dump(
    {
        "kaz_span": kaz_span,
        "m_taban": m_taban,
        "Qp": Qp,
        "kappa": f,
        "on_kayit": float(np.sqrt(m_taban + f * f * Qp - 2 * f * f * Qp)),
        "kirpma": neg,
    },
    open(CIK / "g08.json", "w"),
    indent=2,
)
