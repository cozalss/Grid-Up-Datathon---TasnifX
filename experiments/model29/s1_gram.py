"""s1 -- m6 tabaninda aday yonlerinin Gram matrisi. YEREL, gonderim YOK."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
KOK = Path(__file__).resolve().parents[2]
GON = KOK / "submissions"
BURA = Path(__file__).resolve().parent

TABAN = "tuketim_m6_ikiyon.csv"
ADAY = {
    "g7": "tuketim_g7_span_tau3.csv",
    "y46": "tuketim_y46_amnezik_kirpik.csv",
    "y45": "tuketim_y45_mevsimsel_kirpik.csv",
    "z1": "tuketim_z1_havuz.csv",
    "z2": "tuketim_z2_analog.csv",
    "z3": "tuketim_z3_ikiasama.csv",
    "r1": "tuketim_r1_seviye.csv",
    "r3": "tuketim_r3_ay.csv",
    "y40": "tuketim_y40_sota_temiz.csv",
    "y42": "tuketim_y42_kapasite_temiz.csv",
}


def oku(d):
    t = pd.read_csv(GON / d)
    assert list(t.columns) == ["id", "tuketim"], (d, list(t.columns))
    return t["id"].to_numpy(), np.log1p(t["tuketim"].to_numpy(dtype="float64"))


ids0, b = oku(TABAN)
N = b.size
adlar = list(ADAY)
D = np.empty((len(adlar), N))
for k, a in enumerate(adlar):
    ids, v = oku(ADAY[a])
    assert (ids == ids0).all(), a
    D[k] = v - b

G = (D @ D.T) / N
Q = np.diag(G).copy()
s = np.sqrt(Q)
C = G / np.outer(s, s)

# kurtoz (yonun kendi dagilimi, fazlalik kurtoz + 3)
kurt = []
for k in range(len(adlar)):
    x = D[k]
    x = x - x.mean()
    sd = x.std()
    kurt.append(float((x**4).mean() / sd**4) if sd > 0 else 0.0)

out = {
    "adlar": adlar,
    "N": int(N),
    "Q": {a: float(Q[i]) for i, a in enumerate(adlar)},
    "kurtoz": {a: kurt[i] for i, a in enumerate(adlar)},
    "C": C.tolist(),
    "G": G.tolist(),
    "kosul_tam": float(np.linalg.cond(C)),
}
(BURA / "s1_gram.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
np.save(BURA / "s1_C.npy", C)
np.save(BURA / "s1_Q.npy", Q)

print("Q ve kurtoz:")
for i, a in enumerate(adlar):
    print(f"  {a:4s} Q={Q[i]:.5f}  sqrtQ={s[i]:.4f}  kurt={kurt[i]:7.2f}")
print("\nKOSINUS MATRISI")
print("      " + " ".join(f"{a:>6s}" for a in adlar))
for i, a in enumerate(adlar):
    print(f"{a:>5s} " + " ".join(f"{C[i, j]:+6.3f}" for j in range(len(adlar))))
print(f"\nTAM kosul sayisi (10 yon): {np.linalg.cond(C):.1f}")
# alt kume kosul sayilari (g7 + ikili)
print("\nkosul(g7,A,B) ikili tablo:")
ix = {a: i for i, a in enumerate(adlar)}
c9 = adlar[1:]
print("      " + " ".join(f"{a:>6s}" for a in c9))
for A in c9:
    sat = []
    for B in c9:
        if A == B:
            sat.append("   -  ")
            continue
        sub = np.array([ix["g7"], ix[A], ix[B]])
        sat.append(f"{np.linalg.cond(C[np.ix_(sub, sub)]):6.2f}")
    print(f"{A:>5s} " + " ".join(sat))
