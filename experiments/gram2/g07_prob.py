"""G7 -- v106: v103 + span-DISI prob yonu (perp alt-uzayinin ANA bileseni)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
from g03_sinav import coz_kesik, kur  # noqa: E402

KOK = Path(__file__).resolve().parents[2]
GON = KOK / "submissions"
CIK = Path(__file__).resolve().parent

H = kur("v83")
yon, G, b, m0, D, n = H["yon"], H["G"], H["b"], H["m0"], H["D"], H["n"]
x0 = H["X"][H["i0"]]
w103, kz, rr, *_ = coz_kesik(G, b, r=17)
m103 = m0 - kz
lam, U = np.linalg.eigh(G)
s = np.argsort(lam)[::-1]
lam, U = lam[s], U[:, s]
tut = lam > lam[0] * 1e-10
inv = np.zeros_like(lam)
inv[tut] = 1.0 / lam[tut]

ADAYLAR = [
    "tuketim_p1_sicak_ilce.csv",
    "tuketim_p3_soguk_seviye.csv",
    "tuketim_v96_grupb_optimum.csv",
    "tuketim_v95_gram_grupb.csv",
    "tuketim_v90_temiz_sota.csv",
    "tuketim_v99_mimari_sekil.csv",
]
P = []
print("span-DISI artiklar:")
for f in ADAYLAR:
    dj = np.log1p(pd.read_csv(GON / f)["tuketim"].to_numpy("f8")) - x0
    c = U @ (inv * (U.T @ ((D @ dj) / n)))
    perp = dj - c @ D
    P.append(perp)
    print(f"  {f:34s} q_perp={float(perp @ perp / n):.6f}")
P = np.array(P)
Gp = (P @ P.T) / n
lp, Up = np.linalg.eigh(Gp)
sp = np.argsort(lp)[::-1]
lp, Up = lp[sp], Up[:, sp]
print("\nperp alt-uzayi ozdegerleri: " + "  ".join(f"{v:.5f}" for v in lp))
print(
    "ana bilesen agirliklari: "
    + "  ".join(
        f"{ADAYLAR[i].replace('tuketim_', '').replace('.csv', '')}:{Up[i, 0]:+.3f}"
        for i in range(len(ADAYLAR))
    )
)
e = Up[:, 0] @ P
qe = float(e @ e / n)
e = e / np.sqrt(qe)  # birim: q=1
print(f"ana bilesen q={qe:.6f}; span ile ic carpim max={np.abs((D @ e) / n).max():.2e}")

verim = 0.014681 / np.sqrt(0.035681)  # v101'in span-disi verimliligi b/sqrt(q)
print(f"\n(v101 span-disi verimliligi b_perp/sqrt(q_perp) = {verim:.4f})")
print("Q_ek    alpha    en KOTU(b=0)   v101-verimiyle UMUT   yarim verimle")
sonuc = {
    "adaylar": ADAYLAR,
    "perp_ozdegerler": [float(v) for v in lp],
    "ana_bilesen": {ADAYLAR[i]: float(Up[i, 0]) for i in range(len(ADAYLAR))},
    "m103": float(m103),
    "varyantlar": [],
}
for Qek in [0.002, 0.004, 0.006, 0.010]:
    al = np.sqrt(Qek)
    kotu = np.sqrt(m103 + Qek)
    umut = np.sqrt(max(m103 + Qek - 2 * verim * al, 0))
    yari = np.sqrt(max(m103 + Qek - verim * al, 0))
    print(f"{Qek:.4f}  {al:.4f}   {kotu:.5f}        {umut:.5f}               {yari:.5f}")
    sonuc["varyantlar"].append(
        dict(Q_ek=Qek, en_kotu=float(kotu), umut=float(umut), yari=float(yari))
    )

Qek = 0.004
x = x0 + w103 @ D + np.sqrt(Qek) * e
kirp = int((x < 0).sum())
x = np.clip(x, 0.0, None)
pd.DataFrame({"id": H["ids"], "tuketim": np.expm1(x)}).to_csv(
    GON / "tuketim_v106_prob.csv", index=False
)
d = x - x0
dd = d - w103 @ D
print(
    f"\nYAZILDI tuketim_v106_prob.csv  Q_ek={Qek} kirpilan={kirp} "
    f"Q(v83)={float(d @ d / n):.5f}  Q(v103)={float(dd @ dd / n):.5f}"
)
sonuc["secilen"] = dict(Q_ek=Qek, dosya="tuketim_v106_prob.csv", kirpilan=kirp)
(CIK / "g07_prob.json").write_text(json.dumps(sonuc, indent=2), encoding="utf-8")
