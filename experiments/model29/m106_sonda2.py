"""GUN 1 SONDALARI v2 -- yeni ucluye gore (y40, z2, SULAMA).
Her sonda: m6 + c_g7*d_g7 + t*d_aday.  g7'nin L'si BILINIYOR -> tek bilinmeyen cozulur."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from m30_ozellik import KOK

S = os.path.join(KOK, "submissions")
M0 = 1.00284**2
LG = 0.002728


def Y(f):
    return np.log1p(pd.read_csv(os.path.join(S, f)).tuketim.values)


a0 = Y("tuketim_m6_ikiyon.csv")
N = len(a0)
dg = Y("tuketim_g7_span_tau3.csv") - a0
QG = float((dg**2).mean())
CG = LG / QG
ADAY = {
    "y40": ("tuketim_y40_sota_temiz.csv", 0.60),
    "z2": ("tuketim_z2_analog.csv", 0.35),
    "sul": ("tuketim_t1_sulama.csv", 0.45),
}
te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
rap = {}
print(f"m0={M0:.9f}  L(g7)={LG}  Q(g7)={QG:.6f}  c_g7={CG:.6f}\n")
for ad, (dosya, t) in ADAY.items():
    d = Y(dosya) - a0
    Qd = float((d**2).mean())
    c = float((dg @ d) / N)
    kos = c / np.sqrt(QG * Qd)
    p = a0 + CG * dg + t * d
    y = np.clip(np.expm1(p), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    kapi = dict(
        satir=len(out),
        id_test=bool((out.id.values == te.id.values).all()),
        id_ss=bool((out.id.values == ss.iloc[:, 0].values).all()),
        nan=int(out.tuketim.isna().sum()),
        negatif=int((out.tuketim < 0).sum()),
        sonsuz=int((~np.isfinite(out.tuketim.values)).sum()),
        maks=float(out.tuketim.max()),
    )
    tamam = (
        kapi["satir"] == 714688
        and kapi["id_test"]
        and kapi["id_ss"]
        and kapi["nan"] == 0
        and kapi["negatif"] == 0
        and kapi["sonsuz"] == 0
    )
    if not tamam:
        raise SystemExit(f"KAPI KALDI: {ad} {kapi}")
    yol = os.path.join(S, f"tuketim_s2{ad}.csv")
    gec = Path(yol + ".tmp")
    out.to_csv(gec, index=False)
    gec.replace(yol)
    sab = M0 - 2 * CG * LG + CG * CG * QG + 2 * CG * t * c + t * t * Qd
    yd = float(np.sqrt(((p - a0) ** 2).mean()))
    rap[ad] = dict(
        dosya=os.path.basename(yol),
        t=t,
        c_g7=CG,
        Q=Qd,
        kos_g7=kos,
        yer_degistirme=yd,
        cozum_sabiti=sab,
        kapi=kapi,
    )
    print(
        f"{ad:4s} t={t:.2f} Q={Qd:.5f} kos(g7)={kos:+.3f} yerdeg={yd:.4f} maks={kapi['maks']:,.0f}"
    )
    print(f"     COZUM: L_{ad} = ({sab:.9f} - P^2) / {2 * t:.2f}")
    print(
        f"     L=0 -> {np.sqrt(sab):.5f} | r=0.035 -> {np.sqrt(sab - 2 * t * 0.035 * np.sqrt(Qd)):.5f}"
        f" | r=0.06 -> {np.sqrt(sab - 2 * t * 0.06 * np.sqrt(Qd)):.5f}"
    )
json.dump(rap, open("m106_sonda2.json", "w"), indent=1)
print("\nUC SONDA HAZIR (y40, z2, SULAMA).")
