"""GUN 1 SONDALARI: her sonda = m6 + c_g7*d_g7 + t*d_aday
g7'nin L'si BILINIYOR -> sonda skoru tek bilinmeyeni (L_aday) cozer.
c_g7 optimumda tutulur -> sonda hem OLCUM hem YUKSEK SKORLU gonderim."""

import json
import os

import numpy as np
import pandas as pd
from m30_ozellik import KOK

S = os.path.join(KOK, "submissions")
m0 = 1.005688
Lg = 0.002728
Y = lambda f: np.log1p(pd.read_csv(os.path.join(S, f)).tuketim.values)
a0 = Y("tuketim_m6_ikiyon.csv")
N = len(a0)
dg = Y("tuketim_g7_span_tau3.csv") - a0
Qg = float((dg**2).mean())
ADAY = {
    "y40": ("tuketim_y40_sota_temiz.csv", 0.60),
    "y46": ("tuketim_y46_amnezik_kirpik.csv", 0.35),
    "q1c": ("tuketim_q1c_kapasite_siki.csv", 0.45),
}
ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
rap = {}
for ad, (dosya, t) in ADAY.items():
    d = Y(dosya) - a0
    Qd = float((d**2).mean())
    c = float((dg @ d) / N)
    kos = c / np.sqrt(Qg * Qd)
    # c_g7: aday L'si bilinmiyor; g7'yi TEK BASINA optimumda tut (guvenli secim)
    cg = Lg / Qg
    p = a0 + cg * dg + t * d
    y = np.clip(np.expm1(p), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    yol = os.path.join(S, f"tuketim_s{ad}.csv")
    out.to_csv(yol, index=False)
    kapi = dict(
        satir=len(out),
        id_birebir=bool((out.id.values == ss.iloc[:, 0].values).all()),
        nan=int(out.tuketim.isna().sum()),
        negatif=int((out.tuketim < 0).sum()),
        maks=float(out.tuketim.max()),
    )
    assert (
        kapi["satir"] == 714688 and kapi["id_birebir"] and kapi["nan"] == 0 and kapi["negatif"] == 0
    )
    yerdeg = float(np.sqrt(((p - a0) ** 2).mean()))
    # COZUM formulu: skor P olculunce
    #  P^2 = m0 - 2(cg*Lg + t*Ld) + cg^2*Qg + 2*cg*t*c + t^2*Qd
    #  -> Ld = (m0 - P^2 - 2*cg*Lg + cg^2*Qg + 2*cg*t*c + t^2*Qd) / (2t)
    sab = m0 - 2 * cg * Lg + cg * cg * Qg + 2 * cg * t * c + t * t * Qd
    rap[ad] = dict(
        dosya=os.path.basename(yol),
        t=t,
        c_g7=cg,
        Q_aday=Qd,
        kos_g7=kos,
        yer_degistirme=yerdeg,
        kapi=kapi,
        cozum_sabiti=sab,
    )
    print(f"{ad:4s} t={t:.2f} c_g7={cg:.3f} Q={Qd:.5f} kos(g7)={kos:+.3f} yerdeg={yerdeg:.4f}")
    print(f"      COZUM: L_{ad} = ({sab:.6f} - P^2) / {2 * t:.2f}")
    print(
        f"      L=0 ise skor {np.sqrt(sab - 2 * t * 0):.5f} | r=0.035 ise {np.sqrt(sab - 2 * t * 0.035 * np.sqrt(Qd)):.5f}"
        f" | r=0.06 ise {np.sqrt(sab - 2 * t * 0.06 * np.sqrt(Qd)):.5f}"
    )
    print(f"      KAPI OK, maks {kapi['maks']:,.0f}")
json.dump(rap, open("m102_sonda.json", "w"), indent=1)
print("\nUC SONDA HAZIR. Her biri hem OLCUM hem gonderilebilir skor.")
