"""KAC YON OLCERSEK 2. SIRA OLASILIGI NE OLUR?
Ayni kalibre onsel (m6 tabanina gore gecmis |rho|), degisen tek sey yon sayisi.
Ayrica: tehlikeli adaylar (dusuk Q, yuksek esdogrusallik) fayda mi zarar mi?
"""

import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
S = os.path.join(KOK, "submissions")
SK = json.load(open(os.path.join(KOK, "experiments/model29/olculmus_skorlar.json")))
TABAN = "tuketim_m6_ikiyon.csv"
m0 = SK[TABAN] ** 2
HEDEF, LG, GUR = 0.99940, 0.002728, 1.6e-4


def oku(f):
    df = pd.read_csv(os.path.join(S, f))
    kol = "tuketim" if "tuketim" in df.columns else df.columns[-1]
    return np.log1p(df[kol].values.astype(np.float64))


a0 = oku(TABAN)
N = len(a0)
RHO = np.array(
    [
        0.0080,
        0.0084,
        0.0092,
        0.0110,
        0.0161,
        0.0171,
        0.0191,
        0.0243,
        0.0251,
        0.0278,
        0.0289,
        0.0296,
        0.0310,
        0.0327,
        0.0334,
        0.0341,
        0.0342,
        0.0360,
        0.0360,
        0.0360,
    ]
)

AD = {
    "g7": "tuketim_g7_span_tau3.csv",
    "y40": "tuketim_y40_sota_temiz.csv",
    "z2": "tuketim_z2_analog.csv",
    "sul": "tuketim_t1_sulama.csv",
    "y46": "tuketim_y46_amnezik_kirpik.csv",
    "y45": "tuketim_y45_mevsimsel_kirpik.csv",
    "q1c": "tuketim_q1c_kapasite_siki.csv",
    "t3": "tuketim_t3_turizm.csv",
    "h1": "tuketim_h1_isil.csv",
    "t2": "tuketim_t2_bayram.csv",
    "k5": "tuketim_k5_kesinti.csv",
}
d = {k: oku(v) - a0 for k, v in AD.items()}
print("yukleme tamam\n")


def mc(yonler, olcek=1.0, ns=60_000, tohum=11, lam=0.0, gur=GUR):
    ks = ["g7"] + list(yonler)
    D = np.array([d[k] for k in ks]).T
    G = (D.T @ D) / N
    sq = np.sqrt(np.diag(G))
    Gi = np.linalg.inv(G + lam * np.eye(len(ks)))
    rng = np.random.default_rng(tohum)
    K = len(yonler)
    r = rng.choice(RHO, size=(ns, K)) * olcek * rng.choice([-1.0, 1.0], size=(ns, K))
    L = np.concatenate([np.full((ns, 1), LG), r * sq[1:]], axis=1)
    Lh = L + np.concatenate([np.zeros((ns, 1)), rng.normal(0, gur, (ns, K))], axis=1)
    kk = Lh @ Gi.T
    mse = m0 - 2 * (kk * L).sum(1) + ((kk @ G) * kk).sum(1)
    s = np.sqrt(np.maximum(mse, 1e-12))
    taban = np.sqrt(
        max(
            m0 - np.concatenate([[LG], np.zeros(K)]) @ Gi @ np.concatenate([[LG], np.zeros(K)]),
            1e-12,
        )
    )
    return dict(
        p2=(s < HEDEF).mean(),
        med=np.median(s),
        p95=np.percentile(s, 95),
        taban=taban,
        cond=np.linalg.cond(G),
        k1=np.abs(kk).sum(1).mean(),
        kotu=(s > 1.00284).mean(),
    )


SETLER = [
    ("3 yon (eski plan)", ["y40", "z2", "sul"]),
    ("6 yon", ["y40", "z2", "sul", "y46", "y45", "q1c"]),
    ("8 yon (YENI plan)", ["y40", "z2", "sul", "y46", "y45", "q1c", "t3", "h1"]),
]
print(
    f"{'kume':>20s} {'onsel':>10s} {'gurultu':>9s} {'P(2.)':>7s} {'medyan':>9s} {'P(1.00284 kotu)':>16s}"
)
for ad, ys in SETLER:
    for olcek, et in [(1.0, "gercekci"), (0.5, "kotumser"), (0.25, "cok kotu")]:
        for g, ge in [(1.6e-4, "iyimser"), (4.0e-4, "kotumser")]:
            r = mc(ys, olcek, gur=g)
            print(
                f"{ad:>20s} {et:>10s} {ge:>9s} {r['p2'] * 100:6.1f}% {r['med']:9.5f} {r['kotu'] * 100:15.1f}%"
            )
    print()
