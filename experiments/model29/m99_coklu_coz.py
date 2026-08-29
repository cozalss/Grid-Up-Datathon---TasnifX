"""COK-YONLU COZUCU: birden fazla olculmus yonun ORTAK optimumu.

Kullanim:
    python m99_coklu_coz.py taban.csv=<skor> yon1.csv=<skor> yon2.csv=<skor> ... [--cikti ad.csv]

Cebir (yonler DIK OLMAK ZORUNDA DEGIL -- tam Gram cozulur):
    p = a + sum_j k_j d_j ,  d_j = log1p(yon_j) - log1p(a)
    MSE(k) = m0 - 2 k'L + k'G k        G_ij = <d_i,d_j>/N ,  L_j = (m0 + G_jj - m_j)/2
    k*  = G^-1 L        MSE* = m0 - L' G^-1 L
Ridge (public/private gurultusune karsi, kural 39):  k* = (G + lam*I)^-1 L
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from m30_ozellik import KOK


def yukle(ad):
    return np.log1p(
        pd.read_csv(
            os.path.join(KOK, "submissions", ad) if not os.path.exists(ad) else ad
        ).tuketim.values
    )


def coz(taban_ad, taban_skor, yonler, lam=0.0, cikti=None, yaz=True):
    a = yukle(taban_ad)
    N = len(a)
    m0 = float(taban_skor) ** 2
    D = []
    m = []
    adlar = []
    for ad, sk in yonler:
        D.append(yukle(ad) - a)
        m.append(float(sk) ** 2)
        adlar.append(ad)
    D = np.array(D)
    m = np.array(m)
    G = D @ D.T / N
    L = (m0 + np.diag(G) - m) / 2
    Gr = G + lam * np.eye(len(G))
    k = np.linalg.solve(Gr, L)
    mse = m0 - 2 * k @ L + k @ G @ k
    print(f"taban {taban_ad} skor {taban_skor}  m0={m0:.6f}")
    print(f"{'yon':34s} {'Q(=G_jj)':>10s} {'L':>10s} {'tek-basina kazanc':>18s} {'k*':>9s}")
    for j, ad in enumerate(adlar):
        print(f"  {ad:32s} {G[j, j]:10.6f} {L[j]:+10.6f} {L[j] ** 2 / G[j, j]:18.6f} {k[j]:+9.4f}")
    print("\nGram kosinusleri:")
    s = np.sqrt(np.diag(G))
    for i in range(len(G)):
        print("   " + " ".join(f"{G[i, j] / (s[i] * s[j]):+6.3f}" for j in range(len(G))))
    print(f"\n|k|_1 = {np.abs(k).sum():.4f}   ridge lam={lam}")
    print(f"ORTAK OPTIMUM MSE = {mse:.6f}   RMSLE = {np.sqrt(max(mse, 0)):.5f}")
    print(f"  (tek tek toplasaydik: {m0 - sum(L[j] ** 2 / G[j, j] for j in range(len(G))):.6f})")
    if not yaz:
        return np.sqrt(max(mse, 0)), k
    p = a + (k[:, None] * D).sum(0)
    y = np.clip(np.expm1(p), 0.0, None)
    te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    cikti = cikti or "tuketim_coklu_optimum.csv"
    yol = os.path.join(KOK, "submissions", cikti)
    out.to_csv(yol, index=False)
    ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
    kapi = dict(
        satir=len(out),
        id_birebir=bool((out.id.values == ss.iloc[:, 0].values).all()),
        nan=int(out.tuketim.isna().sum()),
        negatif=int((out.tuketim < 0).sum()),
        maks=float(out.tuketim.max()),
    )
    print("KAPI:", json.dumps(kapi))
    assert (
        kapi["satir"] == 714688 and kapi["id_birebir"] and kapi["nan"] == 0 and kapi["negatif"] == 0
    )
    print(f"YAZILDI {yol}")
    return np.sqrt(max(mse, 0)), k


if __name__ == "__main__":
    arg = [x for x in sys.argv[1:] if not x.startswith("--")]
    cikti = None
    if "--cikti" in sys.argv:
        cikti = sys.argv[sys.argv.index("--cikti") + 1]
    lam = 0.0
    if "--lam" in sys.argv:
        lam = float(sys.argv[sys.argv.index("--lam") + 1])
    tab = arg[0].split("=")
    yon = [a.split("=") for a in arg[1:]]
    coz(tab[0], tab[1], [(a, s) for a, s in yon], lam=lam, cikti=cikti)
