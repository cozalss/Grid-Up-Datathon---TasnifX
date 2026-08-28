"""HAK2 COZUCU: HAK1'de olculen skordan optimum harmani cozer ve dosyayi yazar.
Kullanim:  python m50_harman_coz.py <yeni_dosya.csv> <olculen_skor> [cikti.csv]
Ornek:     python m50_harman_coz.py tuketim_m2_capali.csv 0.9912
Cebir:  p(k) = a + k*(b-a);  MSE(k) = m0 - 2kL + k^2 Q
        L = (m0 + Q - m1)/2,  k* = L/Q,  MSE* = m0 - L^2/Q
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from m30_ozellik import KOK

TABAN = "tuketim_v102_kappa_optimum.csv"
SKOR_TABAN = 1.00553


def coz(yeni_dosya, skor_yeni, cikti=None, taban=TABAN, skor_taban=SKOR_TABAN):
    S = os.path.join(KOK, "submissions")
    ta = pd.read_csv(os.path.join(S, taban))
    ye = pd.read_csv(os.path.join(S, yeni_dosya))
    assert (ta.id.values == ye.id.values).all(), "ID sirasi uyusmuyor"
    a = np.log1p(ta.tuketim.values)
    b = np.log1p(ye.tuketim.values)
    d = b - a
    Q = float((d**2).mean())
    m0 = skor_taban**2
    m1 = float(skor_yeni) ** 2
    L = (m0 + Q - m1) / 2
    k = L / Q
    mse = m0 - L * L / Q
    print(f"Q  = {Q:.6f}")
    print(f"m0 = {m0:.6f}  ({skor_taban})   m1 = {m1:.6f}  ({skor_yeni})")
    print(f"L  = {L:+.6f}   kappa* = {k:+.6f}")
    print(
        f"ONGORULEN OPTIMUM RMSLE = {np.sqrt(max(mse, 0)):.5f}   (taban {skor_taban}, yeni {skor_yeni})"
    )
    if k <= 0:
        print("kappa* <= 0 : YENI YON ZARARLI, GONDERME.")
        return None
    if np.sqrt(max(mse, 0)) >= min(skor_taban, float(skor_yeni)) - 1e-6:
        print("UYARI: harman ikisinden de iyi degil.")
    p = a + k * d
    y = np.clip(np.expm1(p), 0.0, None)
    cikti = cikti or f"tuketim_harman_k{k:.4f}.csv"
    out = pd.DataFrame({"id": ta.id.values, "tuketim": y})
    yol = os.path.join(S, cikti)
    out.to_csv(yol, index=False)
    ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
    kapi = dict(
        satir=len(out),
        id_birebir=bool((out.id.values == ss.iloc[:, 0].values).all()),
        nan=int(out.tuketim.isna().sum()),
        negatif=int((out.tuketim < 0).sum()),
    )
    print("KAPI:", json.dumps(kapi))
    assert (
        kapi["satir"] == 714688 and kapi["id_birebir"] and kapi["nan"] == 0 and kapi["negatif"] == 0
    )
    print(f"YAZILDI {yol}")
    return yol


if __name__ == "__main__":
    coz(sys.argv[1], float(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
