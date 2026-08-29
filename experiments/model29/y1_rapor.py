"""Nihai aday raporu -> y1_yeni_yonler.json"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from y1_olcum import A6, M0, S, olc  # noqa: E402

ADAYLAR = [
    (
        "y40_sota_temiz",
        "tuketim_y40_sota_temiz.csv",
        "SOTA hatti (CatBoost+LGBM+XGB, ~160 oznitelik, rejim uzmanlari) -- BAGIMSIZ KOD TABANI; "
        "LB'de curutulmus olu-trafo bileseni cikarildi",
    ),
    (
        "y45_mevsimsel_kirpik",
        "tuketim_y45_mevsimsel_kirpik.csv",
        "KLASIK zaman serisi (GBM DEGIL): mevsimsel naif + YoY suruklenme + yas-kohort; "
        "curuk bilesen cikarildi, kaldirac winsorize",
    ),
    (
        "y46_amnezik_kirpik",
        "tuketim_y46_amnezik_kirpik.csv",
        "AMNEZIK GBM: trafo-duzeyi gecmis ozellikleri (24 kolon) tamamen atildi; "
        "'taban gecmise fazla mi guveniyor' eksenini yoklar (kappa isareti serbest)",
    ),
    (
        "y42_kapasite_temiz",
        "tuketim_y42_kapasite_temiz.csv",
        "KAPASITE OFSETLI hedef: log1p(tuketim)-log1p(guc); m4 yonuyle %50 ortusuyor",
    ),
    ("y43_mevsimsel_temiz", "tuketim_y43_mevsimsel_temiz.csv", "y45'in winsorize edilmemis hali"),
    ("y41_amnezik_temiz", "tuketim_y41_amnezik_temiz.csv", "y46'nin winsorize edilmemis hali"),
    ("m3_hl1_capali", "tuketim_m3_hl1_capali.csv", "mevcut havasiz surum (referans)"),
    ("y30_sota_ham", "tuketim_y30_sota.csv", "SOTA ham cikti -- curuk bilesen ICINDE (GONDERME)"),
]

D = {}
sonuc = []
for ad, dosya, aciklama in ADAYLAR:
    r = olc(dosya)
    r["ad"] = ad
    r["aciklama"] = aciklama
    Q, yp = r["Q"], r["yeni_pay"]
    r["basabas_skor_L0"] = float(np.sqrt(M0 + Q))
    r["beklenen_kazanc"] = {
        "m4_kalitesi_r0.064": 0.064**2 * yp,
        "yari_kalite_r0.032": 0.032**2 * yp,
        "ceyrek_kalite_r0.016": 0.016**2 * yp,
    }
    r["L_cozum_hassasiyeti"] = float(np.sqrt(M0 + Q) * 5e-6)
    sonuc.append(r)
    D[ad] = np.log1p(pd.read_csv(os.path.join(S, dosya)).tuketim.values) - A6

V102 = np.log1p(pd.read_csv(os.path.join(S, "tuketim_v102_kappa_optimum.csv")).tuketim.values)
D["m4-v102(harcanmis)"] = (
    np.log1p(pd.read_csv(os.path.join(S, "tuketim_m4_hava_capali.csv")).tuketim.values) - V102
)
D["m6-v102(harcanmis)"] = A6 - V102
ks = list(D)
KOS = {
    i: {
        j: float((D[i] * D[j]).mean() / np.sqrt((D[i] ** 2).mean() * (D[j] ** 2).mean()))
        for j in ks
    }
    for i in ks
}

print(f"{'aday':24s} {'Q':>8s} {'yeni%':>6s} {'kos_m4':>7s} {'basabas':>8s} {'bekl.kazanc':>11s}")
for r in sonuc:
    print(
        f"{r['ad']:24s} {r['Q']:8.5f} {100 * r['yeni_pay']:6.1f} {r['kos_m4']:+7.3f} "
        f"{r['basabas_skor_L0']:8.5f} {r['beklenen_kazanc']['m4_kalitesi_r0.064']:11.5f}"
    )
print("\nKOSINUS MATRISI (secilen 3 aday + harcanmis span)")
sec = [
    "y40_sota_temiz",
    "y45_mevsimsel_kirpik",
    "y46_amnezik_kirpik",
    "m4-v102(harcanmis)",
    "m6-v102(harcanmis)",
]
print("                     " + " ".join(f"{k[:10]:>11s}" for k in sec))
for i in sec:
    print(f"{i:20s} " + " ".join(f"{KOS[i][j]:+11.3f}" for j in sec))

json.dump(
    dict(
        taban=dict(dosya="tuketim_m6_ikiyon.csv", LB=1.00284, m0=M0),
        hedef=dict(sira2=1.00041, gereken_dMSE=-0.004888),
        olculmus_span=["m4 - v102", "m6 - v102"],
        kalibrasyon=dict(m4_L_bolu_sqrtQ=0.0640, m4_kazanci=0.005245),
        adaylar=sonuc,
        kosinus=KOS,
        kappa_cozumu="L = (m0 + Q - S^2)/2 ; kappa* = L/Q ; yeni m0 = m0 - L^2/Q",
    ),
    open(os.path.join(BURA, "y1_yeni_yonler.json"), "w"),
    indent=1,
)
print("\nYAZILDI y1_yeni_yonler.json")
