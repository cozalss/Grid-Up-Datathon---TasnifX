"""H2 -- hafta gunu / takvim YONLERININ geometrisi.

h1_q_olcum.py'nin sakladigi test tahminlerinden yon vektorleri kurar ve
mevcut aday havuzuyla kosinuslarini olcer. Yon vektoru zaten taban-bagimsiz
(iki tahminin farki), aday yonleri m6 tabanina gore alinir.

Secim olcutu (docs/59):  |kos| <= 0,20  ·  kurtoz <= 10  ·  Q >= 0,01
"""

import json
import os

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))
S = os.path.join(KOK, "submissions")

ADAYLAR = [
    "tuketim_y40_sota_temiz.csv",
    "tuketim_q1c_kapasite_siki.csv",
    "tuketim_y46_amnezik_kirpik.csv",
    "tuketim_y45_mevsimsel_kirpik.csv",
    "tuketim_z2_analog.csv",
    "tuketim_g7_span_tau3.csv",
]


def logoku(ad):
    return np.log1p(pd.read_csv(os.path.join(S, ad)).tuketim.values)


A6 = logoku("tuketim_m6_ikiyon.csv")
V102 = logoku("tuketim_v102_kappa_optimum.csv")
D4 = logoku("tuketim_m4_hava_capali.csv") - V102

P = {
    a: np.load(os.path.join(BURA, f"h1_p_{a}.npy"))
    for a in ("koy_a", "koy_b", "at", "attak", "arti")
}
YON = {
    "hafta_gunu": P["koy_a"] - P["at"],
    "takvim_taban": P["koy_a"] - P["attak"],
    "genis_takvim": P["arti"] - P["koy_a"],
    "PLASEBO_tohum": P["koy_a"] - P["koy_b"],
}
BAZ = {"m4_ekseni": D4}
for a in ADAYLAR:
    BAZ[a] = logoku(a) - A6

# test penceresinde hafta gunu kirilimi
te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"), parse_dates=["tarih"], usecols=["tarih"])
HG = te.tarih.dt.dayofweek.to_numpy()

rap = {}
for ad, d in YON.items():
    Q = float((d**2).mean())
    z = d / np.sqrt(Q)
    kos = {k: float((d * v).mean() / np.sqrt(Q * (v**2).mean())) for k, v in BAZ.items()}
    gun = {int(g): float(d[g == HG].mean()) for g in range(7)}
    # yonun ne kadari SADECE gun-ekseni (hafta gunu ortalamalari) ile aciklanir
    fit = np.array([gun[g] for g in range(7)])[HG]
    rap[ad] = dict(
        Q=Q,
        rms=float(np.sqrt(Q)),
        kurtoz=float((z**4).mean()),
        maks=float(np.abs(d).max()),
        p999=float(np.quantile(np.abs(d), 0.999)),
        en_kotu1_pay=float(np.sort(d**2)[-len(d) // 100 :].sum() / (d**2).sum()),
        kosinus=kos,
        hafta_gunu_ort=gun,
        hg_ekseni_payi=float((fit**2).mean() / Q),
    )
    print(f"\n=== {ad}")
    print(
        f"  Q={Q:.6f} rms={np.sqrt(Q):.4f} kurtoz={rap[ad]['kurtoz']:.1f} "
        f"maks={rap[ad]['maks']:.3f} p999={rap[ad]['p999']:.4f} "
        f"enkotu%1pay={rap[ad]['en_kotu1_pay']:.3f}"
    )
    print(
        "  kosinus: "
        + "  ".join(f"{k.replace('tuketim_', '')[:16]}={v:+.3f}" for k, v in kos.items())
    )
    print("  hafta gunu ort (Pzt..Paz): " + " ".join(f"{gun[g]:+.4f}" for g in range(7)))
    print(f"  yonun hafta-gunu ekseninde kalan payi: {rap[ad]['hg_ekseni_payi']:.4f}")

json.dump(rap, open(os.path.join(BURA, "h2_geometri.json"), "w"), indent=1)
print("\nYAZILDI h2_geometri.json")
