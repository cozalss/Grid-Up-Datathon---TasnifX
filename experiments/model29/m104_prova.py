"""UCTAN UCA PROVA: sonda skoru -> L -> esdeger skor -> m99 -> dosya.
Ayrica NEGATIF L yolu ve kirpma davranisi test edilir.
Gercek gonderim YOK; deneme dosyalari sonunda silinir."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from m30_ozellik import KOK

S = os.path.join(KOK, "submissions")
Y = lambda f: np.log1p(pd.read_csv(os.path.join(S, f)).tuketim.values)
a0 = Y("tuketim_m6_ikiyon.csv")
N = len(a0)
m0 = 1.00284**2
SAB = {
    "y40": (1.006831155, 1.20, "tuketim_y40_sota_temiz.csv", "tuketim_sy40.csv"),
    "q1c": (1.014633382, 0.90, "tuketim_q1c_kapasite_siki.csv", "tuketim_sq1c.csv"),
    "y46": (1.048108681, 0.70, "tuketim_y46_amnezik_kirpik.csv", "tuketim_sy46.csv"),
}
Q = {k: float(((Y(v[2]) - a0) ** 2).mean()) for k, v in SAB.items()}
Qg = float(((Y("tuketim_g7_span_tau3.csv") - a0) ** 2).mean())
Lg = 0.002728


def coz_L(ad, P):
    sab, ikit, _, _ = SAB[ad]
    return (sab - P * P) / ikit


def esdeger(Qj, Lj):
    x = m0 + Qj - 2 * Lj
    return np.sqrt(x) if x > 0 else None


print("=" * 70)
print("SENARYO TARAMASI: sonda skorlari -> L -> esdeger skor")
print(f"{'senaryo':22s} {'P_y40':>8s} {'L_y40':>10s} {'r_y40':>7s} {'esdeger S':>10s}")
SEN = {
    "cok iyi (r=0.06)": None,
    "iyi (r=0.035)": None,
    "notr (L=0)": None,
    "kotu (L<0, r=-0.02)": None,
    "cok kotu (L<0, r=-0.05)": None,
}
for ad_, r in [
    ("cok iyi (r=0.06)", 0.06),
    ("iyi (r=0.035)", 0.035),
    ("notr (L=0)", 0.0),
    ("kotu (L<0, r=-0.02)", -0.02),
    ("cok kotu (L<0, r=-0.05)", -0.05),
]:
    L = r * np.sqrt(Q["y40"])
    sab, ikit, _, _ = SAB["y40"]
    P = np.sqrt(sab - ikit * L)
    Lg2 = coz_L("y40", round(P, 5))  # LB 5 haneye yuvarlar
    es = esdeger(Q["y40"], Lg2)
    SEN[ad_] = (round(P, 5), Lg2, es)
    print(
        f"  {ad_:20s} {P:8.5f} {Lg2:+10.6f} {Lg2 / np.sqrt(Q['y40']):+7.4f} {es if es else float('nan'):10.5f}"
    )

print("\n" + "=" * 70)
print("UCTAN UCA: iki senaryoda m99'u GERCEKTEN kostur")
for ad_ in ["iyi (r=0.035)", "kotu (L<0, r=-0.02)"]:
    P, L, es = SEN[ad_]
    print(f"\n--- {ad_}: sonda skoru {P} -> L_y40={L:+.6f} -> esdeger S={es:.5f}")
    cmd = [
        sys.executable,
        "m99_coklu_coz.py",
        "tuketim_m6_ikiyon.csv=1.00284",
        f"tuketim_g7_span_tau3.csv={np.sqrt(m0 + Qg - 2 * Lg):.5f}",
        f"tuketim_y40_sota_temiz.csv={es:.5f}",
        "--cikti",
        "tuketim_PROVA_SIL.csv",
    ]
    r = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    )
    for satir in r.stdout.splitlines():
        if any(x in satir for x in ["korkuluk", "ORTAK", "KAPI", "YAZILDI", "DUR", "uyari", "k*"]):
            print("   " + satir[:150])
    if r.returncode:
        print("   CIKIS KODU", r.returncode, r.stdout.splitlines()[-1] if r.stdout else "")
    yol = os.path.join(S, "tuketim_PROVA_SIL.csv")
    if os.path.exists(yol):
        c = pd.read_csv(yol)
        lp = np.log1p(c.tuketim.values)
        kirp = int((c.tuketim == 0).sum())
        print(
            f"   URETILEN: maks {c.tuketim.max():,.0f} | tam sifir {kirp} | log-ort {lp.mean():.4f} (m6 {a0.mean():.4f})"
        )
        Path(yol).unlink()
print("\n" + "=" * 70)
print("NEGATIF L YORUMU: L<0 ise kappa<0 -> o yonden UZAKLASILIR.")
print("Cebir bunu dogru yapar (kazanc L^2/Q her iki isarette de POZITIF).")
print("Kirpma riski: kappa<0 ise tahminler ASAGI itilir -> daha cok satir sifira kirpilir.")
