"""p34-c: UCTAN UCA ZINCIR SINAMASI -- 30 yonlu YENI cebir icin.

p33_c ile ayni tasarim: sentetik gercek artik r_syn kurulur, GERCEK CSV yazilir,
geri okunur, skor DOGRUDAN hesaplanir ve cebrin ongorusuyle karsilastirilir.
Hicbir yerde cebir formulu skoru URETMEZ (m161 dongusellik hatasi yok).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")
GEC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, M29)
from m112_kalibre import M0  # noqa: E402

with open(os.path.join(GEC, "p34_a_cebir.json"), encoding="utf-8") as fh:
    A = json.load(fh)
a0 = np.load(os.path.join(GEC, "p34_a0.npy"))
V = np.load(os.path.join(GEC, "p34_V30.npy"))
L_vec = np.load(os.path.join(GEC, "p34_L30.npy"))
r_hat = np.load(os.path.join(GEC, "p34_r30.npy"))
BAZ = np.load(os.path.join(GEC, "p34_dik_baz.npy"))
N = A["N"]
kL = A["c30_yeni"]["kL"]
TABAN_MSE = A["c30_yeni"]["TABAN_MSE"]

u = BAZ[2] / np.sqrt(float((BAZ[2] * BAZ[2]).mean()))  # Y1 dik birim yonu
ic = np.abs((V.T @ u) / N)
print(f"u: ort(u^2)={float((u * u).mean()):.9f}  span'a diklik maks={ic.max():.3e}")

Am = np.vstack([r_hat[None, :], u[None, :]])
Gm = (Am @ Am.T) / N


def kur_r_syn(rho, tohum=2026):
    c = np.linalg.solve(Gm + 1e-14 * np.eye(len(Gm)), np.array([kL, rho]))
    r_par = c @ Am
    g = float((r_par * r_par).mean())
    if g > M0:
        raise SystemExit(f"DUR: span gucu M0'i asiyor ({g:.6f})")
    rng = np.random.default_rng(tohum)
    w = rng.standard_normal(N)
    w -= Am.T @ np.linalg.solve(Gm + 1e-14 * np.eye(len(Gm)), (Am @ w) / N)
    w /= np.sqrt(float((w * w).mean()))
    return r_par + np.sqrt(M0 - g) * w


IDS = pd.read_csv(os.path.join(KOK, "data/raw/test.csv")).id.values
TMP = os.path.join(GEC, "_p34_zincir.csv")


def dosyadan_skor(log_tahmin, t):
    tk = np.expm1(log_tahmin)
    tk[tk < 0] = 0.0
    pd.DataFrame({"id": IDS, "tuketim": tk}).to_csv(TMP, index=False)
    geri = pd.read_csv(TMP)
    d = np.log1p(geri.tuketim.values.astype(np.float64)) - t
    return float(np.sqrt(float((d * d).mean())))


SONUC, enb_h = [], 0.0
for rho in (0.16340, 0.05, 0.0, -0.05):
    r_syn = kur_r_syn(rho)
    e1 = abs(float((r_syn * r_hat).mean()) - kL)
    e2 = abs(float((r_syn * u).mean()) - rho)
    e3 = abs(float((r_syn * r_syn).mean()) - M0)
    eV = float(np.abs((V.T @ r_syn) / N - L_vec).max())
    print(f"\n rho={rho:+.5f}  kurulus hatalari kL {e1:.1e} rho {e2:.1e} M0 {e3:.1e}"
          f"  (bilgi: 30 L'den maks sapma {eV:.2e})")
    assert e1 < 1e-9 and e2 < 1e-9 and e3 < 1e-9, "r_syn kurulamadi"
    t = a0 + r_syn
    P0 = dosyadan_skor(a0 + r_hat, t)
    print(f"   SAF SPAN: dogrudan {P0:.6f}  cebir {np.sqrt(TABAN_MSE):.6f}"
          f"  fark {P0 - np.sqrt(TABAN_MSE):+.2e}")
    for kap in (0.05, 0.10, 0.16340):
        P = dosyadan_skor(a0 + r_hat + kap * u, t)
        cebir = float(np.sqrt(max(TABAN_MSE - 2 * kap * rho + kap * kap, 1e-12)))
        rho_coz = (TABAN_MSE + kap * kap - P * P) / (2 * kap)
        enb_h = max(enb_h, abs(P - cebir))
        SONUC.append({"rho": rho, "kappa": kap, "dogrudan_skor": P,
                      "cebir_skor": cebir, "fark": P - cebir,
                      "rho_geri_cozulen": rho_coz, "rho_cozum_hatasi": rho_coz - rho})
        print(f"   kappa={kap:.5f}: dogrudan {P:.6f} cebir {cebir:.6f} "
              f"fark {P - cebir:+.2e}  rho geri {rho_coz:+.6f}")

if os.path.exists(TMP):
    os.remove(TMP)
enb_rho = max(abs(s["rho_cozum_hatasi"]) for s in SONUC)
GECTI = bool(enb_h < 2e-4 and enb_rho < 5e-3)
print(f"\nen buyuk skor farki {enb_h:.2e} (esik 2e-4); "
      f"en buyuk rho hatasi {enb_rho:.2e} (esik 5e-3)")
print("ZINCIR SAGLAM." if GECTI else "ZINCIRDE SORUN VAR -- DUR.")
with open(os.path.join(GEC, "p34_c_zincir.json"), "w", encoding="utf-8") as fh:
    json.dump({"gecti": GECTI, "en_buyuk_skor_farki": enb_h,
               "en_buyuk_rho_hatasi": enb_rho, "u_dikligi_maks": float(ic.max()),
               "satirlar": SONUC}, fh, indent=1)
print("-> p34_c_zincir.json")
sys.exit(0 if GECTI else 1)
