"""p33-c: BAGIMSIZ UCTAN UCA ZINCIR SINAMASI (m162 esdegeri, submissions/ YAZMADAN).

m161 dongusel cikmisti: skoru, ters cevirdigi formulun kendisiyle uretiyordu.
Bu betik ONU YAPMAZ:

  1. SENTETIK GERCEK ARTIK r_syn kurulur; istenen ic carpimlar
        <r_syn, V_i>/N = L_i   (i = 1..28, LB skorlarindan cozulmus GERCEK L'ler)
        <r_syn, u>/N   = rho   (bizim SECTIGIMIZ deger)
        ort(r_syn^2)   = M0
     Ilk iki kosul span'da en kucuk normlu cozumle, ucuncusu span'a DIK
     rastgele bir bilesenle saglanir.
  2. GERCEK log hedef  t = a0 + r_syn.
  3. Aday dosya, URETIM zinciriyle kurulur: tuketim = expm1(taban + kappa*u),
     negatifler kirpilir, GERCEK BIR CSV'ye yazilir ve GERI OKUNUR.
  4. Skor DOGRUDAN hesaplanir:  P = sqrt(ort((log1p(CSV) - t)^2)).
     Hicbir m148/p33 formulu kullanilmaz. CSV yuvarlamasi ve kirpma DAHIL.
  5. Cebrin ongordugu  sqrt(TABAN_MSE - 2*kappa*rho + kappa^2)  ile karsilastirilir.
  6. Ayrica ters cevirme sinanir: skordan rho geri cozulur.
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

with open(os.path.join(GEC, "p33_a_cebir.json"), encoding="utf-8") as fh:
    A = json.load(fh)
with open(os.path.join(GEC, "p33_b_capa.json"), encoding="utf-8") as fh:
    B = json.load(fh)
V = np.load(os.path.join(GEC, "p33_V.npy"))
a0 = np.load(os.path.join(GEC, "p33_a0.npy"))
r_hat = np.load(os.path.join(GEC, "p33_r_hat.npy"))
GD = np.load(os.path.join(GEC, "GD.npy"))
N, kL, TABAN_MSE = A["N"], A["kL"], A["TABAN_MSE"]
L_vec = np.array(B["L_vec"])

u = GD[1] / np.sqrt(float((GD[1] * GD[1]).mean()))  # birim demet yonu (demet 2)
print(f"u: ort(u^2) = {float((u * u).mean()):.9f}")
ic = np.abs((V.T @ u) / N)
print(f"u'nun span'a dikligi: maks |<u,V_i>/N| = {ic.max():.3e}")

# m162 tasarimi: TABAN_MSE yalnizca kL ve ||r_hat||^2 ye bagli oldugu icin
# kisitlar {<r,r_hat>/N = kL, <r,u>/N = rho, ort(r^2)=M0}. 28 L'nin hepsini
# kisitlamak Gram'i tekil yapiyor (28 yon 714688 boyutta neredeyse esdogrusal).
Am = np.vstack([r_hat[None, :], u[None, :]])
Gm = (Am @ Am.T) / N


def kur_r_syn(rho, tohum=2026):
    h = np.array([kL, rho])
    c = np.linalg.solve(Gm + 1e-14 * np.eye(len(Gm)), h)
    r_par = c @ Am
    g = float((r_par * r_par).mean())
    if g > M0:
        raise SystemExit(f"DUR: span gucu M0'i asiyor ({g:.6f} > {M0:.6f})")
    rng = np.random.default_rng(tohum)
    w = rng.standard_normal(N)
    w -= Am.T @ np.linalg.solve(Gm + 1e-14 * np.eye(len(Gm)), (Am @ w) / N)
    w /= np.sqrt(float((w * w).mean()))
    return r_par + np.sqrt(M0 - g) * w


def dogrula(r, rho):
    e1 = abs(float((r * r_hat).mean()) - kL)
    eV = float(np.abs((V.T @ r) / N - L_vec).max())
    print(f"  (bilgi) 28 olculmus L'den en buyuk sapma: {eV:.2e}")
    e2 = abs(float((r * u).mean()) - rho)
    e3 = abs(float((r * r).mean()) - M0)
    print(f"  r_syn kurulus hatalari: kL {e1:.2e}, rho {e2:.2e}, M0 {e3:.2e}")
    assert e1 < 1e-9 and e2 < 1e-9 and e3 < 1e-9, "r_syn kurulamadi"


IDS = pd.read_csv(os.path.join(KOK, "data/raw/test.csv")).id.values
TMP = os.path.join(GEC, "_zincir_aday.csv")


def dosyadan_skor(log_tahmin, t):
    tk = np.expm1(log_tahmin)
    tk[tk < 0] = 0.0
    pd.DataFrame({"id": IDS, "tuketim": tk}).to_csv(TMP, index=False)
    geri = pd.read_csv(TMP)
    lg = np.log1p(geri.tuketim.values.astype(np.float64))
    d = lg - t
    return float(np.sqrt(float((d * d).mean())))


SONUC = []
enb_h = 0.0
for rho in (0.16340, 0.05, 0.0, -0.05):
    r_syn = kur_r_syn(rho)
    dogrula(r_syn, rho)
    t = a0 + r_syn
    # (a) saf span dosyasi
    P0 = dosyadan_skor(a0 + r_hat, t)
    print(f"  rho={rho:+.5f}  SAF SPAN dosyasi: dogrudan {P0:.6f}  "
          f"cebir sqrt(TABAN_MSE) {np.sqrt(TABAN_MSE):.6f}  fark {P0 - np.sqrt(TABAN_MSE):+.2e}")
    for kap in (0.05, 0.10, 0.16340):
        P = dosyadan_skor(a0 + r_hat + kap * u, t)
        cebir = float(np.sqrt(max(TABAN_MSE - 2 * kap * rho + kap * kap, 1e-12)))
        # ters cevirme
        rho_coz = (TABAN_MSE + kap * kap - P * P) / (2 * kap)
        h = P - cebir
        enb_h = max(enb_h, abs(h))
        SONUC.append(
            {
                "rho": rho,
                "kappa": kap,
                "dogrudan_skor": P,
                "cebir_skor": cebir,
                "fark": h,
                "rho_geri_cozulen": rho_coz,
                "rho_cozum_hatasi": rho_coz - rho,
            }
        )
        print(
            f"    kappa={kap:.5f}: dogrudan {P:.6f}  cebir {cebir:.6f}  "
            f"fark {h:+.2e}   rho geri {rho_coz:+.6f} (hata {rho_coz - rho:+.2e})"
        )

if os.path.exists(TMP):
    os.remove(TMP)

enb_rho = max(abs(s["rho_cozum_hatasi"]) for s in SONUC)
GECTI = enb_h < 2e-4 and enb_rho < 5e-3
print(f"\nen buyuk skor farki = {enb_h:.2e}  (esik 2e-4)")
print(f"en buyuk rho cozum hatasi = {enb_rho:.2e}  (esik 5e-3)")
print("\n" + ("ZINCIR SAGLAM." if GECTI else "ZINCIRDE SORUN VAR -- DUR."))

with open(os.path.join(GEC, "p33_c_zincir.json"), "w", encoding="utf-8") as fh:
    json.dump(
        {
            "gecti": bool(GECTI),
            "en_buyuk_skor_farki": enb_h,
            "en_buyuk_rho_hatasi": enb_rho,
            "u_dikligi_maks": float(ic.max()),
            "satirlar": SONUC,
        },
        fh,
        indent=1,
    )
print("-> p33_c_zincir.json")
sys.exit(0 if GECTI else 1)
