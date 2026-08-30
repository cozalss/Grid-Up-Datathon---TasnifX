"""IKI RAKIP MODELIN KARSILASTIRMASI ve birlesik beklenti.

MODEL A -- "|c| carpani".
  gerceklesen rho = |c| * ||BETA|| / 1.95,  |c| log-normal
  (medyan 0.57, %90 GA [0.17, 1.26]). ||BETA|| = 0.4832 (136 eksen).
  Bu modelde eksen EKLEMEK dogrudan rho'yu buyutur.

MODEL B -- "doyum".
  Yarim kalan bir blok-disi olcum su egriyi verdi (ayni blokta kurulup
  olculen bilesik icin):
      K=10 ongorulen 0.137 gerceklesen 0.126
      K=25 ongorulen 0.201 gerceklesen 0.149
      K=50 ongorulen 0.278 gerceklesen 0.123
      K=63 ongorulen 0.300 gerceklesen 0.124
  Yani GERCEKLESEN K~25'te doyuyor; eksen eklemek yalnizca TAHMINI sisirir.
  Bu modelde rho, ||BETA||'dan BAGIMSIZ olarak ~0.13-0.15'te tavan yapar.

ONEMLI GOZLEM: iki model MEDYANDA ANLASIYOR.
  Model A medyani: 0.57 * 0.4832 / 1.95 = 0.1412
  Model B araligi: 0.13 - 0.15
Anlasmazlik UST KUYRUKTA: Model A |c|=1.26 icin rho=0.312 diyor, Model B
bunu OLANAKSIZ sayiyor. Yani doyum dogruysa MEDYAN degismez ama 1. SIRA
OLASILIGI cok duser.

Bu betik ikisini de ve karisimlarini hesaplar. Hangi modelin dogru oldugu
n09_K_karari.json ile belirlenecek; o gelene kadar KARISIM en durustu.
"""

import json
import os

import numpy as np

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")
TABAN_MSE = 1.00202690323433
TAVAN = 1.95
S = 400000

with open(os.path.join(M29, "m148_demet.json"), encoding="utf-8") as fh:
    ONG = np.array(json.load(fh)["rho_k_tahmin"], dtype=float)
ONG_TOP = float(np.sqrt((ONG**2).sum()))

with open(os.path.join(M29, "n02_esik_tahmini.json"), encoding="utf-8") as fh:
    E = json.load(fh)
ESIK2 = float(E["rank2_tahmin_1eylul_2359UTC"]["NIHAI_TAHMIN"]["merkez"])
ESIK1 = float(E["rank1_tahmin_1eylul_2359UTC"]["NIHAI_TAHMIN"]["merkez"])
Z2 = E["rank2_tahmin_1eylul_2359UTC"]["NIHAI_TAHMIN"]["80pct_araligi_zarf"]
Z1 = E["rank1_tahmin_1eylul_2359UTC"]["NIHAI_TAHMIN"]["80pct_araligi_zarf"]

rng = np.random.default_rng(37)


def esikler():
    """Esikleri de belirsiz ornekle -- nokta tahmin gibi davranmak sisirir."""
    s2 = (float(Z2[1]) - float(Z2[0])) / (2 * 1.2816)
    s1 = (float(Z1[1]) - float(Z1[0])) / (2 * 1.2816)
    return rng.normal(ESIK1, s1, S), rng.normal(ESIK2, s2, S)


def model_a():
    mu = np.log(0.57)
    sd = (np.log(1.26) - np.log(0.17)) / (2 * 1.6449)
    return np.exp(rng.normal(mu, sd, S)) * ONG_TOP / TAVAN


def model_b():
    """Doyum: rho ~ olculen 0.126-0.149 araliginda, ||BETA||'dan bagimsiz.

    Gozlenen dort deger (0.126, 0.149, 0.123, 0.124) ort 0.1305, sd 0.0125.
    Blok-disi belirsizligi hesaba katmak icin sd'yi iki katina cikariyoruz.
    NEGATIF rho olanaksiz degil (isaret yanlis olabilir) ama bloklara
    bolme isareti LB'de duzelttigi icin alt sinir 0'da kirpilir.
    """
    return np.clip(rng.normal(0.1305, 0.025, S), 0.0, None)


e1, e2 = esikler()
print(f"esikler: 1. sira {ESIK1:.5f} {Z1}   2. sira {ESIK2:.5f} {Z2}")
print(f"||BETA|| = {ONG_TOP:.4f}\n")

print(
    f"{'model':>28s} {'medyan rho':>11s} {'medyan skor':>12s} "
    f"{'P(1.)':>7s} {'P(2.)':>7s} {'P(ilk3)':>8s}"
)
SONUC = {}
for ad, rho, agirlik in [
    ("A |c| carpani", model_a(), None),
    ("B doyum", model_b(), None),
    ("KARISIM %50/%50", None, 0.5),
]:
    if rho is None:
        ra, rb = model_a(), model_b()
        sec = rng.random(S) < agirlik
        rho = np.where(sec, ra, rb)
    sk = np.sqrt(np.maximum(TABAN_MSE - rho**2, 1e-9))
    p1 = float((sk <= e1).mean())
    p2 = float((sk <= e2).mean())
    p3 = float((sk <= 0.99927).mean())
    SONUC[ad] = {
        "medyan_rho": float(np.median(rho)),
        "medyan_skor": float(np.median(sk)),
        "P_1": p1,
        "P_2": p2,
        "P_3": p3,
        "p10_skor": float(np.quantile(sk, 0.10)),
        "p90_skor": float(np.quantile(sk, 0.90)),
    }
    print(
        f"{ad:>28s} {np.median(rho):11.4f} {np.median(sk):12.5f} "
        f"{100 * p1:6.1f}% {100 * p2:6.1f}% {100 * p3:7.1f}%"
    )

print("\nYORUM")
print("  Iki model MEDYANDA anlasiyor (rho ~ 0.13-0.14, skor ~ 0.991).")
print("  Anlasmazlik UST KUYRUKTA: doyum dogruysa 1. sira olasiligi coker,")
print("  ama 3. sira ve ustu olasiligi YINE DE yuksek kalir.")
print("  Her iki modelde de en kotu durum ~1.00101 (saf span) -- yani")
print("  bugunku 1.00115 yedegimizden IYI. Plan asagi yonlu korumali.")

with open(os.path.join(M29, "n13_iki_model.json"), "w", encoding="utf-8") as fh:
    json.dump({"beta": ONG_TOP, "esik_1": ESIK1, "esik_2": ESIK2, "modeller": SONUC}, fh, indent=1)
print("\n-> n13_iki_model.json")
