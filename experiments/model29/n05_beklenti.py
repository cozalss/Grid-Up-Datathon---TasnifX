"""GENIS SPAN kurulumunun beklenen sonucu ve siralama olasiligi.

Girdi varsayimlari (hepsi baska yerde OLCULMUS, burada yalnizca birlestiriliyor):
  TABAN_MSE = 1.00202690          (m148: M0 - 2kL + ||r_hat||^2)
  ongorulen rho blok basina       (m148 ciktisi, ||BETA_b||)
  |c| = 0.57, %90 GA [0.17, 1.26] (m149/m139 carpan kalibrasyonu)
  gerceklesen rho_b = |c| * ongorulen_b / 1.95
      -- 1.95 KATS icinde zaten var (KATS = 1.95*|rho_s|*isaret), bu yuzden
         geri bolunur. Kalibrasyon: ongorulen 0.2522 & |c|=0.57 -> 0.0738.
  esik tahmini n02_esik_tahmini.json'dan (bitis 1 Eylul 23:59 UTC)

BLOK BOLMESININ KAZANCI burada YOK SAYILIR (muhafazakar): bolme yalnizca
goreli agirliklandirmamiz yanlissa kazandirir, dogruysa esitlik verir.
Yani asagidaki sayilar ALT SINIRDIR.
"""

import json
import os

import numpy as np

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")
TABAN_MSE = 1.00202690323433
TAVAN = 1.95

with open(os.path.join(M29, "m148_demet.json"), encoding="utf-8") as fh:
    D = json.load(fh)
ONG = np.array(D["rho_k_tahmin"], dtype=float)
ONG_TOP = float(np.sqrt((ONG**2).sum()))

# --- esik ---------------------------------------------------------------
yol = os.path.join(M29, "n02_esik_tahmini.json")
ESIK2, ESIK1 = 0.9897, 0.9872  # yedek degerler; asil kaynak asagidaki dosya
ESIK2_ZARF = ESIK1_ZARF = None
if os.path.exists(yol):
    with open(yol, encoding="utf-8") as fh:
        E = json.load(fh)
    ESIK2 = float(E["rank2_tahmin_1eylul_2359UTC"]["NIHAI_TAHMIN"]["merkez"])
    ESIK1 = float(E["rank1_tahmin_1eylul_2359UTC"]["NIHAI_TAHMIN"]["merkez"])
    ESIK2_ZARF = E["rank2_tahmin_1eylul_2359UTC"]["NIHAI_TAHMIN"]["80pct_araligi_zarf"]
    ESIK1_ZARF = E["rank1_tahmin_1eylul_2359UTC"]["NIHAI_TAHMIN"]["80pct_araligi_zarf"]
    print(
        f"esik n02'den okundu: 2. sira {ESIK2:.5f} {ESIK2_ZARF}, 1. sira {ESIK1:.5f} {ESIK1_ZARF}"
    )
else:
    print(f"UYARI: {yol} yok -> yedek esikler {ESIK2}/{ESIK1}")

print(f"\nongorulen bloklar: {np.round(ONG, 4).tolist()}")
print(f"ongorulen toplam  = {ONG_TOP:.4f}")
print(f"taban MSE         = {TABAN_MSE:.8f}  -> sifir sinyalde skor {np.sqrt(TABAN_MSE):.5f}")

print(f"\n{'|c|':>6s} {'toplam rho':>11s} {'toplam rho^2':>13s} {'NIHAI SKOR':>11s}  yorum")
for c in [0.0, 0.17, 0.30, 0.45, 0.57, 0.70, 0.81, 1.00, 1.26]:
    rho = c * ONG_TOP / TAVAN
    t2 = rho * rho
    sk = np.sqrt(max(TABAN_MSE - t2, 1e-9))
    if sk <= ESIK1:
        y = "1. SIRA"
    elif sk <= ESIK2:
        y = "2. sira"
    elif sk <= 0.99927:
        y = "3. sira (bugunku esikle)"
    else:
        y = "yetersiz"
    print(f"{c:6.2f} {rho:11.4f} {t2:13.5f} {sk:11.5f}  {y}")

# --- |c| icin log-normal benzeri belirsizlikle olasilik ------------------
# %90 GA [0.17, 1.26] merkez 0.57 -> log uzayinda simetrige yakin
lo, mid, hi = 0.17, 0.57, 1.26
mu = np.log(mid)
sd = (np.log(hi) - np.log(lo)) / (2 * 1.6449)
rng = np.random.default_rng(11)
c = np.exp(rng.normal(mu, sd, 400000))
sk = np.sqrt(np.maximum(TABAN_MSE - (c * ONG_TOP / TAVAN) ** 2, 1e-9))
print(f"\n|c| log-normal: medyan {mid}, sigma_log {sd:.3f}")
print(
    f"beklenen skor  medyan {np.median(sk):.5f}   %10-%90 "
    f"[{np.quantile(sk, 0.10):.5f}, {np.quantile(sk, 0.90):.5f}]"
)
for ad, e in [("1. SIRA", ESIK1), ("2. sira", ESIK2), ("3. sira", 0.99927)]:
    print(f"  P({ad} <= {e:.5f}) = {100 * float((sk <= e).mean()):5.1f}%")

# --- ESIK BELIRSIZLIGI DE KATILIR ---------------------------------------
# Yukaridaki sayilar esigi KESIN biliyormus gibi davranir. Oysa esik de bir
# tahmin (n02'nin zarfi). Ikisini birlikte ornekleyerek daha durust bir
# olasilik elde ederiz. Zarf %80 aralik kabul edilip normal varsayilir.
P2_ZARF = P1_ZARF = None
if ESIK2_ZARF and ESIK1_ZARF:

    def _ornek(zarf, merkez):
        s_ = (float(zarf[1]) - float(zarf[0])) / (2 * 1.2816)  # %80 -> sigma
        return rng.normal(merkez, s_, sk.size)

    e2 = _ornek(ESIK2_ZARF, ESIK2)
    e1 = _ornek(ESIK1_ZARF, ESIK1)
    P2_ZARF = float((sk <= e2).mean())
    P1_ZARF = float((sk <= e1).mean())
    print("\nesik belirsizligi de katildiginda:")
    print(
        f"  P(1. SIRA) = {100 * P1_ZARF:5.1f}%   (kesin esikte {100 * float((sk <= ESIK1).mean()):.1f}%)"
    )
    print(
        f"  P(2. sira) = {100 * P2_ZARF:5.1f}%   (kesin esikte {100 * float((sk <= ESIK2).mean()):.1f}%)"
    )

with open(os.path.join(M29, "n05_beklenti.json"), "w", encoding="utf-8") as fh:
    json.dump(
        {
            "ongorulen_bloklar": ONG.tolist(),
            "ongorulen_toplam": ONG_TOP,
            "taban_mse": TABAN_MSE,
            "esik_2": ESIK2,
            "esik_1": ESIK1,
            "medyan_skor": float(np.median(sk)),
            "p10": float(np.quantile(sk, 0.10)),
            "p90": float(np.quantile(sk, 0.90)),
            "P_1": float((sk <= ESIK1).mean()),
            "P_2": float((sk <= ESIK2).mean()),
            "P_1_esik_belirsizligiyle": P1_ZARF,
            "P_2_esik_belirsizligiyle": P2_ZARF,
            "P_3": float((sk <= 0.99927).mean()),
        },
        fh,
        indent=1,
    )
print("\n-> n05_beklenti.json")
