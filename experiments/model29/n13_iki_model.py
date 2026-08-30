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

# DIKKAT: m148_demet.json OYNAK bir dosyadir -- zincir sinamasi (m161) ve
# n07_temiz_kurulum onu git'ten geri aliyor. Bu betik bir kez ESKI dort
# yonlu surumu (||BETA|| = 0.2775) okuyup YANLIS sayi uretti. Bu yuzden
# okunan degeri DENETLIYORUZ: genis span kurulusunda dort blogun HEPSI
# gercek sinyal tasir (hepsi > 0.10); eski kurulusta ucu ~1e-16 idi.
with open(os.path.join(M29, "m148_demet.json"), encoding="utf-8") as fh:
    ONG = np.array(json.load(fh)["rho_k_tahmin"], dtype=float)
if os.environ.get("BETA_BLOKLAR"):
    ONG = np.array([float(x) for x in os.environ["BETA_BLOKLAR"].split(",")])
    print(f"BETA_BLOKLAR ortam degiskeninden alindi: {ONG.tolist()}")
if (np.abs(ONG) < 0.10).any():
    raise SystemExit(
        f"DUR: m148_demet.json ESKI kurulusu gosteriyor {np.round(ONG, 4).tolist()}.\n"
        "     Genis span kurulusunda dort blogun hepsi > 0.10 olmali.\n"
        "     Once n07_temiz_kurulum.py kos, ya da BETA_BLOKLAR=... ver."
    )
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


# |c| ONSELI. n10 bunu LB'nin KENDI 29 olcumu uzerinde birak-birini-disarida
# ile OLCTU (vekil blok kullanmadan) ve m149'un 0.57 [0.17, 1.26] degerini
# 0.43 [0.18, 0.80] ile degistirdi -- merkez daha dusuk, aralik daha dar.
# Ayni calisma sigma_L'yi de dogrudan olctu: G'nin 3 TAM SIFIR kipinde
# u'L = 0 olmak ZORUNDA, gozlenen sapma saf olcum hatasidir -> 2.94e-06,
# LB yuvarlamasinin 1.02 kati. m112'nin varsaydigi 2.27e-04 (77 kat buyuk)
# veriyle REDDEDILDI.
C_YOL = os.path.join(M29, "n10_c_carpani.json")
if os.path.exists(C_YOL):
    with open(C_YOL, encoding="utf-8") as fh:
        _C = json.load(fh)["c_nihai"]
    C_MID = float(_C["nokta"])
    C_LO, C_HI = (float(x) for x in _C["ga90"])
    print(f"|c| n10'dan OLCULDU: {C_MID:.3f} %90 GA [{C_LO:.3f}, {C_HI:.3f}]")
else:
    C_MID, C_LO, C_HI = 0.57, 0.17, 1.26
    print(f"UYARI: n10_c_carpani.json yok -> eski onsel {C_MID} [{C_LO}, {C_HI}]")


def model_a():
    mu = np.log(C_MID)
    sd = (np.log(C_HI) - np.log(C_LO)) / (2 * 1.6449)
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

# --- BLOK BOLMESI KAZANCI -------------------------------------------------
# Yukaridaki iki model de TEK BILESIGIN gerceklesen rho'sunu tahmin eder.
# Oysa m148 tek bilesik gondermiyor: BETA'yi 4 dik bloga bolup her blogun
# rho'sunu AYRI olcuyor ve OLCULEN degerlerle en iyi bilesimi kuruyor.
#
# CEBIR: bloklarda gercek korelasyon vektoru (rho_1..rho_B), bizim ongordugumuz
# goreli agirliklar w_k = ||BETA_k||/||BETA||. Tek bilesik su kadarini yakalar:
#     rho_bilesik = toplam_k w_k * rho_k
# Blok olcumuyle yakalanan ise:
#     toplam_k rho_k^2
# Cauchy-Schwarz: (toplam w_k rho_k)^2 <= toplam rho_k^2, esitlik ANCAK
# rho w ile ORANTILIYSA. Yani
#     toplam rho_k^2 = rho_bilesik^2 / cos^2(theta)
# theta = gercek rho vektoru ile ongordugumuz agirliklar arasindaki aci.
#
# cos^2(theta) NE OLABILIR? 4 boyutta TAMAMEN RASGELE bir yon icin
# E[cos^2] = 1/B = 0.25. Kusursuz tahminde 1.0. Bizim tahminimiz aile
# duzeyinde bir on bilgidir -- rasgeleden iyi, kusursuzdan uzak.
# Muhafazakar olarak cos^2 ~ Uniform(0.35, 0.95) aliyoruz; bu, kazanc
# carpaninin rho^2'de 1.05 ile 2.9 arasinda olmasi demektir.
#
# BU KAZANC OLCULMEMISTIR -- n09_K_karari.json'un "B blok" sutunu gelince
# gercek degerle degistirilecek. O zamana kadar burasi BELIRSIZ bir
# UST YON gostergesidir, kesin bir kazanc degil.
B_BLOK = 4
cos2 = rng.uniform(0.35, 0.95, S)
print(f"\nBLOK BOLMESI KAZANCI eklendiginde (B={B_BLOK}, cos^2 ~ U(0.35,0.95), OLCULMEDI):")
print(
    f"{'model':>28s} {'medyan rho':>11s} {'medyan skor':>12s} "
    f"{'P(1.)':>7s} {'P(2.)':>7s} {'P(ilk3)':>8s}"
)
for ad in ["A |c| carpani", "B doyum", "KARISIM %50/%50"]:
    if ad == "A |c| carpani":
        rc = model_a()
    elif ad == "B doyum":
        rc = model_b()
    else:
        ra, rb = model_a(), model_b()
        rc = np.where(rng.random(S) < 0.5, ra, rb)
    t2 = np.minimum(rc**2 / cos2, TABAN_MSE - 1e-6)  # tavan: skor >= 0
    sk = np.sqrt(np.maximum(TABAN_MSE - t2, 1e-9))
    SONUC[ad + " +blok"] = {
        "medyan_skor": float(np.median(sk)),
        "P_1": float((sk <= e1).mean()),
        "P_2": float((sk <= e2).mean()),
        "P_3": float((sk <= 0.99927).mean()),
    }
    print(
        f"{ad:>28s} {np.median(np.sqrt(t2)):11.4f} {np.median(sk):12.5f} "
        f"{100 * float((sk <= e1).mean()):6.1f}% "
        f"{100 * float((sk <= e2).mean()):6.1f}% "
        f"{100 * float((sk <= 0.99927).mean()):7.1f}%"
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
