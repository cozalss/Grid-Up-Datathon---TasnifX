"""BIRLESIK BEKLENTI -- n10 (|c|) + n11 (K egrisi) + blok kazanci.

n13 iki RAKIP model tutuyordu. Artik ikisi de olculdu ve CELISMIYORLAR;
farkli seyleri olcuyorlar. Dogru birlesim su:

  1) SEVIYE CAPASI -- n10.
     |c| LB'nin kendi 29 olcumu uzerinde birak-birini-disarida ile
     olculdu: 0.434, %90 GA [0.184, 0.798]. Bu, span yonundeki
     korelasyonu DIK yondeki gercek korelasyona baglar:
         rho = |c| * ||BETA|| / 1.95
     Bu formul ||BETA||'da DOGRUSALDIR.

  2) K DUZELTMESI -- n11.
     Formulun dogrusalligi K buyudukce BOZULUYOR. yaz25 icinde zaman
     bolmesiyle olculdu:  K=25 -> rho 0.1642, K=136 -> rho 0.1226.
     Yani K=25, K=136'nin 1.339 kati. %95 AO [1.12, 1.56], P(iyi)=1.00.
     (guz25/kis26 GECERSIZ vekil: rho_s ile isaret uyumu 0.49 ve 0.30 --
      kis26'da korelasyon -0.426, yani TERS. Test ufku Nis-Tem 2026,
      yaz25 Nis-Tem 2025, ayni mevsim.)

     BIRLESIM: seviyeyi K=136'da |c| ile capala, sonra K oranini uygula.
         rho(25) = |c| * 0.4832 / 1.95 * 1.339
     Neden K=136'da capaliyoruz: |c| gonderim-farki yonlerinde olculdu ve
     onlarin dik payi kucuk (en yuksek 0.344); hangi K rejimine denk
     dustugu bilinmiyor. K=136 capasi MUHAFAZAKAR olan taraf -- eger |c|
     aslinda K=25 rejimine aitse rho'yu OLDUGUNDAN KUCUK tahmin ederiz.

  3) BLOK KAZANCI -- OLCULMEDI.
     m148 tek bilesik gondermiyor; BETA'yi 4 dik bloga bolup her birini
     AYRI olcuyor. Cauchy-Schwarz:
         toplam_k rho_k^2 = rho_bilesik^2 / cos^2(theta)
     theta = gercek rho vektoru ile ongordugumuz agirliklar arasindaki
     aci. YONU kesin, BUYUKLUGU bilinmiyor. cos^2 ~ U(0.35, 0.95)
     alinmistir; 4 boyutta rasgele bir yon icin E[cos^2] = 0.25,
     kusursuz tahminde 1.0.

     Bu kalem ACIKCA AYRI raporlanir -- olculmus sayilarla karistirilmaz.
"""

import json
import os

import numpy as np

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")
TABAN_MSE = 1.00202690323433
TAVAN = 1.95
S = 400000
BETA_136 = 0.4832  # genis span bilesiginin ongorulen rho'su (K=136)

rng = np.random.default_rng(101)

with open(os.path.join(M29, "n10_c_carpani.json"), encoding="utf-8") as fh:
    _C = json.load(fh)["c_nihai"]
C_MID = float(_C["nokta"])
C_LO, C_HI = (float(x) for x in _C["ga90"])

with open(os.path.join(M29, "n02_esik_tahmini.json"), encoding="utf-8") as fh:
    E = json.load(fh)
ESIK2 = float(E["rank2_tahmin_1eylul_2359UTC"]["NIHAI_TAHMIN"]["merkez"])
ESIK1 = float(E["rank1_tahmin_1eylul_2359UTC"]["NIHAI_TAHMIN"]["merkez"])
Z2 = [float(x) for x in E["rank2_tahmin_1eylul_2359UTC"]["NIHAI_TAHMIN"]["80pct_araligi_zarf"]]
Z1 = [float(x) for x in E["rank1_tahmin_1eylul_2359UTC"]["NIHAI_TAHMIN"]["80pct_araligi_zarf"]]
e1 = rng.normal(ESIK1, (Z1[1] - Z1[0]) / (2 * 1.2816), S)
e2 = rng.normal(ESIK2, (Z2[1] - Z2[0]) / (2 * 1.2816), S)

# 1) seviye
c = np.exp(rng.normal(np.log(C_MID), (np.log(C_HI) - np.log(C_LO)) / (2 * 1.6449), S))
rho136 = c * BETA_136 / TAVAN
# 2) K duzeltmesi: 1.339, %95 AO [1.12, 1.56] -> normal yaklasimi
kK = rng.normal(1.339, (1.56 - 1.12) / (2 * 1.96), S)
kK = np.clip(kK, 0.5, None)
rho25 = rho136 * kK
# 3) blok kazanci
cos2 = rng.uniform(0.35, 0.95, S)

print(f"|c|      = {C_MID:.3f}  %90 GA [{C_LO:.3f}, {C_HI:.3f}]   (n10, OLCULDU)")
print("K orani  = 1.339  %95 AO [1.12, 1.56]              (n11, OLCULDU)")
print("cos^2    ~ U(0.35, 0.95)                           (OLCULMEDI)")
print(f"esikler  : 1. sira {ESIK1:.5f}   2. sira {ESIK2:.5f}   (n02, TAHMIN)\n")

print(
    f"{'yapilandirma':>34s} {'medyan rho':>11s} {'medyan skor':>12s} "
    f"{'P(1.)':>7s} {'P(2.)':>7s} {'P(ilk3)':>8s}"
)
SON = {}
for ad, t2 in [
    ("K=136, blok kazanci YOK", rho136**2),
    ("K=25,  blok kazanci YOK", rho25**2),
    ("K=136, blok kazanci VAR", rho136**2 / cos2),
    ("K=25,  blok kazanci VAR", rho25**2 / cos2),
]:
    t2 = np.minimum(t2, TABAN_MSE - 1e-6)
    sk = np.sqrt(np.maximum(TABAN_MSE - t2, 1e-9))
    SON[ad] = {
        "medyan_rho": float(np.median(np.sqrt(t2))),
        "medyan_skor": float(np.median(sk)),
        "p10": float(np.quantile(sk, 0.10)),
        "p90": float(np.quantile(sk, 0.90)),
        "P_1": float((sk <= e1).mean()),
        "P_2": float((sk <= e2).mean()),
        "P_3": float((sk <= 0.99927).mean()),
    }
    print(
        f"{ad:>34s} {np.median(np.sqrt(t2)):11.4f} {np.median(sk):12.5f} "
        f"{100 * SON[ad]['P_1']:6.1f}% {100 * SON[ad]['P_2']:6.1f}% "
        f"{100 * SON[ad]['P_3']:7.1f}%"
    )

print("\n  En kotu durum her yapilandirmada 1.00101 (saf span, toplam rho=0)")
print("  -- bugunku 1.00115 yedegimizden IYI. Plan asagi yonlu korumali.")
print("\n  'blok kazanci VAR' satirlari OLCULMEMIS bir onsele dayanir;")
print("  yonu kesin (Cauchy-Schwarz), buyuklugu degil. Karar verirken")
print("  'blok kazanci YOK' satirlarini TABAN kabul et.")

with open(os.path.join(M29, "n14_birlesik_beklenti.json"), "w", encoding="utf-8") as fh:
    json.dump(
        {
            "c": [C_MID, C_LO, C_HI],
            "K_orani": [1.339, 1.12, 1.56],
            "beta_136": BETA_136,
            "esik_1": ESIK1,
            "esik_2": ESIK2,
            "yapilandirmalar": SON,
        },
        fh,
        indent=1,
    )
print("\n-> n14_birlesik_beklenti.json")
