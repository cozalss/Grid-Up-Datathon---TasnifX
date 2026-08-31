"""NIHAI BEKLENTI -- artik her kalem OLCULDU. n13/n14'un yerini alir.

n14'te "blok kazanci" OLCULMEMIS bir onseldi (cos^2 ~ U(0.35,0.95)).
n09 onu OLCTU. Onsel iyi kalibre cikti (ima ettigi rho kazanci 1.03-1.69,
medyan ~1.28; olculen 1.271) ama artik tahmine gerek yok.

OLCULEN KALEMLER
  |c| = 0.434, %90 GA [0.184, 0.798]        n10, LB'nin 29 olcumu, LOO
  K orani (B=1, K=25 / K=136) = 1.391       n09  (0.1427 / 0.1026)
  Blok kazanci (B=4 / B=1, K=25) = 1.271    n09  (0.1814 / 0.1427)
  CARPAN (blok korelasyonu -> LB rho) = 0.798   m148, n=1 KALIBRASYON

IKI BAGIMSIZ YOL, IKI FARKLI CEVAP -- ikisini de veriyoruz:

  YOL A (|c| capasi, LB'ye dayali):
      rho_LB = |c| * ||BETA_136|| / 1.95 * K_orani * blok_kazanci
             = |c| * 0.4832 / 1.95 * 1.391 * 1.271  =  |c| * 0.4382
      |c| = 0.434 -> rho = 0.190

  YOL B (CARPAN capasi, blok olcumune dayali):
      rho_LB = 0.1814 * 0.798 = 0.145

Fark iki kata yakin ve KAPATILAMIYOR. Sebebi acik: CARPAN = 0.798 m148'in
kendi belgesinde "n=1 kalibrasyon" diye niteleniyor, |c| = 0.434 ise 19
noktalik bir LOO'dan geliyor ama GONDERIM FARKI yonlerinde olculdu (dik
paylari kucuk, en yuksek 0.344) -- bizim demet yonlerimizin dik payi ~1.

MUHAFAZAKAR OLAN YOL B'DIR; taban senaryo olarak onu aliyoruz ve YOL A'yi
ust senaryo olarak gosteriyoruz. GERCEK KALIBRASYON ILK SONDA OLCULDUGUNDE
YAPILACAK -- o zaman iki yoldan hangisinin dogru oldugu da anlasilacak.
"""

import json
import os

import numpy as np

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")
TABAN_MSE = 1.00202690323433
S = 400000

BETA_136 = 0.4832
K_ORANI = 1.391  # n09: B=1'de K=25 / K=136
BLOK_KAZANCI = 1.271  # n09: K=25'te B=4 / B=1
RHO_BLOK_B4 = 0.1814  # n09: K=25, B=4, blok korelasyon birimi
# CARPAN. n18 (31 Agu 08:30) bunu ACTI:
#     CARPAN = rho_dik^LB/kor = (rho_dik/rho_s) * (rho_s/kor) = |c| * T
# seviye ekseni icin |c| = 1.9864, T = 0.4016, carpim 0.7977 ~ 0.798.
# YANI 0.798IN ICINDE 1.95 GOMULU -- ve n10 tam o degeri reddetmisti
# (P(|c| >= 1.95) = 0.0004). Ustelik UC YOLUN UCU DE ayni |c| bacagina
# dayaniyor: BAGIMSIZ KANIT SAYISI UCTEN BIRE dusuyor.
#
# IKI CAPA CELISIYOR ve hicbiri digerini baskilamiyor:
#   |c| = 1.986  n=1   ama DOGRU nesnede (OZNITELIK EKSENI, seviye)
#   |c| = 0.434  n=19  ama FARKLI nesnede (GONDERIM FARKI yonleri;
#                      n10 kendisi oznitelik eksenlerine tasindigi
#                      GOSTERILMEMISTIR diye uyariyor)
# Biri kucuk ornek, digeri yanlis nesne. 4.6 kat ayrisiyorlar.
# Kirilganlik kaniti: rho_s(seviye) belgelerde +0.0156 iken simdi -0.0153
# (ISARET DONMUS) ve T(seviye) LOO da %11 foldda isaret degistiriyor.
CARPAN = 0.798  # UST capa (|c|=1.986, n=1, dogru nesne)
CARPAN_DUZ = 0.166  # ALT capa (|c|=0.434 ile duzeltilmis, n=19, farkli nesne)

rng = np.random.default_rng(7)

with open(os.path.join(M29, "n10_c_carpani.json"), encoding="utf-8") as fh:
    _C = json.load(fh)["c_nihai"]
C_MID, (C_LO, C_HI) = float(_C["nokta"]), (float(x) for x in _C["ga90"])
C_LO, C_HI = (float(x) for x in _C["ga90"])

with open(os.path.join(M29, "n02_esik_tahmini.json"), encoding="utf-8") as fh:
    E = json.load(fh)


def _esik(ad):
    d = E[f"{ad}_tahmin_1eylul_2359UTC"]["NIHAI_TAHMIN"]
    z = [float(x) for x in d["80pct_araligi_zarf"]]
    return float(d["merkez"]), rng.normal(float(d["merkez"]), (z[1] - z[0]) / (2 * 1.2816), S)


# n17: 2. sira esigi 8 SAAT hic degismedi (1. sira 24 saattir sabit).
# n02'nin dun geceki tahmini kaymanin SURECEGINI varsayiyordu; durgunluk
# gozlemi merkezi 0.9897'den 0.99343'e TASIDI. Varsa n17 kullanilir.
_N17 = os.path.join(M29, "n17_esik_guncel.json")


ESIK1, e1 = _esik("rank1")
ESIK2, e2 = _esik("rank2")
if os.path.exists(_N17):
    with open(_N17, encoding="utf-8") as fh:
        _D17 = json.load(fh)
    ESIK2 = float(_D17["merkez"])
    _z = _D17["zarf"]
    # zarf UC SENARYONUN aralikidir (dogrusal / ussel / tamamen durdu),
    # bir guven araligi degil -- duzgun ornekleme icin ucgen dagilim.
    e2 = rng.triangular(float(_z[0]), ESIK2, float(_z[1]), S)
    # 1. SIRA: n02'nin egri tahminini KULLANMA -- lider 31 Agu 06:00'da
    # 0.99009'dan 0.98110'a SICRADI (tek gonderimde -0.00899). Gozlenen
    # deger tahminden iyidir. Bitiste daha da iyilesebilirler; kalan
    # ~1.7 gunde bir sicrama daha olasiligini p=0.35 aliyoruz.
    LIDER = 0.98110
    _sic = rng.random(S) < 0.35
    e1 = np.where(_sic, LIDER - rng.uniform(0.002, 0.009, S), LIDER)
    ESIK1 = float(np.median(e1))
    print(f"n17 kullanildi: 2. sira esigi {ESIK2:.5f} zarf [{_z[0]:.5f}, {_z[1]:.5f}]")
    print(f"1. sira GOZLENEN lider skoru {LIDER:.5f} (sicrama sonrasi), medyan esik {ESIK1:.5f}")
# TUTARLILIK: 1. sira esigi 2. sira esiginden BUYUK OLAMAZ. Ikisini bagimsiz
# ornekleyince 1. siranin genis zarfi (0.9776-0.99025) bazi orneklerde
# 2. siranin uzerine cikiyordu ve P(1.) > P(2.) gibi OLANAKSIZ bir sonuc
# veriyordu. Siralama zorlanir.
e1 = np.minimum(e1, e2)

# YOL A: |c| capasi
c = np.exp(rng.normal(np.log(C_MID), (np.log(C_HI) - np.log(C_LO)) / (2 * 1.6449), S))
rho_A = c * BETA_136 / 1.95 * K_ORANI * BLOK_KAZANCI
# YOL B: CARPAN capasi. n09'un B=4 olcumu %90 GA vermedi; iki bolme yonu
# 0.1478 (temiz) ile 0.1814 (ortalama) arasinda ayrisiyordu -> o araligi
# belirsizlik olarak kullaniyoruz.
rb = rng.uniform(0.1478, 0.1814, S) * CARPAN
rho_B = rb
# YOL C: |c| capasi ama K=25'te DOGRUDAN (n06_kappa.py'nin kullandigi yol).
#   rho_LB = |c| * ||BETA_25|| / 1.95 * blok_kazanci
# Yol A ile Yol C ARASINDAKI FARK, |c| formulunun hangi ||BETA||'da
# capalandigidir. Formul dogrusaldir ama n09 dogrusalligi CURUTTU:
#   ||BETA||  0.2141 -> 0.4788  (x2.24)
#   realized  0.1427 -> 0.1026  (x0.72)
# Yani realized ||BETA|| ile BUYUMUYOR, KUCULUYOR. O halde |c|'yi iki
# ucdan birine uygulamak keyfidir ve ikisi 3 KAT farkli cevap verir.
# Bu yuzden Yol A ve Yol C, |c| yaklasiminin BELIRSIZLIK SINIRLARIDIR,
# ayri tahminler degil.
BETA_25 = 0.2141
rho_C = c * BETA_25 / 1.95 * BLOK_KAZANCI

print(f"|c|           = {C_MID:.3f}  %90 GA [{C_LO:.3f}, {C_HI:.3f}]   (n10 OLCULDU)")
print(f"K orani B=1   = {K_ORANI}   (n09 OLCULDU)")
print(f"blok kazanci  = {BLOK_KAZANCI}   (n09 OLCULDU; n14'un onseli 1.28 idi)")
print(f"CARPAN        = {CARPAN}   (m148, n=1 KALIBRASYON -- zayif halka)")
print(f"esikler       : 1. sira {ESIK1:.5f}   2. sira {ESIK2:.5f}  (n02 TAHMIN)\n")

print(
    f"{'senaryo':>34s} {'medyan rho':>11s} {'medyan skor':>12s} "
    f"{'P(1.)':>7s} {'P(2.)':>7s} {'P(ilk3)':>8s}"
)
SON = {}
for ad, rho in [
    ("YOL Bd (CARPAN duz.) -- EN ALT", rng.uniform(0.1478, 0.1814, S) * CARPAN_DUZ),
    ("YOL C (|c| @ K=25)  -- ALT", rho_C),
    ("YOL B (CARPAN)      -- MERKEZ", rho_B),
    ("YOL A (|c| @ K=136) -- UST", rho_A),
]:
    t2 = np.minimum(rho**2, TABAN_MSE - 1e-6)
    sk = np.sqrt(np.maximum(TABAN_MSE - t2, 1e-9))
    SON[ad] = {
        "medyan_rho": float(np.median(rho)),
        "medyan_skor": float(np.median(sk)),
        "p10": float(np.quantile(sk, 0.10)),
        "p90": float(np.quantile(sk, 0.90)),
        "P_1": float((sk <= e1).mean()),
        "P_2": float((sk <= e2).mean()),
        "P_3": float((sk <= 0.99927).mean()),
    }
    print(
        f"{ad:>34s} {np.median(rho):11.4f} {np.median(sk):12.5f} "
        f"{100 * SON[ad]['P_1']:6.1f}% {100 * SON[ad]['P_2']:6.1f}% "
        f"{100 * SON[ad]['P_3']:7.1f}%"
    )

# --- ESIK SENARYOLARINA AYRISTIRMA ---------------------------------------
# Tek bir P(2.) yuzdesi yaniltici: sayinin buyuk kismi ESIK varsayimindan
# geliyor, skorumuzdan degil. Uc esik senaryosunu ACIKCA ayiralim.
print("\nESIK SENARYOSUNA GORE P(2. sira):")
print(
    f"{'esik senaryosu':>34s} {'esik':>9s} {'gereken rho':>12s} {'alt':>7s} {'merkez':>7s} {'ust':>7s}"
)
for _ad, _e in [
    ("esik BUGUNKU yerinde kalirsa", 0.99536),
    ("egri tahmini (sicramasiz)", 0.99282),
    ("bir rakip 0.009 daha SICRARSA", 0.98382),
]:
    _ger = float(np.sqrt(max(TABAN_MSE - _e * _e, 0)))
    _pc = float((np.sqrt(np.maximum(TABAN_MSE - rho_C**2, 1e-9)) <= _e).mean())
    _pb = float((np.sqrt(np.maximum(TABAN_MSE - rho_B**2, 1e-9)) <= _e).mean())
    _pa = float((np.sqrt(np.maximum(TABAN_MSE - rho_A**2, 1e-9)) <= _e).mean())
    print(
        f"{_ad:>34s} {_e:9.5f} {_ger:12.4f} {100 * _pc:6.1f}% {100 * _pb:6.1f}% {100 * _pa:6.1f}%"
    )
print("  (alt = Yol C, merkez = Yol B / CARPAN, ust = Yol A)")

print("\n  En kotu durum 1.00101 (saf span) -- bugunku 1.00115 yedegimizden IYI.")
print("  ILK SONDA GELINCE bu tablo yeniden kurulacak: olculen rho_1,")
print("  YOL A ile YOL B arasindaki secimi de yapacak.")

with open(os.path.join(M29, "n16_nihai_beklenti.json"), "w", encoding="utf-8") as fh:
    json.dump(
        {
            "c": [C_MID, C_LO, C_HI],
            "K_orani": K_ORANI,
            "blok_kazanci": BLOK_KAZANCI,
            "rho_blok_B4": RHO_BLOK_B4,
            "carpan": CARPAN,
            "esik_1": ESIK1,
            "esik_2": ESIK2,
            "senaryolar": SON,
        },
        fh,
        indent=1,
    )
print("\n-> n16_nihai_beklenti.json")
