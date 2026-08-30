"""HAREKETLI HEDEF: 2. sira esigi gun gectikce sertlesiyor -- yansit.

Kullanicinin uyarisi: "2.lik gun gectikce zorlasiyor". Dogru ve simdiye
kadarki tum hesaplarimiz BUGUNKU esige gore yapildi. Nihai siralama
1 Eylul 23:59 UTC'deki PRIVATE tabloya gore belirlenir; rakipler o ana
kadar iyilesmeye devam eder.

GOZLENEN 2. SIRA ESIGI:
    ~29 Agu          0.99940
    30 Agu 07:51     0.99790
    30 Agu 17:26     0.99614
    30 Agu 22:06     0.99556   (Abdulbaki Bayir yeni girdi)

Bu betik kaymayi modelleyip 2 Eylul 02:59'daki (yerel bitis) esigi kestirir
ve gereken toplam rho^2'yi ONA gore hesaplar. HICBIR DOSYA YAZILMAZ.
"""

import numpy as np

TABAN = 1.00202690
S = 0.1294
KAP_E = 0.05169627
SAB1 = 1.0046992296

# gozlenen esik gecmisi: (bitise kalan gun, esik)
GOZLEM = [(3.2, 0.99940), (2.8, 0.99790), (2.4, 0.99614), (2.2, 0.99556)]
t = np.array([g[0] for g in GOZLEM])
y = np.array([g[1] for g in GOZLEM])
print("GOZLENEN 2. SIRA ESIGI")
for a, b in GOZLEM:
    print(f"  bitise {a:.1f} gun kala: {b:.5f}")
egim = np.polyfit(t, y, 1)[0]  # gun basina degisim (t azalirken y azaliyor)
print(f"\ndogrusal egim: bitise dogru gunde {-egim:+.5f}")
print(f"  son 14 saatte: {(0.99556 - 0.99790) / (14 / 24):+.5f}/gun")

KALAN = 2.2  # 30 Agu 22:30 -> 2 Eylul 02:59
print(f"\nBitise {KALAN:.1f} gun var. Uc senaryo:\n")
SEN = {
    "iyimser  (kayma DURUR)": 0.0,
    "orta     (yarilanarak yavaslar)": 0.0020,
    "gozlenen hiz devam eder": 0.0040,
    "kotumser (hizlanir)": 0.0060,
}
print(f"{'senaryo':>34s} {'nihai esik':>11s} {'gereken rho^2':>14s} {'gereken |c|':>12s}")
NIHAI = {}
for ad, hiz in SEN.items():
    esik = 0.99556 - hiz * KALAN
    g = TABAN - esik**2
    NIHAI[ad] = (esik, g)
    print(f"{ad:>34s} {esik:11.5f} {g:14.5f} {np.sqrt(g) / S:12.3f}")

# --- P(2. sira) hareketli hedefle ---
rng = np.random.default_rng(11)
NS = 200000
mu, sg = np.log(0.57), (np.log(1.26) - np.log(0.17)) / (2 * 1.645)
c = np.exp(rng.normal(mu, sg, NS))
rho1 = c * S
rho_h = rng.normal(0, 0.25 * np.abs(rho1)[:, None], (NS, 3))
t2 = rho1**2 + (rho_h**2).sum(axis=1)
sk = np.sqrt(np.maximum(TABAN - t2, 1e-9))

print(f"\n{'senaryo':>34s} {'P(2. sira)':>11s} {'P(1. sira)':>11s}")
for ad, (esik, g) in NIHAI.items():
    # 1. sira esigi de kayar; ayni hizla varsay
    e1 = 0.99009 - (0.99556 - esik)
    print(f"{ad:>34s} {(sk < esik).mean():11.1%} {(sk < e1).mean():11.1%}")

print("\n" + "=" * 74)
print("D1'IN KARAR ESIKLERI -- hareketli hedefe gore GUNCELLENDI")
print()
print(f"{'senaryo':>34s} {'nihai esik':>11s} {'D1 su skoru vermeli':>21s}")
for ad, (esik, g) in NIHAI.items():
    r_ger = np.sqrt(g)  # toplam gereken; H1 tek basina saglarsa
    p1 = np.sqrt(max(SAB1 - 2 * KAP_E * r_ger, 1e-9))
    print(f"{ad:>34s} {esik:11.5f} {p1:21.5f}")
print()
print("YORUM: D1 skoru bu satirlardan HANGISININ altindaysa, o senaryoda")
print("  2. sira tutar. Ustundeyse o senaryoda tutmaz.")
print()
print("STRATEJIK SONUC:")
print("  Hareketli hedef, ELIMIZDEKI HER KAZANIMI daha degerli kiliyor.")
print("  Ozellikle: (a) taban iyilestirmesi (sigma) artik marjinal degil,")
print("  (b) 5. yon eklemek artik 'degmez' degil, (c) erken bitirmek")
print("  rakiplere zaman birakmaz ama esigi de dondurmaz -- nihai tablo")
print("  bitiste belirlenir, bu yuzden ERKEN BITIRMEK ESIGI KURTARMAZ.")
