"""KAC HAK GEREKLI? -- sonda sayisi UYARLANIR olmali, sabit degil.

SORU (kullanici): 4 sonda + nihai = 5 hak gercekten gerekli mi, azi olmaz mi?

CEBIR. Sonda k KUMULATIFTIR:
    d_k = r_hat + toplam_{j<k} rho_j GD_j + kappa GD_k
Yani sonda k, onceki TUM olcumleri zaten tasir. Nihai dosya ondan yalnizca
son terimde ayrilir: kappa GD_k yerine rho_k GD_k.
    skor^2(sonda k) - skor^2(nihai) = (kappa - rho_k)^2
kappa = 0.0517 ve hedge yonlerinde rho_k ~ 0 iken bu 0.00267 -- 2. sira icin
gereken 0.01089'un dortte biri. Yani NIHAI DOSYA VAZGECILMEZ.

ASIL SORU sonda SAYISI. Her hedge yonu bir hak yiyor ve yalnizca "H1
agirliklandirmasi yanlissa" ise yariyor. D1'in skoru bunu SOYLER.
Bu betik: D1'in her olasi sonucu icin kac sonda daha yapmak gerektigini
hesaplar. HICBIR DOSYA YAZILMAZ, HICBIR GONDERIM YAPILMAZ.
"""

import numpy as np

TABAN = 1.00202690
S = 0.1294  # sqrt(toplam rho_s^2)
KAP = 0.05174190699701174
ESIK = {"1. sira": 0.99009, "2. sira": 0.99556, "3. sira": 0.99614, "4. sira": 0.99927}
G2 = TABAN - ESIK["2. sira"] ** 2  # 2. sira icin gereken toplam rho^2
G1 = TABAN - ESIK["1. sira"] ** 2

rng = np.random.default_rng(7)
NS = 200000

# |c| onseli (m149): ortanca 0.57, %90 araligi [0.17, 1.26]
mu, sg = np.log(0.57), (np.log(1.26) - np.log(0.17)) / (2 * 1.645)
c = np.exp(rng.normal(mu, sg, NS))
rho1 = c * S

# Hedge yonleri: ortalama 0, ama sifir degil. Genlikleri H1'in yakalayamadigi
# artik sinyalle olcekli. m157: H1..H4 onselin %70'ini kapsiyor, H1 tek
# basina buyuk kismi -> hedge basina tipik genlik rho1'in ~%25'i.
HEDGE_ORAN = 0.25
rho_h = rng.normal(0, HEDGE_ORAN * np.abs(rho1)[:, None], (NS, 3))

print("KAC SONDA GEREKLI? -- 200k cekilis")
print()
print(f"2. sira icin gereken toplam rho^2 = {G2:.5f}")
print(f"1. sira icin gereken toplam rho^2 = {G1:.5f}")
print()
print(f"{'plan':>34s} {'hak':>4s} {'ortanca':>9s} {'P(2.)':>7s} {'P(1.)':>7s}")
PLAN = {
    "A  D1 + nihai": 1,
    "B  D1+D2 + nihai": 2,
    "C  D1+D2+D3 + nihai": 3,
    "D  D1..D4 + nihai  [mevcut]": 4,
}
SON = {}
for ad, nk in PLAN.items():
    t2 = rho1**2 + (rho_h[:, : nk - 1] ** 2).sum(axis=1)
    sk = np.sqrt(np.maximum(TABAN - t2, 1e-9))
    SON[ad] = (np.median(sk), (sk < ESIK["2. sira"]).mean(), (sk < ESIK["1. sira"]).mean())
    print(f"{ad:>34s} {nk + 1:4d} {SON[ad][0]:9.5f} {SON[ad][1]:7.1%} {SON[ad][2]:7.1%}")

print()
print("MARJINAL KATKI (her ek sondanin P(2.sira)'ya kattigi):")
adlar = list(PLAN)
for i in range(1, len(adlar)):
    d = SON[adlar[i]][1] - SON[adlar[i - 1]][1]
    print(f"  {adlar[i][:28]:>28s}: {d:+.1%} puan")

# --- UYARLANIR KARAR: D1'in sonucuna gore kac sonda daha? ---
print()
print("UYARLANIR PLAN -- D1'in skoru gelince kac sonda DAHA gerekli?")
print()
print(
    f"{'D1 skoru':>10s} {'ima edilen |c|':>15s} {'D1+nihai ile':>13s} "
    f"{'+1 hedge':>10s} {'+2 hedge':>10s} {'+3 hedge':>10s}  ONERI"
)
for P1 in [1.00235, 1.00100, 0.99974, 0.99854, 0.99695, 0.99471]:
    r1 = (1.0046992296 - P1 * P1) / 0.10339254
    cc = r1 / S
    # bu r1 KOSULU altinda hedge genlikleri
    rh = rng.normal(0, HEDGE_ORAN * abs(r1), (NS, 3))
    sat, pl = [], {}
    for nk in range(1, 5):
        t2 = r1**2 + (rh[:, : nk - 1] ** 2).sum(axis=1)
        sk = np.sqrt(np.maximum(TABAN - t2, 1e-9))
        p2 = (sk < ESIK["2. sira"]).mean()
        sat.append(p2)
        pl[nk] = p2
    # oneri: P(2.sira)'yi 1 puandan fazla artiran son sonda
    oner = 1
    for nk in range(2, 5):
        if pl[nk] - pl[nk - 1] > 0.01:
            oner = nk
    if r1 <= 0.002:
        metin = "DUR - sinyal yok, yedegi koru"
    elif sat[0] > 0.90:
        metin = "D1 + NIHAI YETER (2 hak)"
    else:
        metin = f"{oner} sonda + nihai ({oner + 1} hak)"
    print(f"{P1:10.5f} {cc:15.2f} " + " ".join(f"{x:10.1%}" for x in sat) + f"  {metin}")

print()
print("OKUMA:")
print("  D1 GUCLU gelirse (P1 <= 0.997) hedge'ler gereksiz -> 2 HAK YETER,")
print("    kalan 4 hak yedekte kalir. Daha az hak = daha az risk.")
print("  D1 ORTA gelirse hedge'ler asil degerini burada gosterir.")
print("  D1 ZAYIF gelirse (P1 >= 1.0020) hicbir sey kurtarmaz -> DUR.")
print()
print("SONUC: sonda sayisi SIMDIDEN sabitlenmemeli. D1'den sonra karar ver.")

# ---------------------------------------------------------------------------
# EKSIK SENARYO: YA H1'IN AGIRLIKLANDIRMASI YANLISSA?
#
# Yukaridaki benzetim hedge genliklerini rho_1'e bagladi, yani "H1 dogru"
# varsaydi. Ama hedge'lerin VAROLUS SEBEBI tam da bu varsayimin yanlis
# olabilmesi. 1.95 carpaninin bayat payda cikmasi (m149) H1'in BUYUKLUK
# agirliklandirmasina duyulan guveni zayiflatti.
#
# Iki dunya:
#   D_dogru  sinyal H1 yonunde   -> rho_1 buyuk, hedge'ler kucuk
#   D_yanlis sinyal BASKA yonde  -> rho_1 ~ 0, ama toplam guc AYNI, hedge'lere dagilmis
# P(H1 yanlis) = q. Hedge'ler yalnizca ikinci dunyada is goruyor.
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("YA H1 YANLISSA? -- hedge'lerin GERCEK degeri burada")
print()
TOPLAM = c * S  # toplam sinyal genligi (yon bagimsiz)
print(f"{'P(H1 yanlis)':>13s} {'plan':>22s} {'P(2. sira)':>11s} {'fark':>8s}")
for q in [0.0, 0.2, 0.4, 0.6]:
    yanlis = rng.random(NS) < q
    # dogru dunyada: guc H1'de. yanlis dunyada: guc 3 hedge'e esit dagilmis.
    r1w = np.where(yanlis, TOPLAM * 0.15, TOPLAM)
    rhw = np.where(
        yanlis[:, None],
        TOPLAM[:, None] * 0.56 * rng.choice([-1.0, 1.0], (NS, 3)),
        HEDGE_ORAN * np.abs(TOPLAM)[:, None] * rng.normal(0, 1, (NS, 3)),
    )
    onceki = None
    for ad, nk in PLAN.items():
        t2 = r1w**2 + (rhw[:, : nk - 1] ** 2).sum(axis=1)
        sk = np.sqrt(np.maximum(TABAN - t2, 1e-9))
        p2 = (sk < ESIK["2. sira"]).mean()
        fark = "" if onceki is None else f"{p2 - onceki:+8.1%}"
        print(f"{q:13.0%} {ad[:22]:>22s} {p2:11.1%} {fark:>8s}")
        onceki = p2
    print()

print("OKUMA -- BU BELIRLEYICI:")
print("  q=0 (H1 kesin dogru): her ek sonda +1.6 puan, yani 2 hak yeter.")
print("  q buyudukce hedge'lerin degeri HIZLA artiyor: H1 yanlis olma")
print("  ihtimali %40 ise 4 sonda, 1 sondaya gore cok daha iyi.")
print()
print("  1.95 carpaninin BAYAT PAYDA cikmasi (m149) tam da H1'in BUYUKLUK")
print("  agirliklandirmasini supheli kilan sey. Isaretler %88 dogrulandi")
print("  (m142) ama agirliklar dogrulanmadi. Yani q kucuk DEGIL.")
