"""TEK BILINMEYEN. n16'nin dort "yolu" aslinda tek parametrenin dort degeri.

n18, uc yolun da ayni |c| bacagina dayandigini gosterdi. Bunu sonuna kadar
goturunce cebir sadelesiyor:

    gerceklesen rho_LB = |c| * rho_s(bilesik)

ve bilesigin rho_s'i DOGRUDAN bilinir. KATS[i] = 1.95*|rho_s_i| oldugu icin
    rho_s(bilesik) = ||BETA|| / 1.95 = 0.2141 / 1.95 = 0.1098

Dolayisiyla dort "yol" yalnizca |c|'ye atanan dort farkli degerdir:

    YOL Bd  |c| = 0.25   (CARPAN 0.166 / T_seviye 0.402'ye denk)
    YOL C   |c| = 0.434  (n10, n=19)
    YOL B   |c| = 1.32   (CARPAN 0.798 -> rho 0.145'e denk)
    YOL A   |c| = 1.73   (K=136 capasi)

IKI CAPA, IKI KUSUR:
    |c| = 1.986   n=1    DOGRU nesnede  (seviye, bir OZNITELIK EKSENI --
                         bizim demet yonlerimizle ayni turden)
    |c| = 0.434   n=19   FARKLI nesnede (gonderim FARKI yonleri; n10 kendi
                         raporunda "oznitelik eksenlerine tasindigi
                         GOSTERILMEMISTIR" diye uyariyor)

Biri kucuk ornek, digeri yanlis nesne. Hicbiri digerini baskilamiyor ve
4.6 kat ayrisiyorlar. Bu, masa basinda kapatilamaz.

>>> ONEMLI: ILK SONDA (D1) TAM OLARAK BU SAYIYI OLCER. <<<
D1'in yonu bir OZNITELIK EKSENI bilesigidir -- yani |c|'nin bizim
kullandigimiz nesnedeki degerini dogrudan verir:
    |c| = 1.95 * rho_1_olculen / ongorulen_1     (ongorulen_1 = 0.1302)
Tek gonderim, alti katlik belirsizligi kapatir.
"""

import json
import os

import numpy as np

M29 = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX/experiments/model29"
TABAN_MSE = 1.00202690323433
RHO_S_BILESIK = 0.2141 / 1.95  # = 0.10979
YEDEK = 1.00115
SAF_SPAN = float(np.sqrt(TABAN_MSE))

print(f"rho_s(bilesik) = ||BETA||/1.95 = 0.2141/1.95 = {RHO_S_BILESIK:.5f}")
print(f"saf span skoru (rho=0)        = {SAF_SPAN:.5f}")
print(f"yedegimiz                     = {YEDEK:.5f}\n")

print(f"{'|c|':>7s} {'kaynak':>34s} {'rho':>8s} {'NIHAI SKOR':>11s}  siralama")
SATIR = []
for c, kaynak in [
    (0.184, "n10 %90 GA ALT ucu"),
    (0.434, "n10 nokta tahmini (n=19, farkli nesne)"[:34]),
    (0.798, "n10 %90 GA UST ucu"),
    (1.320, "CARPAN 0.798'in ima ettigi"),
    (1.986, "seviye ekseni (n=1, DOGRU nesne)"),
]:
    rho = c * RHO_S_BILESIK
    sk = float(np.sqrt(max(TABAN_MSE - rho * rho, 1e-9)))
    if sk <= 0.98110:
        s = "1. SIRA"
    elif sk <= 0.99536:
        s = "2. sira"
    elif sk <= 0.99886:
        s = "3.-4."
    elif sk <= YEDEK:
        s = "kucuk kazanc"
    else:
        s = "YEDEGI KULLAN"
    print(f"{c:7.3f} {kaynak:>34s} {rho:8.4f} {sk:11.5f}  {s}")
    SATIR.append({"c": c, "kaynak": kaynak, "rho": rho, "skor": sk, "sira": s})

print("\nOKUNUSU")
print("  Her sey TEK sayiya bagli ve o sayinin iki olcumu 4.6 kat ayrisiyor.")
print("  Alt ucta kazanc yok denecek kadar az; ust ucta 1. sira.")
print("  Ama HER durumda skor <= 1.00101 < 1.00115 (yedek), yani")
print("  GONDERIM BIR SEY KAYBETTIRMEZ -- yalnizca kazandirir ya da kazandirmaz.")
print("\n  D1 bu sayiyi DOGRUDAN olcer:  |c| = 1.95 * rho_1 / 0.1302")

with open(os.path.join(M29, "n19_tek_bilinmeyen.json"), "w", encoding="utf-8") as fh:
    json.dump(
        {
            "rho_s_bilesik": RHO_S_BILESIK,
            "taban_mse": TABAN_MSE,
            "yedek": YEDEK,
            "saf_span": SAF_SPAN,
            "satirlar": SATIR,
        },
        fh,
        indent=1,
    )
print("\n-> n19_tek_bilinmeyen.json")
