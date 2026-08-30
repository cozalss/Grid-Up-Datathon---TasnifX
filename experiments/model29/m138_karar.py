"""KARAR: sigma'yi degistirelim mi? Olcut = 2. SIRAYI TUTTURMA KESINLIGI.

m134 L'nin gurultusunun yuvarlama tabaninda oldugunu buldu (10.7:1).
m136/m137 iki kurulusu capraz sinadi. Bu betik ikisini ASIL HEDEFE gore
karsilastirir: 2. sira (0.99614) icin 1.95 carpaninin ne kadarinin
gerceklesmesi gerekiyor (gereken f). Kucuk f = daha kesin.

  skor^2 = taban_MSE + kappa^2 - 2*rho_ger*kappa
  kappa  = sqrt(tahmin_edilen_taban - hedef^2)      <- kurulusun KENDI tahmini
  gereken rho_ger: skor^2 = hedef^2 kosulundan, GERCEK tabanla
"""

import numpy as np

HEDEF2, HEDEF3 = 0.99614, 0.99927
# m137 capraz sinav: (gercek dunya, kurulus) -> GERCEK taban MSE
TABAN = {
    ("A", "A"): 1.002456,
    ("A", "B"): 1.003981,
    ("B", "A"): 1.002027,
    ("B", "B"): 1.001057,
}
# kurulusun KENDI inandigi taban (kappa'yi bununla secer) ve kendi rho tahmini
INANC = {"A": 1.002112, "B": 1.001057}  # = (saf optimum)^2
RHO = {"A": 0.2522, "B": 0.2494}  # m136
OLASI = {"A": 0.084, "B": 0.900}  # m134 sonsal

print("gereken f = 2. sirayi tutturmak icin 1.95 carpaninin gereken orani\n")
print(
    f"{'kurulus':>8s} {'kappa':>8s} "
    + " ".join(f"{'gercek ' + g:>12s}" for g in "AB")
    + f" {'EN KOTU':>9s} {'beklenen':>9s}"
)
SON = {}
for k in "AB":
    kap = np.sqrt(max(INANC[k] - HEDEF2**2, 1e-12))
    fs = {}
    for g in "AB":
        # 2*rho*kappa - kappa^2 >= taban - hedef^2
        gerek = (TABAN[(g, k)] - HEDEF2**2 + kap**2) / (2 * kap)
        # gercek dunya A ise B'nin r_hat'i sisik -> rho tahmini de sisik
        sisme = np.sqrt(TABAN[(g, k)] / INANC[k]) if TABAN[(g, k)] > INANC[k] else 1.0
        fs[g] = gerek / (RHO[k] / sisme)
    bek = sum(OLASI[g] * fs[g] for g in "AB") / sum(OLASI.values())
    SON[k] = (max(fs.values()), bek)
    print(
        f"{k:>8s} {kap:8.5f} "
        + " ".join(f"{fs[g]:12.3f}" for g in "AB")
        + f" {max(fs.values()):9.3f} {bek:9.3f}"
    )

ek = min(SON, key=lambda k: SON[k][0])
eb = min(SON, key=lambda k: SON[k][1])
print(f"\n  EN KOTU DURUMDA en iyi: {ek}   (gereken f {SON[ek][0]:.3f})")
print(f"  BEKLENENDE en iyi     : {eb}   (gereken f {SON[eb][1]:.3f})")
print("\nHEDEF 'garanti 2. sira' oldugu icin EN KOTU DURUM olcuttur.")
print("Ayrica: mevcut hazir dosya A kurulusudur ve m126 ile ucdan uca")
print("dogrulanmistir. B'ye gecmek yeniden kurulum + yeniden dogrulama")
print("demektir ve dayanagi n=3'luk bir kanittir.")
