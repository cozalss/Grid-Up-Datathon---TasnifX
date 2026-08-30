"""ESIK TAHMINI GUNCELLEMESI -- 31 Agustos 04:15 gozlemiyle.

n02 (30 Agustos 23:52) 2. sira esigini bitiste 0.9897 [0.9870, 0.9908]
tahmin etmisti. O tahmin, gozlenen kaymanin SURECEGI varsayimina dayali
dogrusal modelle ussel-sonumlu modelin ortalamasiydi.

YENI BILGI: 2. sira esigi 30 Agustos 20:09'dan 31 Agustos 04:15'e kadar
-- SEKIZ SAAT -- HIC DEGISMEDI. 1. sira ise 30 Agustos 05:03'ten beri,
YIRMI DORT SAATTIR sabit. Kayma pratikte DURMUS.

2. sira esigi zaman serisi:
    29 Agu ~00:00   0.99940
    30 Agu 07:51    0.99790
    30 Agu 17:26    0.99614
    30 Agu 22:06    0.99556
    31 Agu 04:15    0.99556   <- YENI, 8 saat DEGISMEDI

Ardisik farklar: -0.00150, -0.00176, -0.00058, 0.00000
Son iki aralikta kayma hizi 10 kata yakin dustu.

Bu betik ussel-sonumlu modeli YENI noktayla yeniden uydurur ve iki uc
senaryoyu da gosterir.

NOT: bu bir TAHMINDIR, olcum degil. Rakipler son gun sicrayabilir --
nitekim bu gece Ahmet Bugrahan 1.00118'den 0.99975'e sicradi (bizi 9.
siraya dusurdu). Tepedeki sabitlik, ALTTAKI hareketliligin yoklugu
anlamina GELMIYOR.
"""

import json
import os

import numpy as np
from scipy.optimize import curve_fit

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")

# t = 2026-08-29 00:00 UTC'den itibaren GUN
GOZ2 = [
    (0.00, 0.99940),
    (1.3271, 0.99790),  # 30 Agu 07:51
    (1.7264, 0.99614),  # 30 Agu 17:26
    (1.9208, 0.99556),  # 30 Agu 22:06
    (2.1771, 0.99556),  # 31 Agu 04:15  YENI
]
BITIS = 3.9993  # 1 Eylul 23:59 UTC

t = np.array([a for a, _ in GOZ2])
y = np.array([b for _, b in GOZ2])


def sonumlu(x, y_sonsuz, A, tau):
    return y_sonsuz + A * np.exp(-x / tau)


print("2. SIRA ESIGI -- gozlemler")
for a, b in GOZ2:
    print(f"  t={a:6.4f} gun   {b:.5f}")
print(f"\nardisik farklar: {np.round(np.diff(y), 5).tolist()}")
print("son aralikta 8 saat boyunca DEGISIM YOK\n")

# dogrusal (kayma SURUYOR varsayimi)
b_, a_ = np.polyfit(t, y, 1)
dog = a_ + b_ * BITIS
# ussel sonumlu (kayma DURUYOR varsayimi)
try:
    pop, pcov = curve_fit(
        sonumlu,
        t,
        y,
        p0=[0.995, 0.005, 1.0],
        bounds=([0.95, 0.0, 0.05], [1.01, 0.10, 10.0]),
        maxfev=20000,
    )
    us = float(sonumlu(BITIS, *pop))
    print(f"ussel sonumlu: y_sonsuz={pop[0]:.5f} A={pop[1]:.5f} tau={pop[2]:.3f} gun")
except Exception as e:  # noqa: BLE001
    us, pop = float(y[-1]), None
    print(f"ussel uydurma basarisiz ({e}); son deger kullaniliyor")

# TAMAMEN DURDU senaryosu
durdu = float(y[-1])

print(f"\n{'senaryo':>28s} {'bitisteki 2. sira esigi':>24s}")
print(f"{'dogrusal (kayma suruyor)':>28s} {dog:24.5f}")
print(f"{'ussel sonumlu':>28s} {us:24.5f}")
print(f"{'tamamen durdu':>28s} {durdu:24.5f}")

# Agirlikli merkez: 8 saatlik durgunluk ussel/durdu senaryolarini destekliyor
# ama son gun sicramasi da gercek bir risk (bu gece bir rakip 0.0014
# sicradi). Agirliklar: dogrusal %25, ussel %45, durdu %30.
merkez = 0.25 * dog + 0.45 * us + 0.30 * durdu
alt, ust = min(dog, us, durdu), max(dog, us, durdu)
print(f"\n{'AGIRLIKLI MERKEZ':>28s} {merkez:24.5f}")
print(f"{'zarf':>28s} {f'[{alt:.5f}, {ust:.5f}]':>24s}")

print("\nn02'nin (dun gece) tahmini 0.9897 [0.9870, 0.9908] idi.")
print(f"Sekiz saatlik durgunluk sonrasi merkez {merkez:.5f}'e KAYDI --")
print("yani hedef DUSUNDUGUMUZDEN YAKIN.")

TABAN_MSE = 1.00202690323433
print(f"\n{'esik':>10s} {'gereken toplam rho^2':>22s} {'gereken rho':>12s}")
for e in [durdu, merkez, us, dog]:
    g = TABAN_MSE - e * e
    print(f"{e:10.5f} {g:22.5f} {np.sqrt(max(g, 0)):12.4f}")

with open(os.path.join(M29, "n17_esik_guncel.json"), "w", encoding="utf-8") as fh:
    json.dump(
        {
            "gozlemler": GOZ2,
            "bitis_t": BITIS,
            "dogrusal": float(dog),
            "ussel": float(us),
            "durdu": durdu,
            "merkez": float(merkez),
            "zarf": [float(alt), float(ust)],
            "not": "8 saat degisim yok; 1. sira 24 saattir sabit",
        },
        fh,
        indent=1,
    )
print("\n-> n17_esik_guncel.json")
