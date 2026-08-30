"""IKI IS: kappa'yi yeni rho_pred ile yeniden degerlendir + esik kesitlerini denetle.

1) KAPPA. rho_pred 0.2081 -> 0.2685 cikti. kappa=0.070 eski degere gore
   secilmisti. Her hedef icin gereken gerceklesme oranini yeniden hesapla.

2) SEKIZINCI HATA ARAYISI -- esik kesitleri.
   m122'nin kur() fonksiyonu 'ust25'/'ust10'/'alt10' kesitlerinde esigi
   TEST dagilimindan aliyor ve ayni sayiyi BLOK verisine de uyguluyor:
       fv = xt[isfinite]; v_ = quantile(fv, q); (xt > v_), (xb > v_)
   Iki dagilim farkliysa blok gostergesi cok dengesiz olur (ornegin blokta
   satirlarin %2'si ya da %60'i esigi gecer) ve o eksenin CV korelasyonu --
   yani ISARETI -- guvenilmez hale gelir. Isaret CV'den geldigi icin bu
   dogrudan bilesigin yonunu bozar.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
M29 = os.path.join(KOK, "experiments/model29")
sys.path.insert(0, M29)

HEDEF = {"1. sira": 0.99009, "2. sira": 0.99790, "3. sira": 0.99940, "4. sira": 1.00118}
with open(os.path.join(M29, "m122_nihai.json")) as fh:
    NIH = json.load(fh)
RHO = NIH["rho"]
KAP_ETKIN = NIH["kappa_etkin"]
SABIT = NIH["sabit"]
# MSE_OPT'u sabitten geri cikar: sabit = MSE_OPT + kappa_etkin^2
MSE_OPT = SABIT - KAP_ETKIN**2
print(f"rho_pred = {RHO:.4f}   kappa_etkin = {KAP_ETKIN:.6f}   MSE_OPT = {MSE_OPT:.9f}")
print(f"saf optimum = {np.sqrt(MSE_OPT):.6f}")

print("\n" + "=" * 70)
print("1) KAPPA SECIMI -- her hedef icin gereken GERCEKLESME ORANI f")
print("=" * 70)
print(f"{'kappa':>8s} " + " ".join(f"{k:>10s}" for k in HEDEF))
for kap in [0.050, 0.060, 0.070, 0.0792, 0.090, 0.100, 0.120, 0.1476]:
    hs = []
    for h in HEDEF.values():
        f = (MSE_OPT + kap * kap - h * h) / (2 * kap * RHO)
        hs.append(f"{f:10.3f}" if f > 0 else f"{'ZATEN':>10s}")
    print(f"{kap:8.4f} " + " ".join(hs))
print(
    f"\n  kappa* (2. sira icin f'yi en aza indiren) = {np.sqrt(max(MSE_OPT - 0.99790**2, 0)):.4f}"
)

print("\n  SKOR TABLOSU (satir=kappa, sutun=gerceklesme orani f)")
print(f"{'kappa':>8s} " + " ".join(f"f={f:<5.2f}" for f in [0.0, 0.2, 0.3, 0.5, 0.8, 1.0]))
for kap in [0.050, 0.060, 0.070, 0.0792, 0.100, 0.1476]:
    hs = []
    for f in [0.0, 0.2, 0.3, 0.5, 0.8, 1.0]:
        s = np.sqrt(max(MSE_OPT + kap * kap - 2 * kap * f * RHO, 1e-12))
        hs.append(f"{s:7.5f}")
    print(f"{kap:8.4f} " + " ".join(hs))

print("\n" + "=" * 70)
print("2) ESIK KESITLERI -- test esigi blokta ne kadar dengeli?")
print("=" * 70)
tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
e = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
bf = e[e._blok == "yaz25"]
ESIK = {"ust10": (0.9, True), "ust25": (0.75, True), "alt10": (0.1, False)}
eksenler = NIH["eksenler"]
kesitli = [a for a in eksenler if ":" in a and a.split(":", 1)[1] in ESIK]
print(f"bilesikteki {len(eksenler)} eksenin {len(kesitli)} tanesi esik kesiti\n")
print(f"{'eksen':>34s} {'hedef pay':>10s} {'test payi':>10s} {'blok payi':>10s} {'durum':>10s}")
supheli = []
for ad in kesitli:
    kol, kip = ad.split(":", 1)
    q, ust = ESIK[kip]
    if kol not in tp.columns or kol not in bf.columns:
        continue
    xt = tp[kol].to_numpy()
    xb = bf[kol].to_numpy()
    fv = xt[np.isfinite(xt)]
    if fv.size == 0:
        continue
    v_ = np.quantile(fv, q)
    pt = float(np.mean(xt > v_)) if ust else float(np.mean(xt < v_))
    fb = xb[np.isfinite(xb)]
    pb_ = float(np.mean(fb > v_)) if ust else float(np.mean(fb < v_))
    hedef_pay = 1 - q if ust else q
    # blok payi hedeften cok saparsa gosterge dengesizdir
    kotu = (
        pb_ < 0.02
        or pb_ > 0.60
        or (pb_ / max(hedef_pay, 1e-9) > 3)
        or (pb_ / max(hedef_pay, 1e-9) < 0.33)
    )
    if kotu:
        supheli.append(ad)
    print(
        f"{ad[:34]:>34s} {hedef_pay:10.3f} {pt:10.3f} {pb_:10.3f} "
        f"{'SUPHELI' if kotu else 'tamam':>10s}"
    )
print(f"\n  supheli kesit sayisi: {len(supheli)}")
if supheli:
    print("  ", supheli)
else:
    print("   (esik kesitlerinin hepsi blokta makul dengeli -- sekizinci hata BURADA DEGIL)")

print("\n" + "=" * 70)
print("3) EK DENETIM -- NaN doldurmanin blok/test tutarliligi")
print("=" * 70)
print(f"{'kolon':>24s} {'test NaN%':>10s} {'blok NaN%':>10s} {'not':>28s}")
kols = sorted(
    {a.split(":")[0].split("*")[0] for a in eksenler}
    | {a.split("*")[-1].split(":")[0] for a in eksenler if "*" in a}
)
for kol in kols:
    if kol not in tp.columns or kol not in bf.columns:
        continue
    nt = float(pd.isna(tp[kol]).mean())
    nb = float(pd.isna(bf[kol]).mean())
    if nt > 0.01 or nb > 0.01:
        not_ = "test'te cok NaN" if nt > 0.5 else ("blok'ta cok NaN" if nb > 0.5 else "")
        print(f"{kol:>24s} {nt * 100:10.1f} {nb * 100:10.1f} {not_:>28s}")
print("   (yalniz %1'in uzerinde NaN tasiyan kolonlar listelendi)")
