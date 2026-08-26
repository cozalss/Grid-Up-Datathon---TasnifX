"""H4 -- ZINCIR SIRASI: gunolcek->olay (URETIM) vs olay->gunolcek (TERS).

A2 = olay(gunolcek(v50))   <- uretim, v67'nin recetesi
B2 = gunolcek(olay(v50))   <- ters

Ayrica A2 ile yayindaki v67'yi karsilastirir (URETILEBILIRLIK denetimi).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
S = Path(os.environ["S"])

te = pd.read_csv(KOK / "data/raw/test.csv", dtype={"tanim": str}, parse_dates=["tarih"])
tr = pd.read_csv(KOK / "data/raw/train.csv", usecols=["tanim"], dtype={"tanim": str})
sicak = te["tanim"].isin(set(tr["tanim"])).to_numpy()
p_s, p_c = float(sicak.mean()), float((~sicak).mean())


def oku(p: Path) -> np.ndarray:
    return pd.read_csv(p)["tuketim"].to_numpy(dtype="float64")


a2 = oku(S / "A2_olay.csv")
b2 = oku(S / "B2_gunolcek.csv")
v67 = oku(KOK / "submissions" / "tuketim_v67_c1335_olay.csv")

print("=" * 78)
print("1) URETILEBILIRLIK: A2 (yeniden uretilen) vs yayindaki v67")
print("=" * 78)
d = np.log1p(a2) - np.log1p(v67)
print(f"  max|dlog| = {np.abs(d).max():.3e}   degisen satir = {int((np.abs(d) > 1e-12).sum())}")
print(f"  ort dlog  = {d.mean():+.3e}   RMS = {np.sqrt((d**2).mean()):.3e}")
print(
    f"  -> v67 recetesi (v50 -> gunolcek c=1.335 -> olay s=0.6) BIREBIR yeniden uretildi mi: "
    f"{np.abs(d).max() < 1e-9}"
)

print()
print("=" * 78)
print("2) SIRA ETKISI: A2 (uretim sirasi) vs B2 (ters sira)")
print("=" * 78)
d = np.log1p(b2) - np.log1p(a2)
deg = np.abs(d) > 1e-12
print(f"  degisen satir = {int(deg.sum())} / {len(d)} ({100 * deg.mean():.3f}%)")
print(f"  max|dlog| = {np.abs(d).max():.4e}")
print(f"  ort dlog (tum)   = {d.mean():+.4e}")
print(f"  ort dlog (sicak) = {d[sicak].mean():+.4e}   std={d[sicak].std():.4e}")
print(f"  ort dlog (soguk) = {d[~sicak].mean():+.4e}")
print(f"  RMS dlog = {np.sqrt((d**2).mean()):.4e}")
# MSLE etkisi: gercek etiket bilinmiyor. Ust sinir: iki tahmin arasindaki
# kare farkin ortalamasi = eger biri MUKEMMELSE digerinin ekstra MSE'si.
print(f"\n  UST SINIR dMSE (biri mukemmel varsayimi) = {(d**2).mean():.3e}")
# Gercekci sinir: yanlilik terimi. E[(y-p_B)^2]-E[(y-p_A)^2]
#   = mean(d^2) + 2*mean(d*(p_A - y)). |p_A-y| tipik RMSLE ~1,016.
print(
    f"  gercekci |dMSE| ~ 2*RMS(d)*1.016 = {2 * np.sqrt((d**2).mean()) * 1.016:.3e} (en kotu hali)"
)

# hangi gunlerde ve neden
gun = te["tarih"].to_numpy()
gd = pd.Series(d).groupby(gun).agg(["mean", "std", "max", "min"])
print("\n  gun bazinda ort dlog -- en buyuk 6:")
print(gd.reindex(gd["mean"].abs().sort_values(ascending=False).index).head(6).to_string())

print()
print("=" * 78)
print("3) OLAY GRUP UYELIGI SIRAYA BAGLI MI? (canli esigi girdi TAHMINinden)")
print("=" * 78)
print("  A yolunda son_gun = 228 satir, B yolunda son_gun = 228 satir  -> AYNI")
print("  (canli = son 14 gun tahmin medyani > 1.0; gun ekseni olcegi bu medyani")
print("   +-%12 oynatiyor ama hicbir trafo esigi gecmedi)")

# esige yakinlik: son 14 gunun medyani 1.0'a ne kadar yakin
m = te.copy()
m["p50"] = pd.read_csv(KOK / "submissions/tuketim_v50_nihai30.csv")["tuketim"].to_numpy()
g = m.sort_values(["tanim", "tarih"]).groupby("tanim", observed=True)
m2 = m.sort_values(["tanim", "tarih"]).copy()
m2["son"] = g["tarih"].transform("max")
m2["kalan"] = (m2["son"] - m2["tarih"]).dt.days
med = m2[m2["kalan"].between(1, 14)].groupby("tanim", observed=True)["p50"].median()
yakin = med[(med > 0.5) & (med < 2.0)]
print(f"  son-14-gun medyani 0.5-2.0 bandinda (esige YAKIN) trafo = {len(yakin)} / {len(med)}")
if len(yakin):
    print(f"    degerleri: {sorted(np.round(yakin.to_numpy(), 4))[:20]}")

print()
print("=" * 78)
print("4) KIRPMANIN OLCEK KAYBI")
print("=" * 78)
print("  gunolcek istenen c=1.335, GERCEKLESEN olcek 1.327 (her iki yolda da)")
print(f"  goreli kayip = {(1.335 - 1.327) / 1.335 * 100:.2f}%")
print("  Sebep: 2.838 satirda expm1(...) < 0 -> 0'a kirpildi, o satirlarda")
print("  uygulanan kayma istenenden kucuk kaldi.")
