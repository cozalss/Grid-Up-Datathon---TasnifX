"""1) Test penceresinde bayram var mi?  2) Sicak uzman hafta gununu
gordugu kolonlardan geri kurabiliyor mu?  Salt okuma, model egitmez."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

ONB = KOK / "data" / "interim" / "deney"
test = pd.read_parquet(ONB / "test.parquet")
egitim = pd.read_parquet(ONB / "egitim.parquet")

print("=== 1. TEST PENCERESI ve TATILLER ===")
tarih_kol = [k for k in test.columns if "tarih" in k.lower() or k == "gun"]
print("tarih kolonlari:", tarih_kol)
td = pd.to_datetime(test["tarih"])
print(
    f"test penceresi : {td.min().date()} .. {td.max().date()}  ({td.dt.normalize().nunique()} gun)"
)
ed = pd.to_datetime(egitim["tarih"])
print(f"egitim penceresi: {ed.min().date()} .. {ed.max().date()}")

import holidays  # noqa: E402

tr = holidays.country_holidays("TR", years=[2025, 2026])
print("\n2026 TR resmi tatilleri:")
for g, ad in sorted(tr.items()):
    if g.year == 2026:
        icinde = "  <== TEST ICINDE" if td.min().date() <= g <= td.max().date() else ""
        print(f"  {g}  {ad}{icinde}")

# tatil kolonu testte dolu mu
for k in ("tatil_mi", "tatil_kod", "ramazan_ayi", "tk_haftanin_gunu"):
    if k in test.columns:
        gunluk = test.groupby(test["tarih"])[k].first()
        print(f"\n{k}: testte sifir-disi gun sayisi = {int((gunluk != 0).sum())} / {len(gunluk)}")
        if k == "tatil_mi":
            print("  tatil gunleri:", [str(x.date()) for x in gunluk[gunluk != 0].index])

print("\n=== 2. HAFTA GUNU, GORULEN KOLONLARDAN GERI KURULABILIYOR MU? ===")
# gun bazinda tekillestir
gun = egitim.drop_duplicates("tarih").set_index("tarih").sort_index()
hg = pd.to_datetime(gun.index).dayofweek
adaylar = [
    "ulusal_gunluk",
    "ulusal_tepe",
    "ulusal_tepe_orani",
    "ulusal_yil_once",
    "gun_uzunlugu_saat",
    "gunes_radyasyon",
]
mevcut = [k for k in adaylar if k in gun.columns]
print(f"gun sayisi: {len(gun)}   aday kolon: {mevcut}")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

X = gun[mevcut].to_numpy(dtype="float64")
ok = np.isfinite(X).all(axis=1)
X, y = X[ok], hg[ok]
print(f"finite gun: {len(X)}")
skor = cross_val_score(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
    X,
    y,
    cv=5,
    scoring="accuracy",
)
print(f"7-sinif hafta gunu tahmini (5-kat CV dogruluk): {skor.mean():.3f}  (sans = 0,143)")

# hafta sonu ikili
ws = (y >= 5).astype(int)
skor2 = cross_val_score(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    X,
    ws,
    cv=5,
    scoring="accuracy",
)
print(
    f"hafta sonu ikili (5-kat CV dogruluk)          : {skor2.mean():.3f}  (taban = {max(ws.mean(), 1 - ws.mean()):.3f})"
)

# yalnizca ulusal_gunluk ile
if "ulusal_gunluk" in gun.columns:
    Xu = gun[["ulusal_gunluk"]].to_numpy(dtype="float64")[ok]
    s3 = cross_val_score(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        Xu,
        ws,
        cv=5,
        scoring="accuracy",
    )
    print(f"  yalniz ulusal_gunluk -> hafta sonu           : {s3.mean():.3f}")
