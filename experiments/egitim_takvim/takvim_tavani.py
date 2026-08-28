"""TAKVIM TAVANI -- herhangi bir takvim kolonunun kazandirabilecegi
MSE'nin UST SINIRI, dogrudan ETIKETLERDEN (model kullanmaz, dolayisiyla
dongusel degil).

Yontem (docs/41 sec.5 ile ayni ruh, ama MSE cinsinden):
  r = log1p(tuketim) - trafo ortalamasi - 15 gunluk merkezli hareketli ort
  d(t) = r'nin gun bazinda ortalamasi   (panel gun etkisi)
  takvim modeli: d(t) ~ hafta gunu + tatil gostergeleri
  TAVAN = agirlikli var( takvimin acikladigi kisim )

Bu bir UST SINIR: uretim modelinin bu etkiden HIC yakalamadigini varsayar.
Gercekte ulusal_gunluk / gun_uzunlugu_saat / t_hg_sapma bir kismini tasiyor,
ve v83'un gun-genligi son islemi ayrica gun eksenini duzeltiyor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "src"))

ONB = KOK / "data" / "interim" / "deney"
df = pd.read_parquet(ONB / "egitim.parquet", columns=["tarih", "tanim_num", "tuketim"])
df["tarih"] = pd.to_datetime(df["tarih"])
df["y"] = np.log1p(df["tuketim"].clip(lower=0.0))

# trafo seviyesini cikar
df["r"] = df["y"] - df.groupby("tanim_num")["y"].transform("mean")

# gun bazinda panel etkisi
gunluk = df.groupby("tarih").agg(d_ham=("r", "mean"), n=("r", "size")).sort_index()
# 15 gunluk merkezli hareketli ortalama ile mevsimden arindir
gunluk["taban"] = gunluk["d_ham"].rolling(15, center=True, min_periods=5).mean()
gunluk["d"] = gunluk["d_ham"] - gunluk["taban"]
gunluk = gunluk.dropna(subset=["d"])

import holidays  # noqa: E402

tr = holidays.country_holidays("TR", years=[2025, 2026])


def takvim_tasarim(idx: pd.DatetimeIndex) -> pd.DataFrame:
    X = pd.get_dummies(idx.dayofweek, prefix="hg").astype(float)
    X.index = idx
    tatil = np.array([g.date() in tr for g in idx], dtype=float)
    X["tatil"] = tatil
    # kurban/ramazan bayramlari ayri
    adlar = np.array([tr.get(g.date(), "") or "" for g in idx])
    X["kurban"] = np.char.find(np.char.lower(adlar.astype(str)), "kurban") >= 0
    X["ramazan_b"] = np.char.find(np.char.lower(adlar.astype(str)), "ramazan") >= 0
    X["kurban"] = X["kurban"].astype(float)
    X["ramazan_b"] = X["ramazan_b"].astype(float)
    return X


def tavan(alt: pd.DataFrame, etiket: str) -> None:
    idx = pd.DatetimeIndex(alt.index)
    X = takvim_tasarim(idx).to_numpy()
    w = alt["n"].to_numpy(dtype="float64")
    d = alt["d"].to_numpy(dtype="float64")
    W = np.sqrt(w)[:, None]
    # agirlikli en kucuk kareler
    beta, *_ = np.linalg.lstsq(X * W, d * np.sqrt(w), rcond=None)
    fit = X @ beta
    # gun etkisi zaten ortalamasi ~0; aciklanan agirlikli kare ortalama
    ac = float(np.average(fit**2, weights=w))
    top = float(np.average(d**2, weights=w))
    print(
        f"{etiket:28} gun={len(alt):3}  gun etkisi std={np.sqrt(top):.4f}   "
        f"takvimin acikladigi std={np.sqrt(ac):.4f}   TAVAN dMSE={-ac:.5f}   R2={ac / top:.3f}"
    )


print("Herhangi bir takvim kolonu icin UST SINIR (satir-agirlikli, log^2 = MSE birimi)")
print("Gate esigi karsilastirmasi: kabul icin gereken toplam test dMSE <= -0,00200\n")
tavan(gunluk, "TUM EGITIM (2025-01..2026-03)")
mask = (gunluk.index >= "2025-04-01") & (gunluk.index <= "2025-07-31")
tavan(gunluk[mask], "yaz25 (TESTIN MEVSIMSEL IKIZI)")
mask2 = (gunluk.index >= "2025-12-01") & (gunluk.index <= "2026-03-31")
tavan(gunluk[mask2], "kis26")

print("\n--- en buyuk gun sapmalari (tum egitim) ---")
en = gunluk.reindex(gunluk["d"].abs().sort_values(ascending=False).index).head(12)
for t, sat in en.iterrows():
    ad = tr.get(t.date(), "")
    print(f"  {t.date()}  {t.day_name()[:3]}  d={sat['d']:+.4f}  {ad or ''}")
