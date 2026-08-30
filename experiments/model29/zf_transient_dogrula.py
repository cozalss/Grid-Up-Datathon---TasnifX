"""KRITIK TEST -- ajanin iddiasini KENDIM olc.

IDDIA: yeni bir trafonun ILK gunleri, kendi SONRAKI seviyesinin yalnizca
       +0.20..+0.33 altinda. m6 ise +0.85 varsayiyor.

TEST: egitim verisinde, bitisten en az 130 gun once panele giren trafolari bul.
      x = ilk k gununun ortalama log1p'i   (k = 1,2,4,8)
      y = sonraki 120 gununun ortalama log1p'i
      mean(y - x) = gercek baslangic transienti.

PLASEBO: yerlesik trafolarda RASTGELE bir 2 gunluk pencere al, sonraki 120 gune
         bak. Transient yoksa fark ~0 cikmali. Trafo duzeyinde 200 tekrar.

Ayrica: m6'nin +0.85'i savunulabilir mi? Trafonun kendi ilk gunleri
        ARTEFAKT ise (kismi okuma), transient olcumu bunu ZATEN icerir --
        cunku ayni ilk gunleri capa olarak kullaniyoruz.
"""

import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
tr = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"))
tr["ly"] = np.log1p(tr.tuketim.values.astype(np.float64))
tr["t"] = pd.to_datetime(tr.tarih)
son = tr.t.max()
print(f"egitim {tr.t.min().date()} .. {son.date()}  {tr.tanim.nunique():,} trafo")

ilk = tr.groupby("tanim").t.min()
uyg = ilk[ilk <= son - pd.Timedelta(days=130)].index
print(f"bitisten >=130 gun once giren trafo: {len(uyg):,}")

g = tr[tr.tanim.isin(uyg)].sort_values(["tanim", "t"])
print("\nBASLANGIC TRANSIENTI:  mean( sonraki120gun - ilk k gun )")
print(f"{'k':>4s} {'n trafo':>9s} {'ort fark':>10s} {'SE':>8s} {'medyan':>9s} {'egim':>7s}")
sonuc = {}
for k in [1, 2, 4, 8]:
    xs, ys = [], []
    for t, gg in g.groupby("tanim"):
        v = gg.ly.values
        d0 = gg.t.values
        if len(v) < k + 30:
            continue
        bas = d0[0]
        ilk_k = v[:k]
        sonraki = v[
            (d0 > bas + np.timedelta64(k, "D")) & (d0 <= bas + np.timedelta64(k + 120, "D"))
        ]
        if len(sonraki) < 30:
            continue
        xs.append(ilk_k.mean())
        ys.append(sonraki.mean())
    xs, ys = np.array(xs), np.array(ys)
    fark = ys - xs
    egim = np.polyfit(xs, ys, 1)[0]
    sonuc[k] = fark
    print(
        f"{k:4d} {len(fark):9,} {fark.mean():+10.4f} "
        f"{fark.std() / np.sqrt(len(fark)):8.4f} {np.median(fark):+9.4f} {egim:7.3f}"
    )

# --- PLASEBO: yerlesik trafolarda rastgele pencere ---
print("\nPLASEBO: yerlesik trafolarda RASTGELE 2-gunluk pencere -> sonraki 120 gun")
rng = np.random.default_rng(7)
uzun = [t for t, gg in g.groupby("tanim") if len(gg) >= 200]
print(f"  >=200 satirlik {len(uzun):,} trafo uzerinde 200 tekrar")
bos = []
gr = {t: gg for t, gg in g.groupby("tanim") if t in set(uzun)}
for _ in range(200):
    f = []
    for t in rng.choice(uzun, size=min(300, len(uzun)), replace=False):
        gg = gr[t]
        v = gg.ly.values
        d0 = gg.t.values
        i = rng.integers(20, len(v) - 130)
        bas = d0[i]
        w = v[i : i + 2]
        nx = v[(d0 > bas + np.timedelta64(2, "D")) & (d0 <= bas + np.timedelta64(122, "D"))]
        if len(nx) < 30:
            continue
        f.append(nx.mean() - w.mean())
    bos.append(np.mean(f))
bos = np.array(bos)
g2 = sonuc[2].mean()
z = (g2 - bos.mean()) / bos.std()
print(f"  bos dagilim: ort {bos.mean():+.4f}  sd {bos.std():.4f}")
print(f"  GOZLENEN (k=2): {g2:+.4f}   z = {z:+.2f}")
print(f"\nm6'NIN VARSAYIMI: +0.85   |   OLCULEN: {g2:+.4f}")
print(f"FARK (yanlilik): {0.8518 - g2:+.4f}")
