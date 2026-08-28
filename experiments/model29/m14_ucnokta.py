"""Uc noktalar: yuksek seviyeli tahminler ve sifir-gecisleri."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *

pd.set_option("display.width", 200)
tr = yukle()
kesim = "2025-10-31"
k = pd.Timestamp(kesim)
gec, hed = hazirla(tr, kesim)
tam = pencere_seviye(gec, kesim, None, "mean")
kok = float(gec.ly.mean())
s7 = pencere_seviye(gec, kesim, 7, "mean")
p = geri_dolgu(hed, s7, tam, kok=kok)
e2 = (p - hed.ly.values) ** 2
hed = hed.assign(pred=p, e2=e2)
tot = e2.sum()
# en kotu trafolar
tf = hed.groupby("tanim").e2.sum().sort_values(ascending=False)
print("EN KOTU 15 TRAFO (hata kutlesi payi):")
for t in tf.index[:15]:
    h = hed[hed.tanim == t]
    g = gec[gec.tanim == t]
    print(
        f" {t} guc={h.guc.iloc[0]:6.0f} pay=%{100 * tf[t] / tot:4.1f} n_hed={len(h)} pred_ly={h.pred.iloc[0]:.2f} "
        f"hed_ly ort={h.ly.mean():.2f} min={h.ly.min():.2f} max={h.ly.max():.2f} | "
        f"gec n={len(g)} son7ly={g[g.tarih > k - pd.Timedelta(days=7)].ly.mean():.2f} tumly={g.ly.mean():.2f} "
        f"gec_ilk={g.tarih.min().date()} gec_son={g.tarih.max().date()}"
    )
print(
    f"\nilk 15 trafo toplam pay: %{100 * tf[:15].sum() / tot:.1f}; ilk 50: %{100 * tf[:50].sum() / tot:.1f}; ilk 200: %{100 * tf[:200].sum() / tot:.1f}"
)
print(f"toplam sicak trafo: {hed.tanim.nunique()}")
