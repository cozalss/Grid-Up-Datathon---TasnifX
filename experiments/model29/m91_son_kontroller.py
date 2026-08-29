"""Gonderim oncesi kalan akil saglamalari."""

import os

import numpy as np
import pandas as pd
from m30_ozellik import KOK

S = lambda f: pd.read_csv(os.path.join(KOK, "submissions", f))
te = pd.read_csv(
    os.path.join(KOK, "data/raw/test.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
tr = pd.read_csv(
    os.path.join(KOK, "data/raw/train.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
m4 = S("tuketim_m4_hava_capali.csv")
v102 = S("tuketim_v102_kappa_optimum.csv")

print("1) DOSYA BUTUNLUGU")
print(f"   satir {len(m4):,} | essiz id {m4.id.nunique():,} | tekrar {len(m4) - m4.id.nunique()}")
print(f"   id sirasi test.csv ile birebir: {bool((m4.id.values == te.id.values).all())}")
print(
    f"   NaN {m4.tuketim.isna().sum()} | negatif {(m4.tuketim < 0).sum()} | sonsuz {np.isinf(m4.tuketim).sum()}"
)
print(
    f"   min {m4.tuketim.min():.4f} | maks {m4.tuketim.max():,.0f} | tam sifir {(m4.tuketim == 0).sum()}"
)

print("\n2) 2026-05-11 TOPLU DALGA (testin %25'i, en buyuk tek kohort)")
sog = (~te.tanim.isin(set(tr.tanim))).values
ilk = te.groupby("tanim").tarih.min()
dalga = te.tanim.map(ilk).eq(pd.Timestamp("2026-05-11")).values
print(
    f"   {dalga.sum():,} satir (%{100 * dalga.mean():.1f}), {te[dalga].tanim.nunique():,} trafo, soguk orani %{100 * sog[dalga].mean():.1f}"
)
for ad, p in [("v102", v102), ("m4", m4)]:
    lp = np.log1p(p.tuketim.values)
    print(
        f"   {ad:5s} dalga log-ort {lp[dalga].mean():.4f} std {lp[dalga].std():.4f} | dalga disi {lp[~dalga].mean():.4f}"
    )
# 2025'te ayni desen: toplu giren trafolarin gercek seviyesi
tr2 = tr.copy()
tr2["ly"] = np.log1p(tr2.tuketim)
i2 = tr2.groupby("tanim").tarih.min()
buyuk = i2.value_counts()
g = buyuk[buyuk >= 50].index
print(f"   2025 KARSILASTIRMA: >=50 trafonun ayni gun girdigi {len(g)} olay")
m = tr2.tanim.map(i2).isin(g)
print(
    f"   toplu giren trafolarin ilk 4 aydaki gercek log-ort {tr2[m].ly.mean():.4f} vs digerleri {tr2[~m].ly.mean():.4f}"
)

print("\n3) TOHUM KARARLILIGI (m1 vs m3 vs m4 -- ayni aile, farkli kurulum)")
m1 = S("tuketim_m1_ileri_huber.csv")
m3 = S("tuketim_m3_hl1_capali.csv")
for a, b, ad in [(m3, m4, "m3 vs m4"), (m1, m3, "m1 vs m3")]:
    d = np.log1p(a.tuketim.values) - np.log1p(b.tuketim.values)
    print(
        f"   {ad}: Q={float((d**2).mean()):.6f}  korelasyon {np.corrcoef(np.log1p(a.tuketim), np.log1p(b.tuketim))[0, 1]:.5f}"
    )

print("\n4) UC DEGER GUVENLIGI (docs/52 s17: 8 trafo 1e7-5e7 kWh raporluyor)")
buyuk_tr = tr.groupby("tanim").tuketim.max()
riskli = set(buyuk_tr[buyuk_tr > 1e6].index)
mr = te.tanim.isin(riskli).values
print(f"   train'de >1e6 kWh gormus trafolar: {len(riskli)} adet, testte {mr.sum():,} satir")
if mr.sum():
    print(
        f"   m4 bu satirlarda maks {m4.tuketim.values[mr].max():,.0f} | v102 maks {v102.tuketim.values[mr].max():,.0f}"
    )
