"""EKSEN 4c: sifir-gecmisli trafolar UYANIR mi? P(uyanma) ozellikleri + optimal c."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *
from m17_lab import grupla, ozellik

tr = yukle()
kayit = []
for kesim in KESIMLER + ["2025-12-31", "2026-01-31"]:
    uf = {"2025-12-31": 3, "2026-01-31": 2}.get(kesim, 4)
    k = pd.Timestamp(kesim)
    gec, hed = hazirla(tr, kesim, uf)
    oz = ozellik(gec, kesim)
    gr = grupla(oz)
    zg = gr[gr.isin(["A_tum_sifir", "C_son28_sifir", "D_son7_sifir"])].index
    h = hed[hed.tanim.isin(zg)]
    if len(h) == 0:
        continue
    per = h.groupby("tanim").agg(
        hed_ly=("ly", "mean"), hed_max=("tuketim", "max"), n=("ly", "size")
    )
    per = per.join(oz[["nsat", "maxt", "bosluk", "omur", "ly_all"]]).join(gr.rename("grup"))
    per["guc"] = h.groupby("tanim").guc.first()
    per["kesim"] = kesim
    per["uyandi"] = (per.hed_max >= 1).astype(int)
    # "sifir suresi": tum-sifir icin omur, diger icin son sifir olmayan gunden beri gecen sure
    kayit.append(per.reset_index())
K = pd.concat(kayit)
print(
    f"toplam sifir-durumlu trafo-kesim gozlemi: {len(K)}, uyanma orani %{100 * K.uyandi.mean():.1f}"
)
print("\ngruba gore:")
print(
    K.groupby("grup").agg(
        n=("uyandi", "size"), uyanma=("uyandi", "mean"), hed_ly=("hed_ly", "mean")
    )
)
print("\nA grubu: 'sifir omru' (omur gun) kovalarina gore uyanma:")
A = K[K.grup == "A_tum_sifir"]
A = A.assign(kova=pd.cut(A.omur, [0, 15, 30, 60, 120, 250, 1000]))
print(
    A.groupby("kova", observed=True).agg(
        n=("uyandi", "size"), uyanma=("uyandi", "mean"), hed_ly_ort=("hed_ly", "mean")
    )
)
print("\nA grubu: guc kovalarina gore:")
A2 = A.assign(gk=pd.cut(A.guc, [0, 100, 250, 400, 630, 1000, 1e9]))
print(
    A2.groupby("gk", observed=True).agg(
        n=("uyandi", "size"), uyanma=("uyandi", "mean"), hed_ly_ort=("hed_ly", "mean")
    )
)
print("\nC/D grubu: gecmisteki canli seviye (ly_all) ve bosluk:")
CD = K[K.grup != "A_tum_sifir"]
print(
    CD.groupby("grup").agg(
        n=("uyandi", "size"),
        uyanma=("uyandi", "mean"),
        hed_ly=("hed_ly", "mean"),
        ly_all=("ly_all", "mean"),
    )
)
# satir agirlikli optimal sabit (tum kesimler birlikte)
print("\n=== satir-agirlikli optimal sabit c (tum kesim havuzu) ===")
for gr_ in ["A_tum_sifir", "C_son28_sifir", "D_son7_sifir"]:
    s = K[K.grup == gr_]
    if len(s) == 0:
        continue
    w = s.n.values
    y = s.hed_ly.values
    # E[ly] agirlikli; ama satir ici varyans da var -> yaklasim: c* = agirlikli ort
    c = float((w * y).sum() / w.sum())
    print(
        f"{gr_:16s} trafo={len(s):4d} satir={int(w.sum()):7,d} uyanma %{100 * s.uyandi.mean():.1f} optimal c~{c:.3f}"
    )
json_yaz(
    "eksen4c_uyanma",
    {
        "grup_ozet": K.groupby("grup")
        .agg(n=("uyandi", "size"), uyanma=("uyandi", "mean"), hed_ly=("hed_ly", "mean"))
        .to_dict(),
        "A_omur": {
            str(i): {"n": int(r.n), "uyanma": float(r.uyanma), "hed_ly": float(r.hed_ly_ort)}
            for i, r in A.groupby("kova", observed=True)
            .agg(n=("uyandi", "size"), uyanma=("uyandi", "mean"), hed_ly_ort=("hed_ly", "mean"))
            .iterrows()
        },
    },
)
