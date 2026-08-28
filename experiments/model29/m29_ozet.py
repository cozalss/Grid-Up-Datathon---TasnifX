"""Nihai kestirimcinin parca ayristirmasi + kalan tavan + JSON ozeti."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *
from m17_lab import grupla, ozellik


def tahmin(gec, hed, kesim):
    oz = ozellik(gec, kesim)
    gr = grupla(oz)
    kok = float(gec.ly.mean())
    g = hed.tanim.map(gr).values
    k7 = geri_dolgu(hed, oz.k7, oz.ly_all, kok=kok)
    allm = geri_dolgu(hed, oz.ly_all, kok=kok)
    s28 = geri_dolgu(hed, oz.s28, oz.ly_all, kok=kok)
    gd = gec.copy()
    gd["dw"] = gd.tarih.dt.dayofweek
    gd["dev"] = gd.ly - gd.groupby("tanim").ly.transform("mean")
    dwt = gd[gd.tanim.map(oz.maxt) >= 1].groupby(["tanim", "dw"]).dev.mean()
    dw = np.nan_to_num(
        pd.Series(
            dwt.reindex(
                pd.MultiIndex.from_arrays([hed.tanim.values, hed.tarih.dt.dayofweek.values])
            ).values
        ).values.astype(float)
    )
    p = np.where(g == "B_bayat", 0.7 * s28 + 0.3 * allm, 0.75 * k7 + 0.25 * allm) + 0.5 * dw
    for gg, cc in [("A_tum_sifir", 0.6), ("C_son28_sifir", 1.1), ("D_son7_sifir", 1.3)]:
        p = np.where(g == gg, cc, p)
    return p, g


tr = yukle()
TUM = KESIMLER + ["2025-03-31", "2025-12-31"]
oz_res = {}
print("== sifir-sabitlerinin (A.6/C1.1/D1.3) katkisi, tum kesimler ==")
for kesim in TUM:
    uf = 3 if kesim == "2025-12-31" else 4
    gec, hed = hazirla(tr, kesim, uf)
    y = hed.ly.values
    p, g = tahmin(gec, hed, kesim)
    # sabitsiz surum
    ozf = ozellik(gec, kesim)
    grf = grupla(ozf)
    kok = float(gec.ly.mean())
    k7 = geri_dolgu(hed, ozf.k7, ozf.ly_all, kok=kok)
    allm = geri_dolgu(hed, ozf.ly_all, kok=kok)
    s28 = geri_dolgu(hed, ozf.s28, ozf.ly_all, kok=kok)
    gd = gec.copy()
    gd["dw"] = gd.tarih.dt.dayofweek
    gd["dev"] = gd.ly - gd.groupby("tanim").ly.transform("mean")
    dwt = gd[gd.tanim.map(ozf.maxt) >= 1].groupby(["tanim", "dw"]).dev.mean()
    dw = np.nan_to_num(
        pd.Series(
            dwt.reindex(
                pd.MultiIndex.from_arrays([hed.tanim.values, hed.tarih.dt.dayofweek.values])
            ).values
        ).values.astype(float)
    )
    p_ns = np.where(g == "B_bayat", 0.7 * s28 + 0.3 * allm, 0.75 * k7 + 0.25 * allm) + 0.5 * dw
    r = float(np.sqrt(((y - p) ** 2).mean()))
    rns = float(np.sqrt(((y - p_ns) ** 2).mean()))
    # kalan hata ayristirmasi
    e2 = (p - y) ** 2
    tot = e2.sum()
    par = {
        gg: {
            "n": int((g == gg).sum()),
            "rmsle": float(np.sqrt(e2[g == gg].mean())),
            "kutle": float(e2[g == gg].sum() / tot),
        }
        for gg in np.unique(g)
    }
    # grup-oracle tavani: her grup icin hedefteki trafo-sabit oracle
    ora = hed.groupby("tanim").ly.transform("mean").values
    orac = float(np.sqrt(((ora - y) ** 2).mean()))
    oz_res[kesim] = {
        "nihai": r,
        "sifir_sabitsiz": rns,
        "sabit_kazanci": rns - r,
        "trafo_sabit_oracle": orac,
        "parcalar": par,
    }
    print(
        f"{kesim}: sabitli {r:.4f} | sabitsiz {rns:.4f} | kazanc {rns - r:+.4f} | trafo-sabit ORACLE {orac:.4f}"
    )
    print(
        "    "
        + "  ".join(
            f"{gg[0]}:{v['rmsle']:.3f}(n={v['n']:,},%{100 * v['kutle']:.0f})"
            for gg, v in par.items()
        )
    )
json_yaz("nihai_ayristirma", oz_res)
print(
    f"\n4 ana kesim ort: sabitli {np.mean([oz_res[k]['nihai'] for k in KESIMLER]):.4f} "
    f"sabitsiz {np.mean([oz_res[k]['sifir_sabitsiz'] for k in KESIMLER]):.4f} "
    f"oracle {np.mean([oz_res[k]['trafo_sabit_oracle'] for k in KESIMLER]):.4f}"
)
