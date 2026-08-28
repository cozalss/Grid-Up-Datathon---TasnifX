"""Kestirimci laboratuvari: ozellik tablosu + parcali degerlendirme."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *


def ozellik(gec, kesim):
    """Trafo bazli gecmis ozellikleri. Pencereler hem KESIM hem KENDI son tarihine gore."""
    k = pd.Timestamp(kesim)
    gg = gec.groupby("tanim")
    oz = pd.DataFrame(
        {
            "nsat": gg.size(),
            "maxt": gg.tuketim.max(),
            "sonT": gg.tarih.max(),
            "ilkT": gg.tarih.min(),
            "ly_all": gg.ly.mean(),
            "ly_std": gg.ly.std(),
        }
    )
    oz["bosluk"] = (k - oz.sonT).dt.days
    oz["omur"] = (oz.sonT - oz.ilkT).dt.days + 1
    g2 = gec.join(oz[["sonT"]], on="tanim")
    g2["yas_k"] = (k - g2.tarih).dt.days  # kesime gore yas
    g2["yas_s"] = (g2.sonT - g2.tarih).dt.days  # kendi son tarihine gore yas
    for W in [3, 7, 14, 28, 56, 91, 182]:
        oz[f"k{W}"] = g2[g2.yas_k < W].groupby("tanim").ly.mean()
        oz[f"s{W}"] = g2[g2.yas_s < W].groupby("tanim").ly.mean()
        oz[f"kmax{W}"] = g2[g2.yas_k < W].groupby("tanim").tuketim.max()
        oz[f"smax{W}"] = g2[g2.yas_s < W].groupby("tanim").tuketim.max()
        oz[f"kn{W}"] = g2[g2.yas_k < W].groupby("tanim").size()
    return oz


def grupla(oz):
    g = pd.Series("E_normal", index=oz.index)
    g[~(oz.smax7 >= 1)] = "D_son7_sifir"
    g[~(oz.smax28 >= 1)] = "C_son28_sifir"
    g[oz.bosluk > 14] = "B_bayat"
    g[oz.maxt < 1] = "A_tum_sifir"
    return g


def degerlendir(hed, pred, oz, gr, ad, yaz=True):
    e2 = (np.asarray(pred, float) - hed.ly.values) ** 2
    tot = e2.mean()
    gh = hed.tanim.map(gr).values
    d = {"rmsle": float(np.sqrt(tot))}
    parts = []
    for g in ["A_tum_sifir", "B_bayat", "C_son28_sifir", "D_son7_sifir", "E_normal"]:
        m = gh == g
        if m.sum() == 0:
            continue
        d[g] = {
            "n": int(m.sum()),
            "rmsle": float(np.sqrt(e2[m].mean())),
            "kutle": float(e2[m].sum() / e2.sum()),
        }
        parts.append(f"{g[0]}:{np.sqrt(e2[m].mean()):.3f}(%{100 * e2[m].sum() / e2.sum():.0f})")
    if yaz:
        print(f"{ad:34s} {np.sqrt(tot):8.4f}  " + " ".join(parts))
    return d


if __name__ == "__main__":
    tr = yukle()
    res = {}
    for kesim in KESIMLER:
        gec, hed = hazirla(tr, kesim)
        oz = ozellik(gec, kesim)
        gr = grupla(oz)
        kok = float(gec.ly.mean())
        print(f"\n===== {kesim} sicak n={len(hed):,} =====")
        print(f"{'kestirimci':34s} {'RMSLE':>8s}  parcalar A/B/C/D/E rmsle(%kutle)")
        res[kesim] = {}
        # 1) kesim-cipali pencereler
        for W in [7, 28, 91]:
            p = geri_dolgu(hed, oz[f"k{W}"], oz.ly_all, kok=kok)
            res[kesim][f"kesim{W}"] = degerlendir(hed, p, oz, gr, f"kesim-cipa mean{W}")
        # 2) kendi-cipali pencereler (bayat icin onemli)
        for W in [7, 28, 91]:
            p = geri_dolgu(hed, oz[f"s{W}"], oz.ly_all, kok=kok)
            res[kesim][f"kendi{W}"] = degerlendir(hed, p, oz, gr, f"kendi-cipa mean{W}")
        # 3) harman 0.8*k7 + 0.2*all
        p = 0.8 * geri_dolgu(hed, oz.k7, oz.ly_all, kok=kok) + 0.2 * geri_dolgu(
            hed, oz.ly_all, kok=kok
        )
        res[kesim]["harman"] = degerlendir(hed, p, oz, gr, "0.8*k7+0.2*all")
        # 4) kendi-cipa 7 + harman
        p = 0.8 * geri_dolgu(hed, oz.s7, oz.ly_all, kok=kok) + 0.2 * geri_dolgu(
            hed, oz.ly_all, kok=kok
        )
        res[kesim]["harman_kendi"] = degerlendir(hed, p, oz, gr, "0.8*s7+0.2*all")
    json_yaz("eksen_lab_cipa", res)
