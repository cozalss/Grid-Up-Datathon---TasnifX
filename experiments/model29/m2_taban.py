"""Geri-testte taban cizgileri ve hata ayristirmasi."""

import numpy as np
import pandas as pd
from m1_geriteste import kes, yukle


def ayristir(hed, p, ad):
    L = (np.log1p(np.clip(p, 0, None)) - np.log1p(hed.tuketim.values)) ** 2
    m = hed.soguk.values
    tot = L.mean()
    print(
        f"{ad:32s} RMSLE {np.sqrt(tot):.4f} | soguk {np.sqrt(L[m].mean()):.4f} "
        f"(kutle %{100 * L[m].sum() / L.sum():.1f}) | sicak {np.sqrt(L[~m].mean()):.4f} "
        f"(kutle %{100 * L[~m].sum() / L.sum():.1f})"
    )
    return np.sqrt(tot)


for kesim in ["2025-10-31", "2025-11-30"]:
    print(f"\n===== KESIM {kesim} =====")
    tr = yukle()
    gec, hed = kes(tr, kesim)
    gec = gec.copy()
    gec["ly"] = np.log1p(gec.tuketim)
    hed = hed.copy()

    # 0) global sabit (log uzayinda ortalama = RMSLE optimumu)
    g0 = gec.ly.mean()
    ayristir(hed, np.expm1(g0) * np.ones(len(hed)), "global sabit")

    # 1) trafo bazli log-ortalama, soguk icin global
    tm = gec.groupby("tanim").ly.mean()
    p = hed.tanim.map(tm).fillna(g0).values
    ayristir(hed, np.expm1(p), "trafo log-ort (soguk=global)")

    # 2) trafo bazli SON 28 gun log-ortalama
    son = gec[gec.tarih > pd.Timestamp(kesim) - pd.Timedelta(days=28)]
    tm28 = son.groupby("tanim").ly.mean()
    p = hed.tanim.map(tm28).fillna(hed.tanim.map(tm)).fillna(g0).values
    ayristir(hed, np.expm1(p), "trafo son28 (soguk=global)")

    # 3) soguk icin grup kestirimcileri
    for anah in [["guc"], ["ilce"], ["guc", "bolge"], ["guc", "ilce"], ["guc", "il"]]:
        gm = gec.groupby(anah).ly.mean()
        idx = pd.MultiIndex.from_frame(hed[anah]) if len(anah) > 1 else hed[anah[0]]
        pc = pd.Series(idx).map(gm).values if len(anah) == 1 else gm.reindex(idx).values
        pc = np.where(np.isnan(pc.astype(float)), g0, pc.astype(float))
        p = np.where(
            hed.soguk.values, pc, hed.tanim.map(tm28).fillna(hed.tanim.map(tm)).fillna(g0).values
        )
        ayristir(hed, np.expm1(p), f"son28 + soguk~{'+'.join(anah)}")
