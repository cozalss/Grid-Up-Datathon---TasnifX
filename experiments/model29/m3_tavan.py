"""Soguk ve sicak icin ORACLE tavani: ne kadari indirgenemez gurultu?"""

import numpy as np
from m1_geriteste import kes, yukle

tr = yukle()
for kesim in ["2025-10-31", "2025-11-30"]:
    gec, hed = kes(tr, kesim)
    hed = hed.copy()
    hed["ly"] = np.log1p(hed.tuketim)
    print(f"\n===== {kesim} =====")
    for ad, m in [("SOGUK", hed.soguk.values), ("SICAK", ~hed.soguk.values)]:
        h = hed[m]
        # oracle 1: trafo bazli sabit (hedef penceredeki kendi log-ortalamasi)
        o1 = h.groupby("tanim").ly.transform("mean")
        # oracle 2: trafo x ay
        o2 = h.groupby(["tanim", h.tarih.dt.to_period("M")]).ly.transform("mean")
        # oracle 3: trafo x haftanin gunu x ay
        o3 = h.groupby(["tanim", h.tarih.dt.to_period("M"), h.tarih.dt.dayofweek]).ly.transform(
            "mean"
        )
        print(f"{ad}: n={len(h):,} trafo={h.tanim.nunique():,}")
        print(f"   oracle trafo-sabit      RMSLE {np.sqrt(((o1 - h.ly) ** 2).mean()):.4f}")
        print(f"   oracle trafo x ay       RMSLE {np.sqrt(((o2 - h.ly) ** 2).mean()):.4f}")
        print(f"   oracle trafo x ay x gun RMSLE {np.sqrt(((o3 - h.ly) ** 2).mean()):.4f}")
        print(
            f"   ly dagilimi: ort {h.ly.mean():.3f} std {h.ly.std():.3f} "
            f"sifir %{100 * (h.tuketim == 0).mean():.2f} <1kWh %{100 * (h.tuketim < 1).mean():.2f}"
        )
        tm = h.groupby("tanim").ly.mean()
        print(
            f"   trafo-ort dagilimi: std {tm.std():.3f} | trafo-ici std (ort) {h.groupby('tanim').ly.std().mean():.3f}"
        )
