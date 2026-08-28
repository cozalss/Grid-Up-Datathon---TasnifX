"""Soguk tahminlerde ASIRI YAYILIM var mi? Optimum buzulme (shrinkage) katsayisi."""

import numpy as np
import pandas as pd
from m1_geriteste import kes, yukle

tr2 = yukle()
print(
    f"{'kesim':12s} {'kestirimci':22s} {'sigma_p':>8s} {'sigma_t':>8s} {'rho':>7s} "
    f"{'kappa*':>7s} {'RMSLE_0':>8s} {'RMSLE*':>8s} {'kazanc':>8s}"
)
sonuc = {}
for kesim in ["2025-08-31", "2025-09-30", "2025-10-31", "2025-11-30"]:
    gec, hed = kes(tr2, kesim)
    gec = gec.copy()
    gec["ly"] = np.log1p(gec.tuketim)
    hed = hed.copy()
    hed["ly"] = np.log1p(hed.tuketim)
    gg = gec.ly.mean()
    s = hed[hed.soguk]
    t = s.ly.values
    kest = {
        "guc grubu": s.guc.map(gec.groupby("guc").ly.mean()).fillna(gg).values,
        "log(guc) dogrusal": np.polyval(
            np.polyfit(np.log(gec.guc.clip(lower=1)), gec.ly, 1), np.log(s.guc.clip(lower=1))
        ),
        "guc+bolge": pd.MultiIndex.from_frame(s[["guc", "bolge"]])
        .map(gec.groupby(["guc", "bolge"]).ly.mean())
        .to_numpy(dtype=float),
    }
    for ad, p in kest.items():
        p = np.where(np.isnan(p), gg, p)
        c = p.mean()
        d = p - c
        # kappa* = argmin ||c + k*d - t||^2  ; ayrica merkezi de optimize et
        k = float(np.dot(d, t - t.mean()) / np.dot(d, d)) if np.dot(d, d) > 0 else 0.0
        e0 = np.sqrt(((p - t) ** 2).mean())
        best_c = t.mean()
        e1 = np.sqrt(((best_c + k * d - t) ** 2).mean())
        # merkezi KORUYARAK sadece kappa (gercekci senaryo: merkezi bilmiyoruz)
        k2 = float(np.dot(d, t - c) / np.dot(d, d)) if np.dot(d, d) > 0 else 0.0
        e2 = np.sqrt(((c + k2 * d - t) ** 2).mean())
        rho = np.corrcoef(p, t)[0, 1]
        print(
            f"{kesim:12s} {ad:22s} {d.std():8.4f} {t.std():8.4f} {rho:7.4f} "
            f"{k2:7.4f} {e0:8.4f} {e2:8.4f} {e0 - e2:8.4f}"
        )
        sonuc.setdefault(ad, []).append(k2)
print("\nkappa* ortalamalari (kesimler arasi):")
for ad, v in sonuc.items():
    print(
        f"  {ad:22s} {np.mean(v):.4f}  (std {np.std(v):.4f})  ->  {'ASIRI YAYILIM' if np.mean(v) < 0.9 else 'yayilim uygun'}"
    )
