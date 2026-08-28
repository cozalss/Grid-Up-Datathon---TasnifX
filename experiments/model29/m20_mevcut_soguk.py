"""Mevcut en iyi gonderim (v102) soguk trafolara ne seviye veriyor?
Geri-testte OPTIMUM soguk seviyesi ne? Aradaki fark ucretsiz kazanc mi?"""

import os

import numpy as np
import pandas as pd
from m1_geriteste import KOK, kes, yukle

te = pd.read_csv(
    os.path.join(KOK, "data/raw/test.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
tr = pd.read_csv(
    os.path.join(KOK, "data/raw/train.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
sub = pd.read_csv(os.path.join(KOK, "submissions/tuketim_v102_kappa_optimum.csv"))
assert (sub.id.values == te.id.values).all()
te["p"] = sub.tuketim.values
te["lp"] = np.log1p(te.p)
te["soguk"] = ~te.tanim.isin(set(tr.tanim))

print("=== MEVCUT GONDERIM (v102) TEST UZERINDE ===")
for ad, m in [("SOGUK", te.soguk), ("SICAK", ~te.soguk)]:
    h = te[m]
    print(
        f"{ad}: {len(h):,} satir (%{100 * len(h) / len(te):.1f}) | tahmin log-ort {h.lp.mean():.4f} "
        f"std {h.lp.std():.4f} | <1kWh tahmini %{100 * (h.p < 1).mean():.2f} | =0 %{100 * (h.p <= 0.001).mean():.2f}"
    )
print()
# geri-testte GERCEK soguk seviyesi
tr2 = yukle()
print("=== GERI-TESTTE GERCEK SOGUK SEVIYESI (satir-agirlikli log1p ortalamasi) ===")
for kesim in ["2025-08-31", "2025-09-30", "2025-10-31", "2025-11-30"]:
    gec, hed = kes(tr2, kesim)
    gec = gec.copy()
    gec["ly"] = np.log1p(gec.tuketim)
    hed = hed.copy()
    hed["ly"] = np.log1p(hed.tuketim)
    s = hed[hed.soguk]
    w = hed[~hed.soguk]
    gucm = gec.groupby("guc").ly.mean()
    tahm = s.guc.map(gucm).fillna(gec.ly.mean())
    print(
        f"{kesim}: soguk GERCEK ort {s.ly.mean():.4f} | guc-grubu TAHMINI ort {tahm.mean():.4f} "
        f"| fark {tahm.mean() - s.ly.mean():+.4f} | sicak gercek {w.ly.mean():.4f}"
    )
    # sabit kaydirma optimumu
    e0 = np.sqrt(((tahm.values - s.ly.values) ** 2).mean())
    d = (s.ly.values - tahm.values).mean()
    e1 = np.sqrt(((tahm.values + d - s.ly.values) ** 2).mean())
    print(f"          soguk RMSLE {e0:.4f} -> sabit kaydirma {d:+.4f} ile {e1:.4f}")
