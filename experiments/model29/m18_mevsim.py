"""EKSEN 2: mevsimsellik. Ay etkisi ne kadar buyuk, yildan yila kararli mi?"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *

tr = yukle()
tr["ly"] = np.log1p(tr.tuketim)
tr["ym"] = tr.tarih.dt.to_period("M")
tr["ay"] = tr.tarih.dt.month
# saglikli trafolar: 2025-01..2026-03 tam dolu, hic sifir yok
say = tr.groupby("tanim").size()
mx = tr.groupby("tanim").tuketim.min()
tam = say[say >= 450].index
sub = tr[tr.tanim.isin(tam)].copy()
sub = sub[sub.groupby("tanim").tuketim.transform("min") > 0]
print("tam+sifirsiz trafo:", sub.tanim.nunique())
sub["dev"] = sub.ly - sub.groupby("tanim").ly.transform("mean")
prof = sub.groupby("ym").dev.mean()
print("\nAY ETKISI (trafo-ici sapma ortalamasi, log uzayi):")
for m_, v in prof.items():
    print(f"  {m_} {v:+.4f}")
print(
    f"ay etkisi std: {prof.std():.4f}  (trafo-ici gunluk std ~{sub.groupby('tanim').ly.std().mean():.3f})"
)
# yildan yila kararlilik: 2025-01..03 vs 2026-01..03
print("\nYIL-UZERI KARARLILIK (2025 vs 2026, Ocak-Mart):")
for m in [1, 2, 3]:
    a = prof.get(pd.Period(f"2025-{m:02d}"))
    b = prof.get(pd.Period(f"2026-{m:02d}"))
    print(f"  ay {m}: 2025 {a:+.4f}  2026 {b:+.4f}  fark {b - a:+.4f}")
# trafo bazli ay etkisi kararli mi? 2025Q1 vs 2026Q1 trafo sapmalari korelasyonu
q25 = (
    sub[sub.ym.isin([pd.Period("2025-01"), pd.Period("2025-02"), pd.Period("2025-03")])]
    .groupby("tanim")
    .dev.mean()
)
q26 = (
    sub[sub.ym.isin([pd.Period("2026-01"), pd.Period("2026-02"), pd.Period("2026-03")])]
    .groupby("tanim")
    .dev.mean()
)
print(
    f"\ntrafo-bazli Q1 sapmasi: 2025 std {q25.std():.3f}, 2026 std {q26.std():.3f}, korelasyon {q25.corr(q26):.3f}"
)
# NISAN-TEMMUZ (gercek hedef) etkisi
nt = [pd.Period(f"2025-{m:02d}") for m in [4, 5, 6, 7]]
print(f"\nGERCEK HEDEF Nisan-Temmuz 2025 ortalama sapmasi: {prof[nt].mean():+.4f}")
mar = prof[pd.Period("2026-03")]
print(
    f"2026-03 (son ay) sapmasi: {mar:+.4f} -> beklenen Nis-Tem kaymasi: {prof[nt].mean() - prof[pd.Period('2025-03')]:+.4f} (2025 ici Mar->NisTem)"
)
# trafo bazli Nis-Tem sapmasi dagilimi
nts = sub[sub.ym.isin(nt)].groupby("tanim").dev.mean()
mrt = sub[sub.ym == pd.Period("2025-03")].groupby("tanim").dev.mean()
print(
    f"trafo bazli (NisTem2025 - Mar2025) farki: ort {(nts - mrt).mean():+.3f} std {(nts - mrt).std():.3f}"
)
json_yaz("eksen2_ay_profili", {str(a): float(v) for a, v in prof.items()})
json_yaz(
    "eksen2_ozet",
    {
        "ay_etkisi_std": float(prof.std()),
        "trafo_ici_gunluk_std": float(sub.groupby("tanim").ly.std().mean()),
        "q1_korelasyon": float(q25.corr(q26)),
        "nistem2025_sapma": float(prof[nt].mean()),
        "mar2025_sapma": float(prof[pd.Period("2025-03")]),
        "mar2026_sapma": float(mar),
    },
)
