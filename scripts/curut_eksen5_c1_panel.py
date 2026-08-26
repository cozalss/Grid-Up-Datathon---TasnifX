"""CURUTME 1 -- sabit panelde YoY buyume olcumu, SIFIR KONFONDU ayrik."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path("C:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX")

tr = pd.read_csv(
    KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
)
print("train", tr.shape, tr["tarih"].min(), tr["tarih"].max())

tr["ay"] = tr["tarih"].dt.to_period("M")
tr["r"] = np.log1p(tr["tuketim"]) - np.log1p(tr["guc"])
tr["sifir"] = (tr["tuketim"] <= 0).astype(int)

# --- PANEL: 15 ayin hepsinde kayit + her ay en az bir sifir-olmayan gun
g = (
    tr.groupby(["tanim", "ay"], observed=True)
    .agg(n=("r", "size"), nz=("sifir", lambda s: int((s == 0).sum())))
    .reset_index()
)
ay_sayisi = g.groupby("tanim")["ay"].nunique()
nz_ay = g[g["nz"] > 0].groupby("tanim")["ay"].nunique()
tam = set(ay_sayisi[ay_sayisi == 15].index) & set(nz_ay[nz_ay == 15].index)
print("PANEL n =", len(tam))

p = tr[tr["tanim"].isin(tam)].copy()

# --- aylik ortalama r, TRAFO ICINDE, iki surum: sifirlar DAHIL / HARIC
ay_dahil = p.groupby(["tanim", "ay"], observed=True)["r"].mean().unstack()
ph = p[p["tuketim"] > 0]
ay_haric = ph.groupby(["tanim", "ay"], observed=True)["r"].mean().unstack()
sifir_ay = p.groupby(["tanim", "ay"], observed=True)["sifir"].mean().unstack()

print("\nAYLIK ORT r  (panel geneli)")
print(f"{'ay':>9} {'sifirDAHIL':>11} {'sifirHARIC':>11} {'sifir_pay':>10}")
for a in ay_dahil.columns:
    print(
        f"{str(a):>9} {ay_dahil[a].mean():+11.4f} {ay_haric[a].mean():+11.4f} "
        f"{sifir_ay[a].mean() * 100:10.2f}"
    )

q1_25 = [pd.Period("2025-01"), pd.Period("2025-02"), pd.Period("2025-03")]
q1_26 = [pd.Period("2026-01"), pd.Period("2026-02"), pd.Period("2026-03")]

print("\nYoY Q1 (ay ay)")
print(f"{'ay':>6} {'DAHIL':>9} {'HARIC':>9} {'dSifirPay':>10}")
for a25, a26 in zip(q1_25, q1_26):
    d = (ay_dahil[a26] - ay_dahil[a25]).mean()
    h = (ay_haric[a26] - ay_haric[a25]).mean()
    ds = (sifir_ay[a26] - sifir_ay[a25]).mean()
    print(f"{a25.strftime('%b'):>6} {d:+9.4f} {h:+9.4f} {ds * 100:+10.2f}")

L25q1_d = ay_dahil[q1_25].mean(axis=1)
L26q1_d = ay_dahil[q1_26].mean(axis=1)
L25q1_h = ay_haric[q1_25].mean(axis=1)
L26q1_h = ay_haric[q1_26].mean(axis=1)
print(
    "\nQ1 YoY buyume  DAHIL: ort %+.5f medyan %+.5f"
    % ((L26q1_d - L25q1_d).mean(), (L26q1_d - L25q1_d).median())
)
print(
    "Q1 YoY buyume  HARIC: ort %+.5f medyan %+.5f"
    % ((L26q1_h - L25q1_h).mean(), (L26q1_h - L25q1_h).median())
)

out = pd.DataFrame(
    {
        "L25q1_d": L25q1_d,
        "L26q1_d": L26q1_d,
        "L25q1_h": L25q1_h,
        "L26q1_h": L26q1_h,
    }
)
for a in ay_dahil.columns:
    out["d_" + str(a)] = ay_dahil[a]
    out["h_" + str(a)] = ay_haric[a]
out.to_csv(KOK / "reports" / "_c1_panel.csv")
print("\nyazildi reports/_c1_panel.csv")
