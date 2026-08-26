"""CURUTME -- HEDEF SATIRLARIN TAMAMINDA (500.295) olcum: SATIR AGIRLIKLI b.
Ayrica v62'nin gercekten neyi degistirdigini denetle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path("C:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX")

tr = pd.read_csv(
    KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
)
tr["r"] = np.log1p(tr["tuketim"].clip(lower=0.0)) - np.log1p(tr["guc"])
te = pd.read_csv(
    KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
)
v55 = pd.read_csv(KOK / "submissions/tuketim_v55_gunolcek.csv", encoding="utf-8")
te = te.merge(v55, on="id", validate="one_to_one")
te["r"] = np.log1p(te["tuketim"].clip(lower=0.0)) - np.log1p(te["guc"])

SICAK = set(tr["tanim"].unique())
q1_26 = tr[(tr["tarih"] >= "2026-01-01") & (tr["tarih"] <= "2026-03-31")]
CANLI = set(q1_26.loc[q1_26["tuketim"] > 0, "tanim"].unique())
HEDEF = SICAK & CANLI
m_hedef = te["tanim"].isin(HEDEF)
print(
    f"HEDEF trafo {len(HEDEF):,} | HEDEF satir {int(m_hedef.sum()):,} / {len(te):,}"
    f"  (%{100 * m_hedef.mean():.2f})"
)

# --- v62 denetimi
v62 = pd.read_csv(KOK / "submissions/tuketim_v62_seviye06.csv", encoding="utf-8")
d = np.log1p(v62["tuketim"].to_numpy()) - np.log1p(v55["tuketim"].to_numpy())
print(
    f"\nv62 - v55: degisen satir {int((np.abs(d) > 1e-9).sum()):,}"
    f"  ort kayma {d[np.abs(d) > 1e-9].mean():+.6f}"
    f"  min {d.min():+.6f} maks {d.max():+.6f}"
)
print(
    f"   HEDEF tanimina uyan degisen satir: {int(((np.abs(d) > 1e-9) & m_hedef.to_numpy()).sum()):,}"
)
sic = te["tanim"].isin(SICAK).to_numpy()
print(
    f"   SICAK satir sayisi (v62'nin gercek hedefi): {int(sic.sum()):,} (%{100 * sic.mean():.2f})"
)


# --- pencere ortalamalari, TUM sicak trafolar icin
def win(df, a, b, col="r"):
    q = df[(df["tarih"] >= a) & (df["tarih"] <= b)]
    return q.groupby("tanim", observed=True)[col].agg(["mean", "size"])


w1 = win(tr, "2025-01-01", "2025-03-31")
w2 = win(tr, "2026-01-01", "2026-03-31")
w3 = win(tr, "2025-04-01", "2025-07-31")
w4 = win(te, "2026-04-01", "2026-07-31")
D = pd.DataFrame(
    {
        "L25q1": w1["mean"],
        "n1": w1["size"],
        "L26q1": w2["mean"],
        "n2": w2["size"],
        "L25yz": w3["mean"],
        "n3": w3["size"],
        "P26yz": w4["mean"],
        "n4": w4["size"],
    }
)
D = D[D.index.isin(HEDEF)]
tam = D.dropna(subset=["L25q1", "L26q1", "L25yz", "P26yz"])
# yeterli gun sarti
tam = tam[(tam["n1"] >= 20) & (tam["n2"] >= 20) & (tam["n3"] >= 20) & (tam["n4"] >= 20)]
print(f"\nHEDEF trafolarin dort pencerede de >=20 gunu olan: {len(tam):,} / {len(HEDEF):,}")
kaps = te.loc[m_hedef & te["tanim"].isin(tam.index)].shape[0]
print(
    f"   bunlarin kapsadigi HEDEF satir: {kaps:,} / {int(m_hedef.sum()):,}  (%{100 * kaps / m_hedef.sum():.1f})"
)

tam = tam.copy()
tam["ACIK"] = (tam["L25yz"] - tam["L25q1"]) - (tam["P26yz"] - tam["L26q1"])
tam["buyume_q1"] = tam["L26q1"] - tam["L25q1"]
tam["ima_yoy"] = tam["P26yz"] - tam["L25yz"]
tam["w"] = te.loc[m_hedef].groupby("tanim").size().reindex(tam.index).fillna(0)


def ag(x, w):
    return float((x * w).sum() / w.sum())


print("\n=== TUM HEDEF (panel disi dahil) vs SABIT PANEL ===")
print(f"{'olcut':>12} {'trafo-ort':>11} {'SATIR-agirlikli':>16} {'medyan':>10}")
for c in ("buyume_q1", "ima_yoy", "ACIK"):
    print(f"{c:>12} {tam[c].mean():+11.5f} {ag(tam[c], tam['w']):+16.5f} {tam[c].median():+10.5f}")

print("\nKIRPMA TABLOSU -- TUM HEDEF, SATIR AGIRLIKLI ACIK")
s = tam.reindex(tam["ACIK"].abs().sort_values(ascending=False).index)
for K in (0, 1, 5, 10, 25, 50, 100, 200, 400):
    q = s.iloc[K:]
    print(
        f"  K={K:>4} n={len(q):>5} ACIK trafo-ort {q['ACIK'].mean():+.5f}"
        f"  satir-ag {ag(q['ACIK'], q['w']):+.5f}  medyan {q['ACIK'].median():+.5f}"
    )

# --- panel-ici / panel-disi ayrimi
ay = tr.copy()
ay["ay"] = ay["tarih"].dt.to_period("M")
g = (
    ay.groupby(["tanim", "ay"], observed=True)
    .agg(nz=("tuketim", lambda s: int((s > 0).sum())))
    .reset_index()
)
ays = ay.groupby("tanim")["ay"].nunique()
nza = g[g["nz"] > 0].groupby("tanim")["ay"].nunique()
PANEL = set(ays[ays == 15].index) & set(nza[nza == 15].index)
tam["panelde"] = tam.index.isin(PANEL)
print("\nPANEL ICI vs DISI (hedef kumede)")
for k, q in tam.groupby("panelde"):
    print(
        f"  panelde={k}  n={len(q):>5}  satir={int(q['w'].sum()):>7,}"
        f"  ACIK trafo-ort {q['ACIK'].mean():+.5f}  satir-ag {ag(q['ACIK'], q['w']):+.5f}"
        f"  medyan {q['ACIK'].median():+.5f}  buyumeQ1 ort {q['buyume_q1'].mean():+.5f}"
    )

tam.to_csv(KOK / "reports" / "_c8_hedef.csv")
print("\nyazildi reports/_c8_hedef.csv")
