"""CURUTME -- KALICILIK CAPASI: 2025 gecmisi GEREKMEYEN kurgu.
b_i = [L26q1_i + S] - P26yz_i ;  S = panel mevsimsel kaldirma (hava duzeltmeli).
Bu TUM hedef satirlari kapsar. Olculebilen / olculemeyen satirlarin karsilastirmasi."""

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
q126 = tr[(tr["tarih"] >= "2026-01-01") & (tr["tarih"] <= "2026-03-31")]
HEDEF = SICAK & set(q126.loc[q126["tuketim"] > 0, "tanim"].unique())
th = te[te["tanim"].isin(HEDEF)].copy()
print(f"HEDEF satir {len(th):,}  trafo {th['tanim'].nunique():,}")

L26q1 = q126.groupby("tanim", observed=True)["r"].agg(["mean", "size"])
P26 = th.groupby("tanim", observed=True)["r"].agg(["mean", "size"])
A = pd.DataFrame(
    {"L26q1": L26q1["mean"], "n26q1": L26q1["size"], "P26": P26["mean"], "nte": P26["size"]}
).dropna()
A["kald26"] = A["P26"] - A["L26q1"]

# 2025 referans kaldirma, SABIT PANEL
tr["ay"] = tr["tarih"].dt.to_period("M")
g = (
    tr.groupby(["tanim", "ay"], observed=True)
    .agg(nz=("tuketim", lambda s: int((s > 0).sum())))
    .reset_index()
)
ays = tr.groupby("tanim")["ay"].nunique()
nza = g[g["nz"] > 0].groupby("tanim")["ay"].nunique()
PANEL = set(ays[ays == 15].index) & set(nza[nza == 15].index)
pt = tr[tr["tanim"].isin(PANEL)]
k25 = (
    pt[(pt["tarih"] >= "2025-04-01") & (pt["tarih"] <= "2025-07-31")].groupby("tanim")["r"].mean()
    - pt[(pt["tarih"] >= "2025-01-01") & (pt["tarih"] <= "2025-03-31")].groupby("tanim")["r"].mean()
)
S25 = float(k25.mean())
print(f"2025 mevsimsel kaldirma S25 (panel ort) = {S25:+.5f}  medyan {k25.median():+.5f}")

# olculebilir mi?
w1 = tr[(tr["tarih"] >= "2025-01-01") & (tr["tarih"] <= "2025-03-31")].groupby("tanim")["r"].size()
w3 = tr[(tr["tarih"] >= "2025-04-01") & (tr["tarih"] <= "2025-07-31")].groupby("tanim")["r"].size()
A["olculebilir"] = (
    (A.index.map(w1).fillna(0) >= 20) & (A.index.map(w3).fillna(0) >= 20) & (A["n26q1"] >= 20)
)
A["w"] = A["nte"]


def ag(x, w):
    return float((x * w).sum() / w.sum())


print("\n=== v55'IN IMA ETTIGI Q1->YAZ KALDIRMASI (kald26), hedef kume ===")
for k, q in A.groupby("olculebilir"):
    print(
        f"  olculebilir={k}  trafo {len(q):>5}  satir {int(q['w'].sum()):>7,}"
        f"  kald26 satir-ag {ag(q['kald26'], q['w']):+.5f}  medyan {q['kald26'].median():+.5f}"
        f"  | ham b = S25 - kald26 = {S25 - ag(q['kald26'], q['w']):+.5f}"
    )
print(
    f"  TUMU        trafo {len(A):>5}  satir {int(A['w'].sum()):>7,}"
    f"  kald26 satir-ag {ag(A['kald26'], A['w']):+.5f}"
    f"  | ham b = {S25 - ag(A['kald26'], A['w']):+.5f}"
)

print("\n--- olculebilir / olculemez gruplarin GOZLENEBILIR ozellikleri")
gc = te.drop_duplicates("tanim").set_index("tanim")
A["guc"] = gc["guc"].reindex(A.index)
A["bolge"] = gc["lokasyon"].str.split(">").str[1].reindex(A.index)
for k, q in A.groupby("olculebilir"):
    print(
        f"  olculebilir={k}: guc medyan {q['guc'].median():>7.0f}  L26q1 ort {q['L26q1'].mean():+.4f}"
        f"  P26 ort {q['P26'].mean():+.4f}  test gun ort {q['nte'].mean():.1f}"
        f"  2026Q1 gun ort {q['n26q1'].mean():.1f}"
    )
    print("     bolge dagilimi:", q["bolge"].value_counts(normalize=True).round(3).to_dict())

# --- HAVA DUZELTMESI bandi
print("\n=== YANLILIK b = (S25 + hava) - kald26 ===")
for hava in (-0.0518, -0.0574, -0.0845):
    b_all = (S25 + hava) - ag(A["kald26"], A["w"])
    b_olc = (S25 + hava) - ag(A.loc[A["olculebilir"], "kald26"], A.loc[A["olculebilir"], "w"])
    b_yok = (S25 + hava) - ag(A.loc[~A["olculebilir"], "kald26"], A.loc[~A["olculebilir"], "w"])
    print(
        f"  hava {hava:+.4f}:  TUM {b_all:+.5f}   olculebilir {b_olc:+.5f}   olculemez {b_yok:+.5f}"
    )

A.to_csv(KOK / "reports" / "_c9_kalicilik.csv")
print("\nyazildi reports/_c9_kalicilik.csv")
