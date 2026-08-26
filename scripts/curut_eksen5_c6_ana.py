"""CURUTME ANA -- v55'in ima ettigi yaz seviyesi, YANLILIK b, kirpma tablosu,
genelleme, ULUSAL yuk capasi."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path("C:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX")
sys.path.insert(0, str(KOK / "src"))

# ================= 1. ULUSAL YUK (EPIAS) YoY =================
ep = pd.read_parquet(KOK / "data/external/epias/tuketim_saatlik.parquet")
ep["tarih"] = pd.to_datetime(ep["zaman"]).dt.normalize()
gun = ep.groupby("tarih")["consumption"].sum()
print("ULUSAL yuk kapsamasi", gun.index.min().date(), "->", gun.index.max().date())
lg = np.log(gun)
print("\nULUSAL AYLIK log-YoY (2026 vs 2025)")
for m in range(1, 8):
    a = lg[f"2025-{m:02d}"].mean()
    b = lg[f"2026-{m:02d}"].mean()
    print(f"   ay {m:02d}: {b - a:+.4f}")
q1 = lg["2025-01-01":"2025-03-31"].mean(), lg["2026-01-01":"2026-03-31"].mean()
yz = lg["2025-04-01":"2025-07-31"].mean(), lg["2026-04-01":"2026-07-31"].mean()
print(
    f"   Q1  YoY {q1[1] - q1[0]:+.4f}   Nis-Tem YoY {yz[1] - yz[0]:+.4f}"
    f"   FARK (yaz - Q1) {(yz[1] - yz[0]) - (q1[1] - q1[0]):+.4f}"
)

# ================= 2. PANEL + v55 =================
tr = pd.read_csv(
    KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
)
tr["r"] = np.log1p(tr["tuketim"].clip(lower=0.0)) - np.log1p(tr["guc"])
tr["ay"] = tr["tarih"].dt.to_period("M")
g = (
    tr.groupby(["tanim", "ay"], observed=True)
    .agg(n=("r", "size"), nz=("tuketim", lambda s: int((s > 0).sum())))
    .reset_index()
)
ays = g.groupby("tanim")["ay"].nunique()
nza = g[g["nz"] > 0].groupby("tanim")["ay"].nunique()
PANEL = set(ays[ays == 15].index) & set(nza[nza == 15].index)

te = pd.read_csv(
    KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
)
v55 = pd.read_csv(KOK / "submissions/tuketim_v55_gunolcek.csv", encoding="utf-8")
te = te.merge(v55, on="id", validate="one_to_one")
te["r"] = np.log1p(te["tuketim"].clip(lower=0.0)) - np.log1p(te["guc"])
te["ay"] = te["tarih"].dt.to_period("M")
print("\ntest", te.shape, te["tarih"].min().date(), te["tarih"].max().date())


def wmean(df, mask, col="r"):
    return df.loc[mask].groupby("tanim", observed=True)[col].mean()


trp = tr[tr["tanim"].isin(PANEL)]
tep = te[te["tanim"].isin(PANEL)]
print(f"panel {len(PANEL)} trafo; testte bulunan {tep['tanim'].nunique()}")

L25q1 = wmean(trp, (trp["tarih"] >= "2025-01-01") & (trp["tarih"] <= "2025-03-31"))
L26q1 = wmean(trp, (trp["tarih"] >= "2026-01-01") & (trp["tarih"] <= "2026-03-31"))
L25yz = wmean(trp, (trp["tarih"] >= "2025-04-01") & (trp["tarih"] <= "2025-07-31"))
P26yz = wmean(tep, (tep["tarih"] >= "2026-04-01") & (tep["tarih"] <= "2026-07-31"))

df = pd.concat(
    [L25q1.rename("L25q1"), L26q1.rename("L26q1"), L25yz.rename("L25yz"), P26yz.rename("P26yz")],
    axis=1,
).dropna()
print(f"dort pencerede de dolu: n={len(df)}")
df["buyume_q1"] = df["L26q1"] - df["L25q1"]
df["kald25"] = df["L25yz"] - df["L25q1"]
df["kald26"] = df["P26yz"] - df["L26q1"]
df["ima_yoy"] = df["P26yz"] - df["L25yz"]
df["ACIK"] = df["kald25"] - df["kald26"]
for c in ("buyume_q1", "kald25", "kald26", "ima_yoy", "ACIK"):
    print(f"  {c:>10}  ort {df[c].mean():+.5f}  medyan {df[c].median():+.5f}")

print("\nv55 IMA ETTIGI AYLIK YoY (panel, ay ay)")
for m in (4, 5, 6, 7):
    a = wmean(trp, trp["ay"] == pd.Period(f"2025-{m:02d}"))
    b = wmean(tep, tep["ay"] == pd.Period(f"2026-{m:02d}"))
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    print(
        f"   ay {m:02d}: 2025 {j['a'].mean():+.4f}  v55 2026 {j['b'].mean():+.4f}"
        f"  YoY {(j['b'] - j['a']).mean():+.4f}  (n={len(j)})"
    )

# ================= 3. HAVA DUZELTMESI =================
HAVA_YAZ = -0.0518  # c3'ten: b_hdd*dHDD + b_cdd*dCDD, Nis-Tem 2026 vs 2025
df["b_i"] = df["ACIK"] + HAVA_YAZ
print(f"\nYANLILIK b = ACIK + ({HAVA_YAZ:+.4f})")
print(f"   ort {df['b_i'].mean():+.5f}   medyan {df['b_i'].median():+.5f}")

print("\nKURAL 1 -- KIRPMA TABLOSU (en buyuk |.| K trafo atilinca)")
print(f"{'K':>5} {'n':>6} {'ACIK ort':>10} {'ACIK med':>10} {'b ort':>10} {'b med':>10}")
s = df.reindex(df["ACIK"].abs().sort_values(ascending=False).index)
for K in (0, 1, 5, 10, 25, 50, 100, 200):
    q = s.iloc[K:]
    print(
        f"{K:>5} {len(q):>6} {q['ACIK'].mean():+10.5f} {q['ACIK'].median():+10.5f}"
        f" {q['b_i'].mean():+10.5f} {q['b_i'].median():+10.5f}"
    )

df.to_csv(KOK / "reports" / "_c6_panel_acik.csv")
print("\nyazildi reports/_c6_panel_acik.csv")
