"""CURUTME 2 -- Q1 2026 sicramasi TREND mi, HAVA mi?

Ay kuklalari + dogrusal trend: trend YALNIZCA tekrarli Oca/Sub/Mar'dan
kimliklenir (= Q1 YoY). Uzerine HDD/CDD eklenince trendin ne kadari
havayla aciklaniyor?  Sonra 2026 Nis-Tem havasi GERCEKTEN elimizde
(hava_gunluk 2026-08-28'e kadar dolu) -> yaz YoY tahmini.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path("C:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX")
sys.path.insert(0, str(KOK / "src"))
from gridup.turkish import join_key  # noqa: E402

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
PANEL = sorted(set(ays[ays == 15].index) & set(nza[nza == 15].index))
print(f"PANEL {len(PANEL)} trafo")

p = tr[tr["tanim"].isin(set(PANEL))].copy()
p["rc"] = p["r"] - p.groupby("tanim", observed=True)["r"].transform("mean")
L = p.groupby("tarih")["rc"].mean()  # gunluk seviye endeksi (trafo etkisi cikti)
print("gunluk endeks", len(L), "gun")

# ---- ilce agirlikli hava
p["ilce_key"] = p["lokasyon"].str.split(">").str[-1].str.strip().map(join_key)
w = p.drop_duplicates("tanim")["ilce_key"].value_counts(normalize=True)
hava = pd.read_parquet(
    KOK / "data/external/hava_gunluk.parquet",
    columns=["ilce_key", "tarih", "sicaklik_ort", "sicaklik_max"],
)
hava = hava.drop_duplicates(["ilce_key", "tarih"])
hava["tarih"] = pd.to_datetime(hava["tarih"])
hava = hava[hava["ilce_key"].isin(w.index)].copy()
hava["w"] = hava["ilce_key"].map(w)
eksik = set(w.index) - set(hava["ilce_key"].unique())
print(
    "hava'da olmayan ilce:",
    eksik,
    " kapsanan agirlik %.4f" % w[list(set(w.index) & set(hava["ilce_key"]))].sum(),
)
hava["ws"] = hava["w"] * hava["sicaklik_ort"]
T = hava.groupby("tarih").apply(
    lambda d: (d["w"] * d["sicaklik_ort"]).sum() / d["w"].sum(), include_groups=False
)
T.name = "T"
X = pd.DataFrame({"T": T})
X["hdd"] = (18.0 - X["T"]).clip(lower=0.0)
X["cdd"] = (X["T"] - 22.0).clip(lower=0.0)
X["cdd2"] = X["cdd"] ** 2
X["hdd2"] = X["hdd"] ** 2

print("\n--- Q1 HAVA KARSILASTIRMASI (panel agirlikli ort sicaklik / HDD)")
for m in ["01", "02", "03"]:
    a = X.loc[f"2025-{m}"]
    b = X.loc[f"2026-{m}"]
    print(
        f"  ay {m}: T25 {a['T'].mean():6.2f}  T26 {b['T'].mean():6.2f}  dT {b['T'].mean() - a['T'].mean():+6.2f}"
        f" | HDD25 {a['hdd'].mean():6.2f} HDD26 {b['hdd'].mean():6.2f} dHDD {b['hdd'].mean() - a['hdd'].mean():+6.2f}"
    )

# ---- regresyon: ay kuklalari + trend (+hava)
d = X.loc[L.index].copy()
d["L"] = L
d["moy"] = d.index.month
d["t"] = (d.index - pd.Timestamp("2025-01-01")).days / 365.25
d["dow"] = d.index.dayofweek


def fit(cols, ad):
    D = pd.get_dummies(d["moy"], prefix="m", drop_first=True).astype(float)
    W = pd.get_dummies(d["dow"], prefix="w", drop_first=True).astype(float)
    A = pd.concat([pd.Series(1.0, index=d.index, name="c"), D, W, d[cols]], axis=1)
    y = d["L"].to_numpy()
    M = A.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(M, y, rcond=None)
    res = y - M @ beta
    s2 = res @ res / (len(y) - M.shape[1])
    cov = s2 * np.linalg.pinv(M.T @ M)
    se = np.sqrt(np.diag(cov))
    out = dict(zip(A.columns, beta))
    ses = dict(zip(A.columns, se))
    print(f"\n[{ad}]  R2={1 - res.var() / y.var():.4f}  rezidu sd={res.std():.4f}")
    for c in cols:
        print(f"    {c:>6} {out[c]:+.5f}  (SH {ses[c]:.5f}, t {out[c] / ses[c]:+.1f})")
    return out, ses, res


b1, s1, r1 = fit(["t"], "A: ay kuklalari + trend")
b2, s2, r2 = fit(["t", "hdd", "cdd"], "B: + HDD/CDD")
b3, s3, r3 = fit(["t", "hdd", "cdd", "hdd2", "cdd2"], "C: + kareler")

print("\n=== TREND (yillik log birimi) ===")
for ad, b, s in (("A ham", b1, s1), ("B hava-duzeltilmis", b2, s2), ("C hava2", b3, s3)):
    print(f"  {ad:>20}: {b['t']:+.5f}  +/- {s['t']:.5f}")

# ---- yaz YoY tahmini: trend + hava farki
print("\n=== 2026 Nis-Tem vs 2025 Nis-Tem ===")
y25 = X.loc["2025-04-01":"2025-07-31"]
y26 = X.loc["2026-04-01":"2026-07-31"]
print(
    f"  T:   2025 {y25['T'].mean():.2f}   2026 {y26['T'].mean():.2f}   dT {y26['T'].mean() - y25['T'].mean():+.2f}"
)
print(
    f"  HDD: 2025 {y25['hdd'].mean():.3f} 2026 {y26['hdd'].mean():.3f} d {y26['hdd'].mean() - y25['hdd'].mean():+.3f}"
)
print(
    f"  CDD: 2025 {y25['cdd'].mean():.3f} 2026 {y26['cdd'].mean():.3f} d {y26['cdd'].mean() - y25['cdd'].mean():+.3f}"
)
for ad, b in (("B", b2), ("C", b3)):
    hv = 0.0
    for c in ("hdd", "cdd", "hdd2", "cdd2"):
        if c in b:
            hv += b[c] * (y26[c].mean() - y25[c].mean())
    print(f"  [{ad}] trend {b['t']:+.5f} + hava {hv:+.5f} = YAZ YoY {b['t'] + hv:+.5f}")
# ayni hesap Q1 icin (kontrol)
q25 = X.loc["2025-01-01":"2025-03-31"]
q26 = X.loc["2026-01-01":"2026-03-31"]
for ad, b in (("B", b2), ("C", b3)):
    hv = sum(
        b[c] * (q26[c].mean() - q25[c].mean()) for c in ("hdd", "cdd", "hdd2", "cdd2") if c in b
    )
    print(
        f"  [{ad}] Q1 KONTROL: trend {b['t']:+.5f} + hava {hv:+.5f} = {b['t'] + hv:+.5f}  (olculen ham ~+0.098)"
    )
