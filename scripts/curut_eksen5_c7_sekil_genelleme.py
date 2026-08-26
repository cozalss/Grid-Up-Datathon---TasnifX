"""CURUTME -- (A) buyume SUREKLI TREND mi yoksa 2025 ICINDE tek seferlik BASAMAK mi?
(B) surdurulebilir CDD esnekligi, (C) panel disina genelleme (tum hedef satirlar)."""

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
PANEL = set(ays[ays == 15].index) & set(nza[nza == 15].index)
p = tr[tr["tanim"].isin(PANEL)].copy()
p["rc"] = p["r"] - p.groupby("tanim", observed=True)["r"].transform("mean")

aylik = p.groupby(["tanim", "ay"], observed=True)["rc"].mean().unstack()
AYLAR = list(aylik.columns)

# --- (A1) MEVSIMSEL GENLIGE gore ayir: DUZ YUKLU trafolarda seviye yolu
# mevsim sinyali = panel ortalamasinin aylik profili
prof = aylik.mean(axis=0)
merk = prof - prof.mean()
# her trafonun mevsim yuku: aylik rc'sinin panel profiline regresyonu
Y = aylik.sub(aylik.mean(axis=1), axis=0)
xg = merk - merk.mean()
beta = (Y * xg).sum(axis=1) / (xg * xg).sum()
print("mevsim yuku beta: p10 %.2f p50 %.2f p90 %.2f" % tuple(beta.quantile([0.1, 0.5, 0.9])))

DUZ = beta[beta.abs() <= beta.abs().quantile(0.30)].index  # en duz %30
print(
    f"\n(A) DUZ (mevsimsiz) {len(DUZ)} trafonun AYLIK SEVIYE YOLU  [mevsim ~yok => saf surukleme]"
)
duz = aylik.loc[DUZ]
duzm = duz.mean(axis=0) - duz.mean(axis=0).mean()
for a in AYLAR:
    n = int(duz[a].notna().sum())
    print(
        f"   {a}  ort {duz[a].mean():+.4f}  medyan {duz[a].median():+.4f}  (merkezli {duzm[a]:+.4f}, n={n})"
    )
# duz altkumede Q1 YoY ve 2025-ici egim
q25 = [pd.Period("2025-01"), pd.Period("2025-02"), pd.Period("2025-03")]
q26 = [pd.Period("2026-01"), pd.Period("2026-02"), pd.Period("2026-03")]
byq = duz[q26].mean(axis=1) - duz[q25].mean(axis=1)
print(f"   DUZ altkume Q1 YoY: ort {byq.mean():+.5f} medyan {byq.median():+.5f}")
# 2025 ICI dogrusal egim (Nis-Ara 2025, 9 ay) -- yillik birime cevir
ic = [pd.Period(f"2025-{m:02d}") for m in range(4, 13)]
tt = np.arange(len(ic)) / 12.0
Z = duz[ic]
zc = Z.sub(Z.mean(axis=1), axis=0)
tc = tt - tt.mean()
egim = (zc * tc).sum(axis=1) / (tc * tc).sum()
print(
    f"   DUZ altkume 2025 Nis-Ara ICI egim (yillik): ort {egim.mean():+.5f} medyan {egim.median():+.5f}"
)
# tum panel icin ayni ic-egim
Zp = aylik[ic]
zcp = Zp.sub(Zp.mean(axis=1), axis=0)
egp = (zcp * tc).sum(axis=1) / (tc * tc).sum()
print(
    f"   TUM panel 2025 Nis-Ara ICI egim (yillik): ort {egp.mean():+.5f} medyan {egp.median():+.5f}"
)

# --- (A2) BASAMAK TESTI: buyume 2025 icinde mi gerceklesti?
# her trafo icin mevsim-duzeltilmis seviye: rc - beta_i * merk
sa = aylik.sub(pd.DataFrame(np.outer(beta, merk), index=aylik.index, columns=AYLAR))
buy = aylik[q26].mean(axis=1) - aylik[q25].mean(axis=1)
ust = buy.sort_values(ascending=False).index[:100]
print("\n(A2) EN BUYUK 100 'BUYUYEN' TRAFONUN MEVSIM-DUZELTILMIS SEVIYE YOLU")
for a in AYLAR:
    print(f"   {a}  {sa.loc[ust, a].mean():+.4f}")
print("   -> 2025 Nis-Tem seviyesi 2025 Q1'e mi 2026 Q1'e mi yakin?")
z1 = sa.loc[ust, q25].mean(axis=1).mean()
z2 = sa.loc[ust, [pd.Period(f"2025-{m:02d}") for m in (4, 5, 6, 7)]].mean(axis=1).mean()
z3 = sa.loc[ust, q26].mean(axis=1).mean()
print(
    f"   2025Q1 {z1:+.4f} | 2025 Nis-Tem {z2:+.4f} | 2026Q1 {z3:+.4f}"
    f" | yaz25 basamagin %{100 * (z2 - z1) / max(z3 - z1, 1e-9):.0f}'inde"
)

# --- (B) SURDURULEBILIR CDD esnekligi: yumusatilmis CDD ile
p["ilce_key"] = p["lokasyon"].str.split(">").str[-1].str.strip().map(join_key)
w = p.drop_duplicates("tanim")["ilce_key"].value_counts(normalize=True)
h = pd.read_parquet(
    KOK / "data/external/hava_gunluk.parquet", columns=["ilce_key", "tarih", "sicaklik_ort"]
).drop_duplicates(["ilce_key", "tarih"])
h["tarih"] = pd.to_datetime(h["tarih"])
h = h[h["ilce_key"].isin(w.index)].copy()
h["w"] = h["ilce_key"].map(w)
T = (
    h.groupby("tarih")
    .apply(lambda d: (d["w"] * d["sicaklik_ort"]).sum() / d["w"].sum(), include_groups=False)
    .sort_index()
)
X = pd.DataFrame({"T": T})
X["hdd"] = (18 - X["T"]).clip(lower=0)
X["cdd"] = (X["T"] - 22).clip(lower=0)
for k in (3, 7, 14):
    X[f"hdd{k}"] = X["hdd"].rolling(k, min_periods=1).mean()
    X[f"cdd{k}"] = X["cdd"].rolling(k, min_periods=1).mean()
L = p.groupby("tarih")["rc"].mean()
d = X.loc[L.index].copy()
d["L"] = L
d["moy"] = d.index.month
d["dow"] = d.index.dayofweek
d["t"] = (d.index - pd.Timestamp("2025-01-01")).days / 365.25


def fit(cols):
    D = pd.get_dummies(d["moy"], prefix="m", drop_first=True).astype(float)
    W = pd.get_dummies(d["dow"], prefix="w", drop_first=True).astype(float)
    A = pd.concat([pd.Series(1.0, index=d.index, name="c"), D, W, d[cols]], axis=1)
    M = A.to_numpy(float)
    y = d["L"].to_numpy()
    b, *_ = np.linalg.lstsq(M, y, rcond=None)
    res = y - M @ b
    return dict(zip(A.columns, b)), 1 - res.var() / y.var()


print("\n(B) HAVA ESNEKLIGI -- farkli yumusatma")
y25 = X.loc["2025-04-01":"2025-07-31"]
y26 = X.loc["2026-04-01":"2026-07-31"]
for suf in ("", "3", "7", "14"):
    cols = ["t", f"hdd{suf}", f"cdd{suf}"]
    b, r2 = fit(cols)
    hv = b[f"hdd{suf}"] * (y26[f"hdd{suf}"].mean() - y25[f"hdd{suf}"].mean()) + b[f"cdd{suf}"] * (
        y26[f"cdd{suf}"].mean() - y25[f"cdd{suf}"].mean()
    )
    print(
        f"   yumusatma {suf or '1':>2}g: R2 {r2:.4f} trend {b['t']:+.5f} "
        f"cdd {b[f'cdd{suf}']:+.5f}  -> YAZ hava etkisi {hv:+.5f}  YAZ YoY {b['t'] + hv:+.5f}"
    )
# aylik profilden esneklik (mevsim genligi)
am = d.groupby(d.index.to_period("M")).agg(
    L=("L", "mean"), cdd=("cdd", "mean"), hdd=("hdd", "mean")
)
yazay = [pd.Period(f"2025-{m:02d}") for m in (5, 6, 7, 8, 9)]
sub = am.loc[yazay]
sl = np.polyfit(sub["cdd"], sub["L"], 1)[0]
print(
    f"   AYLIK (May-Eyl 2025, 5 nokta) CDD egimi {sl:+.5f}"
    f"  -> YAZ hava etkisi {sl * (y26['cdd'].mean() - y25['cdd'].mean()):+.5f}"
)
