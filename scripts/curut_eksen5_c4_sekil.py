"""CURUTME 3 -- (a) hava verisi GERCEK mi tahmin mi, (b) buyumenin SEKLI,
(c) yaz aylik sicaklik farki, (d) CDD katsayisinin saglamligi."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path("C:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX")
sys.path.insert(0, str(KOK / "src"))
from gridup.turkish import join_key  # noqa: E402

h = pd.read_parquet(
    KOK / "data/external/hava_gunluk.parquet",
    columns=["ilce_key", "tarih", "sicaklik_ort", "sicaklik_max", "hava_tahmin"],
)
h["tarih"] = pd.to_datetime(h["tarih"])
h["ay"] = h["tarih"].dt.to_period("M")
son = h[h["tarih"] >= "2026-01-01"]
print("hava_tahmin bayragi, 2026 aylik ort (1 = tahmin):")
print(son.groupby("ay")["hava_tahmin"].mean().round(4).to_string())
print("\nmetadata:")
import json

md = KOK / "data/external/hava_gunluk.parquet.metadata.json"
if md.exists():
    j = json.loads(md.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                k: v
                for k, v in j.items()
                if k in ("cekilis", "fetched_at", "kaynak", "source", "url", "tarih")
            },
            ensure_ascii=False,
        )[:800]
    )

# ---- panel
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
L = p.groupby("tarih")["rc"].mean()
p["ilce_key"] = p["lokasyon"].str.split(">").str[-1].str.strip().map(join_key)
w = p.drop_duplicates("tanim")["ilce_key"].value_counts(normalize=True)
hv = h[h["ilce_key"].isin(w.index)].drop_duplicates(["ilce_key", "tarih"]).copy()
hv["w"] = hv["ilce_key"].map(w)
T = hv.groupby("tarih").apply(
    lambda d: (d["w"] * d["sicaklik_ort"]).sum() / d["w"].sum(), include_groups=False
)
X = pd.DataFrame({"T": T})
X["hdd"] = (18 - X["T"]).clip(lower=0)
X["cdd"] = (X["T"] - 22).clip(lower=0)

print("\n--- AYLIK panel-agirlikli ORT SICAKLIK, 2025 vs 2026")
print(f"{'ay':>4} {'T2025':>7} {'T2026':>7} {'dT':>7} {'CDD25':>7} {'CDD26':>7} {'dCDD':>7}")
for m in range(1, 9):
    try:
        a = X.loc[f"2025-{m:02d}"]
        b = X.loc[f"2026-{m:02d}"]
    except KeyError:
        continue
    print(
        f"{m:>4} {a['T'].mean():7.2f} {b['T'].mean():7.2f} {b['T'].mean() - a['T'].mean():+7.2f}"
        f" {a['cdd'].mean():7.3f} {b['cdd'].mean():7.3f} {b['cdd'].mean() - a['cdd'].mean():+7.3f}"
    )

# ---- buyumenin SEKLI: ay kuklalari + hava, TRENDSIZ; aylik rezidu
d = X.loc[L.index].copy()
d["L"] = L
d["moy"] = d.index.month
d["dow"] = d.index.dayofweek
D = pd.get_dummies(d["moy"], prefix="m", drop_first=True).astype(float)
W = pd.get_dummies(d["dow"], prefix="w", drop_first=True).astype(float)
A = pd.concat([pd.Series(1.0, index=d.index, name="c"), D, W, d[["hdd", "cdd"]]], axis=1)
M = A.to_numpy(float)
y = d["L"].to_numpy()
beta, *_ = np.linalg.lstsq(M, y, rcond=None)
d["res"] = y - M @ beta
print("\n--- HAVA+AY DUZELTILMIS AYLIK REZIDU (buyumenin sekli; Oca/Sub/Mar iki yil da var)")
mr = d.groupby(d.index.to_period("M"))["res"].mean()
for k, v in mr.items():
    print(f"   {k}  {v:+.4f}")

# ---- CDD katsayisi: yalniz YAZ aylarindan (Haz-Eyl 2025) yeniden
yaz = d.loc["2025-06-01":"2025-09-30"].copy()
Dy = pd.get_dummies(yaz["moy"], prefix="m", drop_first=True).astype(float)
Wy = pd.get_dummies(yaz["dow"], prefix="w", drop_first=True).astype(float)
Ay = pd.concat([pd.Series(1.0, index=yaz.index, name="c"), Dy, Wy, yaz[["cdd"]]], axis=1)
My = Ay.to_numpy(float)
yy = yaz["L"].to_numpy()
by, *_ = np.linalg.lstsq(My, yy, rcond=None)
resy = yy - My @ by
s2 = resy @ resy / (len(yy) - My.shape[1])
sey = np.sqrt(np.diag(s2 * np.linalg.pinv(My.T @ My)))
i = list(Ay.columns).index("cdd")
print(
    f"\nCDD katsayisi YALNIZ Haz-Eyl 2025'ten: {by[i]:+.5f} (SH {sey[i]:.5f})  [tum ornek +0.0496]"
)

# ---- AYLIK duzeyde regresyon (gunluk gurultu yok)
am = d.groupby(d.index.to_period("M")).agg(
    L=("L", "mean"), hdd=("hdd", "mean"), cdd=("cdd", "mean")
)
am["moy"] = [x.month for x in am.index]
am["t"] = [(x.year - 2025) + (x.month - 1) / 12 for x in am.index]
print("\nAYLIK panel (15 nokta) ham L:")
print(am[["L", "hdd", "cdd"]].round(4).to_string())
