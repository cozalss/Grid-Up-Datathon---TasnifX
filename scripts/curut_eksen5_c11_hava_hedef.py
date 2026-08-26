"""CURUTME -- (1) TRAFO BAZINDA hava duyarliligi -> HEDEF-AGIRLIKLI hava duzeltmesi.
(2) S25 atfinin GERCEK capraz dogrulamasi (donor havuzu ikiye bolunerek)."""

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
tr["ilce_key"] = tr["lokasyon"].str.split(">").str[-1].str.strip().map(join_key)
te = pd.read_csv(
    KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
)
h = pd.read_parquet(
    KOK / "data/external/hava_gunluk.parquet", columns=["ilce_key", "tarih", "sicaklik_ort"]
).drop_duplicates(["ilce_key", "tarih"])
h["tarih"] = pd.to_datetime(h["tarih"])
h["cdd"] = (h["sicaklik_ort"] - 22).clip(lower=0)
h["hdd"] = (18 - h["sicaklik_ort"]).clip(lower=0)

x = tr.merge(h[["ilce_key", "tarih", "cdd", "hdd"]], on=["ilce_key", "tarih"], how="left")
assert x["cdd"].notna().all()
x["ay"] = x["tarih"].dt.to_period("M")
k = ["tanim", "ay"]
for c in ("r", "cdd", "hdd"):
    x[c + "_d"] = x[c] - x.groupby(k, observed=True)[c].transform("mean")
gg = x.groupby("tanim", observed=True)
S = pd.DataFrame(
    {
        "scc": gg.apply(lambda d: float((d["cdd_d"] ** 2).sum()), include_groups=False),
        "shh": gg.apply(lambda d: float((d["hdd_d"] ** 2).sum()), include_groups=False),
        "sch": gg.apply(lambda d: float((d["cdd_d"] * d["hdd_d"]).sum()), include_groups=False),
        "scr": gg.apply(lambda d: float((d["cdd_d"] * d["r_d"]).sum()), include_groups=False),
        "shr": gg.apply(lambda d: float((d["hdd_d"] * d["r_d"]).sum()), include_groups=False),
        "n": gg.size(),
    }
)
det = S["scc"] * S["shh"] - S["sch"] ** 2
S["c"] = np.where(det > 1e-6, (S["shh"] * S["scr"] - S["sch"] * S["shr"]) / det, np.nan)
S["hh"] = np.where(det > 1e-6, (S["scc"] * S["shr"] - S["sch"] * S["scr"]) / det, np.nan)
S = S[S["n"] >= 120]
print(f"hava duyarliligi kestirilen trafo: {len(S):,}")
print(
    "  c (CDD22) : p10 %.4f p50 %.4f p90 %.4f  ort %.4f"
    % (*S["c"].quantile([0.1, 0.5, 0.9]), S["c"].mean())
)
print(
    "  h (HDD18) : p10 %.4f p50 %.4f p90 %.4f  ort %.4f"
    % (*S["hh"].quantile([0.1, 0.5, 0.9]), S["hh"].mean())
)


# ---- her ILCE icin gercek yaz hava farki
def wd(ilce, a1, b1, a2, b2, col):
    q = h[h["ilce_key"] == ilce]
    return (
        q.loc[(q["tarih"] >= a2) & (q["tarih"] <= b2), col].mean()
        - q.loc[(q["tarih"] >= a1) & (q["tarih"] <= b1), col].mean()
    )


ilceler = sorted(tr["ilce_key"].unique())
dcdd = {i: wd(i, "2025-04-01", "2025-07-31", "2026-04-01", "2026-07-31", "cdd") for i in ilceler}
dhdd = {i: wd(i, "2025-04-01", "2025-07-31", "2026-04-01", "2026-07-31", "hdd") for i in ilceler}
q_dcdd = {i: wd(i, "2025-01-01", "2025-03-31", "2026-01-01", "2026-03-31", "cdd") for i in ilceler}
q_dhdd = {i: wd(i, "2025-01-01", "2025-03-31", "2026-01-01", "2026-03-31", "hdd") for i in ilceler}
print(
    "\nilce dCDD (yaz) p10/p50/p90: %.3f %.3f %.3f"
    % tuple(np.quantile(list(dcdd.values()), [0.1, 0.5, 0.9]))
)
print(
    "ilce dHDD (Q1)  p10/p50/p90: %.3f %.3f %.3f"
    % tuple(np.quantile(list(q_dhdd.values()), [0.1, 0.5, 0.9]))
)

# ---- hedef kume ve agirliklar
SICAK = set(tr["tanim"].unique())
q126 = tr[(tr["tarih"] >= "2026-01-01") & (tr["tarih"] <= "2026-03-31")]
HEDEF = SICAK & set(q126.loc[q126["tuketim"] > 0, "tanim"].unique())
th = te[te["tanim"].isin(HEDEF)]
W = th.groupby("tanim").size()
meta = te.drop_duplicates("tanim").set_index("tanim")
il = meta["lokasyon"].str.split(">").str[-1].str.strip().map(join_key)
guc = meta["guc"]
kov = pd.qcut(guc, 6, labels=False, duplicates="drop")

B = pd.DataFrame({"w": W})
B["ilce"] = il.reindex(B.index)
B["kov"] = kov.reindex(B.index)
B["c"] = S["c"].reindex(B.index)
B["hh"] = S["hh"].reindex(B.index)
# atfet: ilce x kova ortalamasi
don = S.join(pd.DataFrame({"ilce": il, "kov": kov}), how="inner").dropna(subset=["ilce"])
stc = don.groupby(["ilce", "kov"])["c"].agg(["mean", "size"])
sth = don.groupby(["ilce", "kov"])["hh"].agg(["mean", "size"])
ic_, ih_ = don.groupby("ilce")["c"].mean(), don.groupby("ilce")["hh"].mean()
key = list(zip(B["ilce"], B["kov"]))
B["c_atf"] = (
    B["c"]
    .fillna(
        pd.Series(
            [stc["mean"].get(k, np.nan) if stc["size"].get(k, 0) >= 10 else np.nan for k in key],
            index=B.index,
        )
    )
    .fillna(B["ilce"].map(ic_))
    .fillna(don["c"].mean())
)
B["h_atf"] = (
    B["hh"]
    .fillna(
        pd.Series(
            [sth["mean"].get(k, np.nan) if sth["size"].get(k, 0) >= 10 else np.nan for k in key],
            index=B.index,
        )
    )
    .fillna(B["ilce"].map(ih_))
    .fillna(don["hh"].mean())
)
B["dcdd"] = B["ilce"].map(dcdd)
B["dhdd"] = B["ilce"].map(dhdd)
B["qdcdd"] = B["ilce"].map(q_dcdd)
B["qdhdd"] = B["ilce"].map(q_dhdd)
B["hava_yaz"] = B["c_atf"] * B["dcdd"] + B["h_atf"] * B["dhdd"]
B["hava_q1"] = B["c_atf"] * B["qdcdd"] + B["h_atf"] * B["qdhdd"]
B["duzeltme"] = B["hava_yaz"] - B["hava_q1"]


def ag(v, w):
    return float((v * w).sum() / w.sum())


print("\nHEDEF-AGIRLIKLI hava duzeltmesi (GUNLUK esneklikle):")
print(
    f"   yaz {ag(B['hava_yaz'], B['w']):+.5f}   Q1 {ag(B['hava_q1'], B['w']):+.5f}"
    f"   net {ag(B['duzeltme'], B['w']):+.5f}"
)
print("   (panel karsiligi -0.0518 / +0.0034 / -0.0552)")
print(f"   hedef-agirlikli ort c {ag(B['c_atf'], B['w']):.5f} vs panel 0.0496")
SCALE = 0.0794 / 0.0496  # aylik/gunluk esneklik orani (c7)
print(f"   SURDURULEBILIR olcek x{SCALE:.2f} -> net {ag(B['duzeltme'], B['w']) * SCALE:+.5f}")

# ---- S25 atfinin CAPRAZ DOGRULAMASI
o1 = (
    tr[(tr["tarih"] >= "2025-01-01") & (tr["tarih"] <= "2025-03-31")]
    .groupby("tanim")["r"]
    .agg(["mean", "size"])
)
o3 = (
    tr[(tr["tarih"] >= "2025-04-01") & (tr["tarih"] <= "2025-07-31")]
    .groupby("tanim")["r"]
    .agg(["mean", "size"])
)
k25 = (o3["mean"] - o1["mean"]).dropna()
k25 = k25[(o1["size"].reindex(k25.index) >= 20) & (o3["size"].reindex(k25.index) >= 20)]
K = pd.DataFrame({"k25": k25}).join(pd.DataFrame({"ilce": il, "kov": kov}), how="inner").dropna()
rng = np.random.default_rng(7)
hata = []
for rep in range(20):
    m = rng.random(len(K)) < 0.5
    d1, d2 = K[m], K[~m]
    s = d1.groupby(["ilce", "kov"])["k25"].agg(["mean", "size"])
    si = d1.groupby("ilce")["k25"].mean()
    kk = list(zip(d2["ilce"], d2["kov"]))
    pr = pd.Series(
        [s["mean"].get(z, np.nan) if s["size"].get(z, 0) >= 15 else np.nan for z in kk],
        index=d2.index,
    )
    pr = pr.fillna(d2["ilce"].map(si)).fillna(d1["k25"].mean())
    hata.append(float((d2["k25"] - pr).mean()))
print(
    f"\nS25 ATFI capraz dogrulama (20 yarim-ornek): yanlilik ort {np.mean(hata):+.5f}"
    f"  SH {np.std(hata) / np.sqrt(20):.5f}  (0'a yakin olmali)"
)
B.to_csv(KOK / "reports" / "_c11_hava.csv")
