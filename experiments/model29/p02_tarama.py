"""p02 tarama: benim oznitelik kumemin ORACLE ust siniri.
Tur sayisini yaz25'e BAKARAK seciyorum -- bu KOPYA CEKMEDIR, dolayisiyla
elde edilen sayi benim yaklasimimin ULASILABILIR EN IYISI (iyimser sinir).
Amac: aradaki fark AYAR mi yoksa OZNITELIK mi, onu ayirmak.
"""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p02_oznitelik import DIS, KESIM, blok_kur, grup_onceligi, ham  # noqa

K = "c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = f"{K}/data/interim/deney"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


tr, te = ham()
meta = pd.concat([tr[["tanim", "guc", "lokasyon"]], te[["tanim", "guc", "lokasyon"]]])
meta = meta.drop_duplicates("tanim").reset_index(drop=True)
sp = meta.lokasyon.fillna(">").str.split(">", expand=True)
meta["ilce"] = sp[2].fillna("YOK")

bloklar = {}
for ad in ("yaz25", "guz25", "kis26"):
    k0, k1 = KESIM[ad]
    kes = pd.Timestamp(k0)
    h = tr[tr.tarih < kes]
    s = tr[(tr.tarih >= kes) & (tr.tarih <= pd.Timestamp(k1))][
        ["tanim", "guc", "tarih", "lokasyon", "y"]
    ].copy()
    grp = grup_onceligi(h[["tanim", "y"]], meta[["tanim", "guc", "ilce"]])
    d = blok_kur(s.reset_index(drop=True), h, meta, k0, grp)
    u = pd.DataFrame({"tanim": d.tanim.unique()})
    ty = h.groupby("tanim", observed=True).y.mean()
    for n in (3, 4, 5, 6):
        mm = ty.groupby(ty.index.str[:n]).agg(["mean", "size"])
        p = u.tanim.str[:n]
        u[f"k_on{n}"] = p.map(mm["mean"]).to_numpy()
        u[f"k_on{n}_n"] = p.map(mm["size"]).to_numpy()
    bloklar[ad] = d.merge(u, on="tanim", how="left")
    log(ad, bloklar[ad].shape)

eg = pd.read_parquet(f"{DN}/egitim.parquet", columns=["tanim", "tarih"] + DIS)
eg["tanim"] = eg.tanim.astype(str)
eg = eg.drop_duplicates(["tanim", "tarih"])
for ad in bloklar:
    n0 = len(bloklar[ad])
    bloklar[ad] = bloklar[ad].merge(eg, on=["tanim", "tarih"], how="left")
    assert len(bloklar[ad]) == n0
del eg

KAT = ["il", "bolge", "ilce", "gk", "tk_hg", "tk_ay"]
ATLA = {"tanim", "tarih", "lokasyon", "y", "id", "tuketim"}
for ad in bloklar:
    for c in KAT:
        bloklar[ad][c] = bloklar[ad][c].astype(str)
kats = {c: sorted(set().union(*[set(bloklar[a][c]) for a in bloklar])) for c in KAT}
for ad in bloklar:
    for c in KAT:
        bloklar[ad][c] = pd.Categorical(bloklar[ad][c], categories=kats[c])
    d = bloklar[ad]
    sofs = (
        d.k_on6.fillna(d.k_on5)
        .fillna(d.k_on4)
        .fillna(d.gr_ilce_gk)
        .fillna(d.gr_gk)
        .fillna(d.gr_guc)
    )
    d["_ofs"] = d.h_ort.fillna(sofs).fillna(float(tr.y.mean())).to_numpy()

TUM = [c for c in bloklar["guz25"].columns if c not in ATLA]
SOZ = [c for c in TUM if not c.startswith("h_")]
trn = pd.concat([bloklar["guz25"], bloklar["kis26"]], ignore_index=True)
assert (trn.tarih >= pd.Timestamp("2025-08-01")).all()
Y = bloklar["yaz25"]

R = {}
# ---- soguk icin BASIT referanslar ----
sy = Y[Y.soguk == 1]
R["ref_soguk_sabit_ortalama"] = float(np.sqrt(((sy._ofs - sy.y) ** 2).mean()))
c = sy.y.mean()
R["ref_soguk_en_iyi_tek_sabit"] = float(np.sqrt(((c - sy.y) ** 2).mean()))
orc = sy.groupby("tanim").y.transform("mean")
R["ref_soguk_ORACLE_trafo_ort"] = float(np.sqrt(((orc - sy.y) ** 2).mean()))
log("soguk referanslar", {k: round(v, 4) for k, v in R.items()})

PAR = dict(
    objective="regression",
    metric="l2",
    learning_rate=0.04,
    num_leaves=127,
    min_data_in_leaf=200,
    feature_fraction=0.6,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    num_threads=8,
    verbosity=-1,
    max_bin=255,
)
TUR = [10, 20, 40, 80, 160, 320, 640, 1280]
for et, oz, msk in (("sicak", TUM, 0), ("soguk", SOZ, 1)):
    a = trn[trn.soguk == msk]
    m = lgb.train(
        PAR,
        lgb.Dataset(a[oz], label=a.y - a._ofs, categorical_feature=KAT),
        num_boost_round=max(TUR),
    )
    v = Y[Y.soguk == msk]
    for t in TUR:
        p = np.clip(m.predict(v[oz], num_iteration=t) + v._ofs.to_numpy(), 0, None)
        R[f"{et}_tur{t}"] = float(np.sqrt(((p - v.y.to_numpy()) ** 2).mean()))
    log(et, {t: round(R[f"{et}_tur{t}"], 5) for t in TUR})

sc = min(R[f"sicak_tur{t}"] for t in TUR)
so = min(R[f"soguk_tur{t}"] for t in TUR)
n1 = (Y.soguk == 0).sum()
n2 = (Y.soguk == 1).sum()
R["ORACLE_duz"] = float(np.sqrt((n1 * sc**2 + n2 * so**2) / (n1 + n2)))
R["ORACLE_test_agirlikli"] = float(np.sqrt(0.7784 * sc**2 + 0.2216 * so**2))
log("ORACLE", R["ORACLE_duz"], R["ORACLE_test_agirlikli"])
json.dump(R, open(f"{K}/experiments/model29/p02_tarama.json", "w"), indent=1)
