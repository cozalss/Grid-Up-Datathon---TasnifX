"""p02 v2: sicak/soguk AYRI model, trafo-blokli erken durdurma (yaz25'e HIC dokunmadan)."""

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
SC = (
    "C:/Users/Cem/AppData/Local/Temp/claude/"
    "c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    "e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad"
)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


tr, te = ham()
meta = pd.concat([tr[["tanim", "guc", "lokasyon"]], te[["tanim", "guc", "lokasyon"]]])
meta = meta.drop_duplicates("tanim")
sp = meta.lokasyon.fillna(">").str.split(">", expand=True)
meta["ilce"] = sp[2].fillna("YOK")

bloklar = {}
for ad, (k0, k1) in KESIM.items():
    kes = pd.Timestamp(k0)
    h = tr[tr.tarih < kes]
    assert h.tarih.max() < kes
    if ad == "test":
        s = te[["id", "tanim", "guc", "tarih", "lokasyon"]].copy()
    else:
        s = tr[(tr.tarih >= kes) & (tr.tarih <= pd.Timestamp(k1))][
            ["tanim", "guc", "tarih", "lokasyon", "y"]
        ].copy()
    grp = grup_onceligi(h[["tanim", "y"]], meta[["tanim", "guc", "ilce"]])
    d = blok_kur(s.reset_index(drop=True), h, meta, k0, grp)
    bloklar[ad] = d
    log(ad, d.shape, "soguk", round(float(d.soguk.mean()), 4))

eg = pd.read_parquet(f"{DN}/egitim.parquet", columns=["tanim", "tarih"] + DIS)
tp = pd.read_parquet(f"{DN}/test.parquet", columns=["tanim", "tarih"] + DIS)
dis = pd.concat([eg, tp], ignore_index=True)
dis["tanim"] = dis.tanim.astype(str)
dis = dis.drop_duplicates(["tanim", "tarih"])
for ad in bloklar:
    n0 = len(bloklar[ad])
    bloklar[ad] = bloklar[ad].merge(dis, on=["tanim", "tarih"], how="left")
    assert len(bloklar[ad]) == n0
del eg, tp, dis
log("disgudumlu birlestirildi")

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
    o = d.h_ort.fillna(d.gr_ilce_gk).fillna(d.gr_gk).fillna(d.gr_guc)
    d["_ofs"] = o.fillna(float(tr.y.mean())).to_numpy()

TUM = [c for c in bloklar["guz25"].columns if c not in ATLA]
SOZ = [c for c in TUM if not c.startswith("h_")]  # soguk: gecmis ozniteligi YOK
log("sicak oznitelik", len(TUM), "soguk oznitelik", len(SOZ))

PAR = dict(
    objective="regression",
    metric="l2",
    learning_rate=0.04,
    num_leaves=255,
    min_data_in_leaf=100,
    feature_fraction=0.6,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    num_threads=8,
    verbosity=-1,
    max_bin=255,
)

trn = pd.concat([bloklar["guz25"], bloklar["kis26"]], ignore_index=True)
assert (trn.tarih >= pd.Timestamp("2025-08-01")).all(), "SIZINTI: yaz25 satiri egitimde"
rs = np.random.RandomState(7)
tans = trn.tanim.unique()
hold = set(rs.choice(tans, size=int(0.12 * len(tans)), replace=False))
trn["_hold"] = trn.tanim.isin(hold)
log("egitim", trn.shape, "ayrilan trafo", len(hold))

SON = {}
for et, oz, msk in (("sicak", TUM, 0), ("soguk", SOZ, 1)):
    a = trn[trn.soguk == msk]
    A, B = a[~a._hold], a[a._hold]
    dA = lgb.Dataset(A[oz], label=A.y - A._ofs, categorical_feature=KAT)
    dB = lgb.Dataset(B[oz], label=B.y - B._ofs, categorical_feature=KAT)
    m = lgb.train(
        PAR,
        dA,
        num_boost_round=4000,
        valid_sets=[dB],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    ni = max(50, int(m.best_iteration * 1.1))
    log(
        et,
        "n",
        len(a),
        "best_iter",
        m.best_iteration,
        "-> nihai",
        ni,
        "ayrilan RMSE",
        round(float(m.best_score["valid_0"]["l2"]) ** 0.5, 5),
    )
    mf = lgb.train(
        PAR, lgb.Dataset(a[oz], label=a.y - a._ofs, categorical_feature=KAT), num_boost_round=ni
    )
    for ad in ("yaz25", "test"):
        d = bloklar[ad]
        s = d.soguk == msk
        if s.sum() == 0:
            continue
        d.loc[s, "p2"] = np.clip(
            mf.predict(d.loc[s, oz], num_iteration=ni) + d.loc[s, "_ofs"].to_numpy(), 0.0, None
        )

y = bloklar["yaz25"].y.to_numpy()
p = bloklar["yaz25"].p2.to_numpy()
sg = bloklar["yaz25"].soguk.to_numpy()
r = p - y
SON = dict(
    duz=float(np.sqrt((r * r).mean())),
    sicak=float(np.sqrt((r[sg == 0] ** 2).mean())),
    soguk=float(np.sqrt((r[sg == 1] ** 2).mean())),
)
SON["test_agirlikli"] = float(np.sqrt(0.2216 * SON["soguk"] ** 2 + 0.7784 * SON["sicak"] ** 2))
log("v2 yaz25", SON)
bloklar["yaz25"][["tanim", "tarih", "y", "soguk", "p2"]].to_parquet(f"{SC}/p02_yaz25_v2.parquet")
bloklar["test"][["id", "tanim", "tarih", "soguk", "p2"]].to_parquet(f"{SC}/p02_test_v2.parquet")
json.dump(SON, open(f"{K}/experiments/model29/p02_temiz_taban_v2.json", "w"), indent=1)
