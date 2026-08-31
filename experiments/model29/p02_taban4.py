"""p02 v4: TAHMIN PENCERESI varlik oznitelikleri eklendi.
test.csv hangi (trafo,gun) satirlarinin isteneceğini SOYLER -- bu bilgi yasal,
hedef sizintisi DEGIL. Bir trafonun pencerede kac gun gorundugu, ilk/son ofseti,
doluluk orani soguk trafo icin guclu bir isaret.
Tur sayilari yaz25'e BAKMADAN secildi: sicak icin v1 semasi (kis26'da egit,
guz25'te erken durdur), soguk icin ayrilan trafo semasi."""
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
SC = ("C:/Users/Cem/AppData/Local/Temp/claude/"
      "c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
      "e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad")
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)


tr, te = ham()
meta = pd.concat([tr[["tanim", "guc", "lokasyon"]], te[["tanim", "guc", "lokasyon"]]])
meta = meta.drop_duplicates("tanim").reset_index(drop=True)
sp = meta.lokasyon.fillna(">").str.split(">", expand=True)
meta["ilce"] = sp[2].fillna("YOK")

bloklar = {}
for ad, (k0, k1) in KESIM.items():
    kes, son = pd.Timestamp(k0), pd.Timestamp(k1)
    h = tr[tr.tarih < kes]
    assert h.tarih.max() < kes
    if ad == "test":
        s = te[["id", "tanim", "guc", "tarih", "lokasyon"]].copy()
    else:
        s = tr[(tr.tarih >= kes) & (tr.tarih <= son)][
            ["tanim", "guc", "tarih", "lokasyon", "y"]].copy()
    s = s.reset_index(drop=True)
    grp = grup_onceligi(h[["tanim", "y"]], meta[["tanim", "guc", "ilce"]])
    d = blok_kur(s, h, meta, k0, grp)
    # --- TAHMIN PENCERESI varlik oznitelikleri (HEDEF KULLANILMAZ) ---
    o = (s.tarih - kes).dt.days
    w = pd.DataFrame({"tanim": s.tanim, "o": o}).groupby("tanim").o.agg(["size", "min", "max"])
    w.columns = ["w_gun", "w_ilk", "w_son"]
    w["w_yayilma"] = w.w_son - w.w_ilk + 1
    w["w_doluluk"] = w.w_gun / w.w_yayilma
    w["w_pay"] = w.w_gun / ((son - kes).days + 1)
    d = d.merge(w, left_on="tanim", right_index=True, how="left")
    d["w_bas_yeni"] = (d.w_ilk > 3).astype(int)
    d["w_ofs_ilkten"] = (d.tarih - kes).dt.days - d.w_ilk
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
    d["_ofs"] = d.h_ort.fillna(d.gr_guc).fillna(d.gr_gk).fillna(float(tr.y.mean())).to_numpy()

TUM = [c for c in bloklar["guz25"].columns if c not in ATLA]
SOZ = [c for c in TUM if not c.startswith("h_")]
trn = pd.concat([bloklar["guz25"], bloklar["kis26"]], ignore_index=True)
assert (trn.tarih >= pd.Timestamp("2025-08-01")).all(), "SIZINTI: yaz25 egitimde"
log("sicak oz", len(TUM), "soguk oz", len(SOZ))

PAR = dict(objective="regression", metric="l2", learning_rate=0.04,
           num_leaves=127, min_data_in_leaf=200, feature_fraction=0.6,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
           num_threads=8, verbosity=-1, max_bin=255)
rs = np.random.RandomState(7)
tans = trn.tanim.unique()
hold = set(rs.choice(tans, size=int(0.15 * len(tans)), replace=False))
trn["_hold"] = trn.tanim.isin(hold)

R = {}
for et, oz, msk in (("sicak", TUM, 0), ("soguk", SOZ, 1)):
    a = trn[trn.soguk == msk]
    if et == "sicak":  # kis26'da egit, guz25'te dogrula (blok-disi)
        A = a[a.tarih >= "2025-12-01"]
        B = a[a.tarih < "2025-12-01"]
    else:              # gorulmemis trafolar
        A, B = a[~a._hold], a[a._hold]
    m = lgb.train(PAR, lgb.Dataset(A[oz], label=A.y - A._ofs, categorical_feature=KAT),
                  num_boost_round=4000,
                  valid_sets=[lgb.Dataset(B[oz], label=B.y - B._ofs, categorical_feature=KAT)],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    ni = max(40, int(m.best_iteration * 1.1))
    log(et, "best_iter", m.best_iteration, "-> nihai", ni)
    mf = lgb.train(PAR, lgb.Dataset(a[oz], label=a.y - a._ofs, categorical_feature=KAT),
                   num_boost_round=ni)
    for ad in ("yaz25", "test"):
        d = bloklar[ad]
        s2 = d.soguk == msk
        d.loc[s2, "p4"] = np.clip(mf.predict(d.loc[s2, oz], num_iteration=ni)
                                  + d.loc[s2, "_ofs"].to_numpy(), 0, None)
    v = bloklar["yaz25"]
    vv = v[v.soguk == msk]
    R[et] = float(np.sqrt(((vv.p4 - vv.y) ** 2).mean()))
    log(et, "yaz25 RMSLE", round(R[et], 5))
    if et == "soguk":
        im = pd.Series(mf.feature_importance("gain"), index=oz).sort_values(ascending=False)
        log("soguk en onemli 12:", list(im.head(12).index))

Y = bloklar["yaz25"]
r = Y.p4.to_numpy() - Y.y.to_numpy()
R["duz"] = float(np.sqrt((r * r).mean()))
R["test_agirlikli"] = float(np.sqrt(0.7784 * R["sicak"] ** 2 + 0.2216 * R["soguk"] ** 2))
log("v4 yaz25", R)
Y[["tanim", "tarih", "y", "soguk", "p4"]].to_parquet(f"{SC}/p02_yaz25_v4.parquet")
bloklar["test"][["id", "tanim", "tarih", "soguk", "p4"]].to_parquet(f"{SC}/p02_test_v4.parquet")
json.dump(R, open(f"{K}/experiments/model29/p02_temiz_taban_v4.json", "w"), indent=1)
