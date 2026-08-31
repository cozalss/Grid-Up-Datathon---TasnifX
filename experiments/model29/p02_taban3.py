"""p02 v3: kimlik-onek/komsuluk oncelikleri + modele uygun dogrulama semasi.

Sicak model  : dogrulama = egitim bloklarinin SON 40 gunu (uzak ufku taklit eder)
Soguk model  : dogrulama = ayrilan TRAFOLAR (yaz25 soguk satirlari da gorulmemis trafo)
Her iki durumda da yaz25 hedefi HICBIR yerde yok.
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
meta["num"] = pd.to_numeric(meta.tanim, errors="coerce")


def kimlik_oncelik(h, d):
    """Kimlik onegi + en yakin kimlik komsulari -- YALNIZCA gecmisten."""
    ty = h.groupby("tanim", observed=True).y.mean().rename("ty")
    ref = meta.merge(ty, left_on="tanim", right_index=True, how="inner")
    ref = ref.dropna(subset=["num"]).sort_values("num").reset_index(drop=True)
    for n in (3, 4, 5, 6):
        pre = ref.tanim.str[:n]
        mm = ref.groupby(pre).ty.agg(["mean", "size"])
        p = d.tanim.str[:n]
        d[f"k_on{n}"] = p.map(mm["mean"]).to_numpy()
        d[f"k_on{n}_n"] = p.map(mm["size"]).to_numpy()
    # kimlik ekseninde en yakin K komsu (kendisi haric)
    rn = ref.num.to_numpy()
    rt = ref.ty.to_numpy()
    rtan = ref.tanim.to_numpy()
    dn = pd.to_numeric(d.tanim, errors="coerce").to_numpy()
    pos = np.searchsorted(rn, dn)
    for kk in (5, 25):
        v = np.full(len(d), np.nan)
        for i in range(len(d)):
            if not np.isfinite(dn[i]):
                continue
            a, b = max(0, pos[i] - kk), min(len(rn), pos[i] + kk)
            sel = rtan[a:b] != d.tanim.iat[i]
            if sel.any():
                v[i] = rt[a:b][sel].mean()
        d[f"k_komsu{kk}"] = v
    return d


bloklar = {}
for ad, (k0, k1) in KESIM.items():
    kes = pd.Timestamp(k0)
    h = tr[tr.tarih < kes]
    assert h.tarih.max() < kes
    if ad == "test":
        s = te[["id", "tanim", "guc", "tarih", "lokasyon"]].copy()
    else:
        s = tr[(tr.tarih >= kes) & (tr.tarih <= pd.Timestamp(k1))][
            ["tanim", "guc", "tarih", "lokasyon", "y"]].copy()
    grp = grup_onceligi(h[["tanim", "y"]], meta[["tanim", "guc", "ilce"]])
    d = blok_kur(s.reset_index(drop=True), h, meta, k0, grp)
    # kimlik oncelikleri trafo duzeyinde -> tekil trafolarda hesapla, geri bagla
    u = pd.DataFrame({"tanim": d.tanim.unique()})
    u = kimlik_oncelik(h[["tanim", "y"]], u)
    d = d.merge(u, on="tanim", how="left")
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
    sog_ofs = (d.k_on6.fillna(d.k_on5).fillna(d.k_komsu5).fillna(d.k_on4)
               .fillna(d.gr_ilce_gk).fillna(d.gr_gk).fillna(d.gr_guc))
    d["_ofs"] = d.h_ort.fillna(sog_ofs).fillna(float(tr.y.mean())).to_numpy()

TUM = [c for c in bloklar["guz25"].columns if c not in ATLA]
SOZ = [c for c in TUM if not c.startswith("h_")]
log("sicak oz", len(TUM), "soguk oz", len(SOZ))

PAR = dict(objective="regression", metric="l2", learning_rate=0.04,
           num_leaves=127, min_data_in_leaf=200, feature_fraction=0.6,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
           num_threads=8, verbosity=-1, max_bin=255)

trn = pd.concat([bloklar["guz25"], bloklar["kis26"]], ignore_index=True)
assert (trn.tarih >= pd.Timestamp("2025-08-01")).all(), "SIZINTI: yaz25 egitimde"
# sicak dogrulama: her blogun SON 40 gunu
son = (((trn.tarih >= "2025-10-22") & (trn.tarih <= "2025-11-30"))
       | (trn.tarih >= "2026-02-20"))
rs = np.random.RandomState(7)
tans = trn.tanim.unique()
hold = set(rs.choice(tans, size=int(0.15 * len(tans)), replace=False))
trn["_v_sicak"] = son
trn["_v_soguk"] = trn.tanim.isin(hold)

SON = {}
for et, oz, msk, vk in (("sicak", TUM, 0, "_v_sicak"), ("soguk", SOZ, 1, "_v_soguk")):
    a = trn[trn.soguk == msk]
    A, B = a[~a[vk]], a[a[vk]]
    m = lgb.train(PAR, lgb.Dataset(A[oz], label=A.y - A._ofs, categorical_feature=KAT),
                  num_boost_round=5000,
                  valid_sets=[lgb.Dataset(B[oz], label=B.y - B._ofs, categorical_feature=KAT)],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    ni = max(100, int(m.best_iteration * 1.1))
    log(et, "n", len(a), "best_iter", m.best_iteration, "-> nihai", ni,
        "dogrulama RMSE", round(float(m.best_score["valid_0"]["l2"]) ** 0.5, 5))
    mf = lgb.train(PAR, lgb.Dataset(a[oz], label=a.y - a._ofs, categorical_feature=KAT),
                   num_boost_round=ni)
    for ad in ("yaz25", "test"):
        d = bloklar[ad]
        s = d.soguk == msk
        d.loc[s, "p3"] = np.clip(mf.predict(d.loc[s, oz], num_iteration=ni)
                                 + d.loc[s, "_ofs"].to_numpy(), 0.0, None)
    if et == "soguk":
        yv = bloklar["yaz25"]
        s = yv.soguk == 1
        rr = yv.loc[s, "_ofs"].to_numpy() - yv.loc[s, "y"].to_numpy()
        log("  (soguk SADECE ofset capasi RMSLE", round(float(np.sqrt((rr * rr).mean())), 5), ")")

y = bloklar["yaz25"].y.to_numpy()
p = bloklar["yaz25"].p3.to_numpy()
sg = bloklar["yaz25"].soguk.to_numpy()
r = p - y
SON = dict(duz=float(np.sqrt((r * r).mean())),
           sicak=float(np.sqrt((r[sg == 0] ** 2).mean())),
           soguk=float(np.sqrt((r[sg == 1] ** 2).mean())))
SON["test_agirlikli"] = float(np.sqrt(0.2216 * SON["soguk"] ** 2 + 0.7784 * SON["sicak"] ** 2))
log("v3 yaz25", SON)
bloklar["yaz25"][["tanim", "tarih", "y", "soguk", "p3"]].to_parquet(f"{SC}/p02_yaz25_v3.parquet")
bloklar["test"][["id", "tanim", "tarih", "soguk", "p3"]].to_parquet(f"{SC}/p02_test_v3.parquet")
json.dump(SON, open(f"{K}/experiments/model29/p02_temiz_taban_v3.json", "w"), indent=1)
