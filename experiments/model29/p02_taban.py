"""p02: SIFIRDAN temiz taban model. yaz25 geri-testi + mevcut boru hatti karsilastirmasi."""
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
meta = meta.drop_duplicates("tanim")
sp = meta.lokasyon.fillna(">").str.split(">", expand=True)
meta["ilce"] = sp[2].fillna("YOK") if sp.shape[1] > 2 else "YOK"
log("ham okundu", tr.shape, te.shape)

# ---- SIZINTI KAPISI 1: her blok icin gecmis KESIN olarak kesimden once ----
bloklar = {}
for ad, (k0, k1) in KESIM.items():
    kes = pd.Timestamp(k0)
    h = tr[tr.tarih < kes]
    assert h.tarih.max() < kes, f"{ad}: gecmis kesimi asiyor"
    if ad == "test":
        s = te[["id", "tanim", "guc", "tarih", "lokasyon"]].copy()
    else:
        s = tr[(tr.tarih >= kes) & (tr.tarih <= pd.Timestamp(k1))][
            ["tanim", "guc", "tarih", "lokasyon", "y"]].copy()
    grp = grup_onceligi(h[["tanim", "y"]], meta[["tanim", "guc", "ilce"]])
    d = blok_kur(s.reset_index(drop=True), h, meta, k0, grp)
    d["_blok"] = ad
    bloklar[ad] = d
    log(ad, d.shape, "soguk", round(float(d.soguk.mean()), 4),
        "gecmis gun", int((kes - h.tarih.min()).days))

# ---- SIZINTI KAPISI 2: yaz25 hedefi hicbir gecmis penceresinde yok ----
kes_y = pd.Timestamp("2025-04-01")
for ad in ("yaz25",):
    h = tr[tr.tarih < pd.Timestamp(KESIM[ad][0])]
    assert (h.tarih < kes_y).all()
log("SIZINTI KAPISI: yaz25 ozniteligi yalnizca 2025-01-01..03-31'den turedi -- TAMAM")

# ---- disgudumlu sutunlar (hedefe bagli DEGIL) ----
eg = pd.read_parquet(f"{DN}/egitim.parquet", columns=["tanim", "tarih"] + DIS)
tp = pd.read_parquet(f"{DN}/test.parquet", columns=["tanim", "tarih"] + DIS)
dis = pd.concat([eg, tp], ignore_index=True)
dis["tanim"] = dis.tanim.astype(str)
dis = dis.drop_duplicates(["tanim", "tarih"])
assert not any(c.startswith(("t_", "g_", "gp_", "p_", "ozet")) for c in DIS)
assert "soguk_mu" not in DIS and "tuketim" not in DIS
for ad in bloklar:
    n0 = len(bloklar[ad])
    bloklar[ad] = bloklar[ad].merge(dis, on=["tanim", "tarih"], how="left")
    assert len(bloklar[ad]) == n0
del eg, tp, dis
log("disgudumlu sutunlar birlestirildi")

KAT = ["il", "bolge", "ilce", "gk", "tk_hg", "tk_ay"]
ATLA = {"tanim", "tarih", "lokasyon", "y", "id", "_blok", "tuketim"}
OZ = [c for c in bloklar["guz25"].columns if c not in ATLA]
for ad in bloklar:
    for c in KAT:
        bloklar[ad][c] = bloklar[ad][c].astype(str)
kats = {c: sorted(set().union(*[set(bloklar[a][c]) for a in bloklar])) for c in KAT}
for ad in bloklar:
    for c in KAT:
        bloklar[ad][c] = pd.Categorical(bloklar[ad][c], categories=kats[c])
log("oznitelik sayisi", len(OZ))


def ofset(d):
    """Seviye capasi: sicak -> kendi gecmis ortalamasi, soguk -> grup."""
    o = d.h_ort.copy()
    o = o.fillna(d.gr_ilce_gk).fillna(d.gr_gk).fillna(d.gr_guc)
    return o.fillna(float(tr.y.mean())).to_numpy()


for ad in bloklar:
    bloklar[ad]["_ofs"] = ofset(bloklar[ad])
if "_ofs" not in OZ:
    OZ.append("_ofs")

PAR = dict(objective="regression", metric="l2", learning_rate=0.05,
           num_leaves=127, min_data_in_leaf=200, feature_fraction=0.7,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
           num_threads=8, verbosity=-1, max_bin=127)


def veri(d, ofs_hedef=True):
    y = d.y.to_numpy() - (d._ofs.to_numpy() if ofs_hedef else 0.0)
    return lgb.Dataset(d[OZ], label=y, categorical_feature=KAT, free_raw_data=False)


def rmsle(p, d):
    return float(np.sqrt(np.mean((p - d.y.to_numpy()) ** 2)))


SONUC = {}
for ofs_hedef in (True, False):
    et = "ofsetli" if ofs_hedef else "ham"
    # tur sayisi: kis26'da egit, guz25'te erken durdur (yaz25'e HIC dokunmaz)
    dtr = veri(bloklar["kis26"], ofs_hedef)
    dvl = veri(bloklar["guz25"], ofs_hedef)
    m = lgb.train(PAR, dtr, num_boost_round=3000, valid_sets=[dvl],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    ni = int(m.best_iteration * 1.15)
    log(et, "erken durdurma turu (guz25 dogrulama)", m.best_iteration, "-> nihai", ni)
    # nihai: guz25+kis26 birlikte
    trn = pd.concat([bloklar["guz25"], bloklar["kis26"]], ignore_index=True)
    assert (trn.tarih >= pd.Timestamp("2025-08-01")).all(), "yaz25 satiri egitimde!"
    mf = lgb.train(PAR, veri(trn, ofs_hedef), num_boost_round=ni)
    for ad in ("yaz25", "test"):
        d = bloklar[ad]
        p = mf.predict(d[OZ], num_iteration=ni)
        if ofs_hedef:
            p = p + d._ofs.to_numpy()
        bloklar[ad][f"p_{et}"] = np.clip(p, 0.0, None)
    r = rmsle(bloklar["yaz25"][f"p_{et}"].to_numpy(), bloklar["yaz25"])
    SONUC[et] = r
    log(et, "yaz25 RMSLE", round(r, 6))

# karisim
for ad in ("yaz25", "test"):
    bloklar[ad]["p_kar"] = 0.5 * (bloklar[ad].p_ofsetli + bloklar[ad].p_ham)
SONUC["karisim"] = rmsle(bloklar["yaz25"].p_kar.to_numpy(), bloklar["yaz25"])
log("karisim yaz25 RMSLE", round(SONUC["karisim"], 6))

bloklar["yaz25"].to_parquet(f"{SC}/p02_yaz25.parquet")
bloklar["test"][["id", "tanim", "tarih", "soguk", "p_ofsetli", "p_ham", "p_kar"]].to_parquet(
    f"{SC}/p02_test.parquet")
json.dump(SONUC, open(f"{K}/experiments/model29/p02_temiz_taban.json", "w"), indent=1)
log("BITTI", SONUC)
