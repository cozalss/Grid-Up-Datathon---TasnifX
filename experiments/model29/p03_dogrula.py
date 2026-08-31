"""p03 dogrulama: F1 (iki-asamali) ve F4 (idnum-komsu) saglam mi, GERCEK teste tasinir mi?

Kritik risk (README KURAL): idnum kimlik-ezberi yaz25'te cozulebilir, TESTTE %0.
F4'un kapsamini GERCEK test kesiminde (2026-03-31) olcmeden kazanc sayilmaz.
"""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p03_tezgah as T  # noqa: E402
from p03_fikir_ortak import komsu_ozellik  # noqa: E402

BURA = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
R = {}


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


tr, te = T.ortam()

# ---------- A. F4 kapsami: yaz25 vs GERCEK TEST ----------
def kapsam(gec, hed):
    hm = hed.drop_duplicates("tanim")[["tanim", "ilce", "idnum", "guc"]]
    kom = komsu_ozellik(gec, hm)
    v = hed.tanim.map(kom)
    return float(v.notna().mean()), kom


d_gec = tr[(tr.tarih <= T.D_KESIM) & (tr.tarih >= T.D_GEC_BAS)]
d_hed = tr[(tr.tarih > T.D_KESIM) & (tr.tarih <= T.D_HED_SON)]
d_soguk = set(d_hed.tanim) - set(d_gec.tanim)
kap_d, _ = kapsam(d_gec, d_hed)
sd = d_hed[d_hed.tanim.isin(d_soguk)]
kap_ds, _ = kapsam(d_gec, sd)

t_gec = tr[tr.tarih >= "2026-01-01"]
t_gec_tam = tr
t_soguk = set(te.tanim) - set(tr.tanim)
st = te[te.tanim.isin(t_soguk)]
kap_t, _ = kapsam(t_gec_tam, te)
kap_ts, _ = kapsam(t_gec_tam, st)
R["f4_kapsam"] = {
    "yaz25_tum_satir": kap_d, "yaz25_soguk_satir": kap_ds,
    "test_tum_satir": kap_t, "test_soguk_satir": kap_ts,
    "yaz25_soguk_satir_orani": float(d_hed.tanim.isin(d_soguk).mean()),
    "test_soguk_satir_orani": float(te.tanim.isin(t_soguk).mean()),
}
log("F4 kapsam:", json.dumps(R["f4_kapsam"], ensure_ascii=False))

# ---------- B. F1'in ayristirmasi ve tohum saglamligi ----------
Xe, ye, Xd, yd, hd, _ = T.veri(tr)
soguk_d = hd.tanim.isin(d_soguk).to_numpy()
PK = dict(objective="l2", metric="l2", learning_rate=0.04, num_leaves=63,
          min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=5.0, num_threads=8, verbose=-1)
PKC = dict(PK, objective="binary", metric="binary_logloss")
TUR = 600
poz = ye > 0
z_e = (ye == 0).astype(int)

tohumlar = [7, 17, 27]
sonuc = {"taban": [], "f1": [], "f1_soguk": [], "f1_sicak": []}
son_p0 = son_p1 = None
for s in tohumlar:
    pk = dict(PK, seed=s)
    p0 = lgb.train(pk, lgb.Dataset(Xe, ye), num_boost_round=TUR).predict(Xd)
    P0d = lgb.train(dict(PKC, seed=s), lgb.Dataset(Xe, z_e), num_boost_round=TUR).predict(Xd)
    ppos = lgb.train(pk, lgb.Dataset(Xe[poz], ye[poz]), num_boost_round=TUR).predict(Xd)
    p1 = (1 - P0d) * ppos
    sonuc["taban"].append(T.rmsle(yd, p0))
    sonuc["f1"].append(T.rmsle(yd, p1))
    sonuc["f1_soguk"].append(float(np.sqrt(np.mean((p1[soguk_d] - yd[soguk_d]) ** 2))))
    sonuc["f1_sicak"].append(float(np.sqrt(np.mean((p1[~soguk_d] - yd[~soguk_d]) ** 2))))
    son_p0, son_p1 = p0, p1
    log(f"tohum {s}: taban {sonuc['taban'][-1]:.5f} f1 {sonuc['f1'][-1]:.5f}")

R["f1_tohum"] = {k: {"ort": float(np.mean(v)), "std": float(np.std(v)), "hepsi": v}
                 for k, v in sonuc.items()}
R["f1_kazanc_ort"] = float(np.mean(sonuc["taban"]) - np.mean(sonuc["f1"]))

# kova bazinda ayristirma
y = hd.tuketim.to_numpy()
kesik = [-1, 0, 1, 10, 50, 100, 500, 1000, 5000, 1e5, 1e12]
etik = ["=0", "(0,1]", "(1,10]", "(10,50]", "(50,100]", "(100,500]",
        "(500,1e3]", "(1e3,5e3]", "(5e3,1e5]", ">1e5"]
kova = pd.cut(y, bins=kesik, labels=etik)
df = pd.DataFrame({"kova": kova, "e0": (son_p0 - yd) ** 2, "e1": (son_p1 - yd) ** 2})
g = df.groupby("kova", observed=False).agg(n=("e0", "size"), e0=("e0", "sum"), e1=("e1", "sum"))
g["pay_hata_taban"] = g.e0 / g.e0.sum()
g["pay_hata_f1"] = g.e1 / g.e1.sum()
g["duzelme"] = (g.e0 - g.e1) / g.e0.sum()
print(g.to_string())
R["f1_kova"] = json.loads(g.reset_index().to_json(orient="records"))

# ---------- C. F1 + F4 birlesik (F4 yalnizca yaz25'te gecerliyse) ----------
with open(os.path.join(BURA, "p03_dogrula.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
