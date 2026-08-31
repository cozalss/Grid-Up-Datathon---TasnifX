"""p03 fikir-2: amac fonksiyonu (RMSLE ortalama ister, medyan degil) + F1xF4 birlesimi."""

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


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


tr, te = T.ortam()
Xe, ye, Xd, yd, hd, d_soguk = T.veri(tr)
soguk_d = hd.tanim.isin(d_soguk).to_numpy()
poz = ye > 0
z_e = (ye == 0).astype(int)
TABAN_PK = dict(metric="l2", learning_rate=0.04, num_leaves=63, min_data_in_leaf=200,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=5.0, num_threads=8, verbose=-1)
TUR = 600
TOHUM = [7, 17, 27]
R = {}


def kos(pk_ek, X_e, X_d, y_e, tur=TUR):
    ps = []
    for s in TOHUM:
        pk = dict(TABAN_PK, seed=s, **pk_ek)
        ps.append(lgb.train(pk, lgb.Dataset(X_e, y_e), num_boost_round=tur).predict(X_d))
    return np.mean(ps, axis=0)


# ---- komsu ozellikli matrisler ----
e_gec = tr[(tr.tarih <= T.E_KESIM) & (tr.tarih >= T.E_GEC_BAS)]
e_hed = tr[(tr.tarih > T.E_KESIM) & (tr.tarih <= T.E_HED_SON)]
e_hed = e_hed[~e_hed.tanim.isin(d_soguk)].reset_index(drop=True)
d_gec = tr[(tr.tarih <= T.D_KESIM) & (tr.tarih >= T.D_GEC_BAS)]
ke = komsu_ozellik(e_gec[~e_gec.tanim.isin(d_soguk)],
                   e_hed.drop_duplicates("tanim")[["tanim", "ilce", "idnum"]])
kd = komsu_ozellik(d_gec, hd.drop_duplicates("tanim")[["tanim", "ilce", "idnum"]])
Xe2 = Xe.assign(k_komsu=e_hed.tanim.map(ke).to_numpy())
Xd2 = Xd.assign(k_komsu=hd.tanim.map(kd).to_numpy())
log("komsu ozellikleri hazir")

# ---- A. amac fonksiyonu karsilastirmasi (tek asamali) ----
R["amac_tek_asamali"] = {}
for ad, ek in [("l2", dict(objective="l2")),
               ("huber2", dict(objective="huber", alpha=2.0, lambda_l2=20.0)),
               ("l1", dict(objective="l1"))]:
    p = kos(ek, Xe, Xd, ye)
    R["amac_tek_asamali"][ad] = T.rmsle(yd, p)
    log(f"amac {ad}: {R['amac_tek_asamali'][ad]:.5f}")
taban = R["amac_tek_asamali"]["l2"]
R["taban_l2"] = taban

# ---- B. iki asamali, amac karsilastirmasi ----
PKC = dict(TABAN_PK, objective="binary", metric="binary_logloss")
P0 = np.mean([lgb.train(dict(PKC, seed=s), lgb.Dataset(Xe, z_e),
                        num_boost_round=TUR).predict(Xd) for s in TOHUM], axis=0)
R["amac_iki_asamali"] = {}
for ad, ek in [("l2", dict(objective="l2")),
               ("huber2", dict(objective="huber", alpha=2.0, lambda_l2=20.0)),
               ("l1", dict(objective="l1"))]:
    pp = kos(ek, Xe[poz], Xd, ye[poz])
    v = T.rmsle(yd, (1 - P0) * pp)
    R["amac_iki_asamali"][ad] = v
    log(f"iki-asamali {ad}: {v:.5f} (kazanc {taban - v:+.5f})")

# ---- C. F1 x F4 birlesimi ----
P0b = np.mean([lgb.train(dict(PKC, seed=s), lgb.Dataset(Xe2, z_e),
                         num_boost_round=TUR).predict(Xd2) for s in TOHUM], axis=0)
pp2 = kos(dict(objective="l2"), Xe2[poz], Xd2, ye[poz])
p_f14 = (1 - P0b) * pp2
p_f4 = kos(dict(objective="l2"), Xe2, Xd2, ye)
R["f4_tek"] = {"rmsle": T.rmsle(yd, p_f4), "kazanc": taban - T.rmsle(yd, p_f4)}
R["f1xf4"] = {"rmsle": T.rmsle(yd, p_f14), "kazanc": taban - T.rmsle(yd, p_f14)}
for k in ("f4_tek", "f1xf4"):
    log(f"{k}: {R[k]['rmsle']:.5f} ({R[k]['kazanc']:+.5f})")
R["f1xf4_soguk"] = float(np.sqrt(np.mean((p_f14[soguk_d] - yd[soguk_d]) ** 2)))
R["f1xf4_sicak"] = float(np.sqrt(np.mean((p_f14[~soguk_d] - yd[~soguk_d]) ** 2)))

# ---- D. F1'in iki bileseni ayri ayri ----
pp_sadece = kos(dict(objective="l2"), Xe[poz], Xd, ye[poz])
R["f1_bilesen"] = {
    "sadece_sifirsiz_egitim": T.rmsle(yd, pp_sadece),
    "sadece_sifirsiz_egitim_kazanc": taban - T.rmsle(yd, pp_sadece),
    "tam_iki_asamali": R["amac_iki_asamali"]["l2"],
}
log(f"F1 bilesen: sadece sifirsiz egitim {R['f1_bilesen']['sadece_sifirsiz_egitim']:.5f}")

with open(os.path.join(BURA, "p03_fikir2.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
