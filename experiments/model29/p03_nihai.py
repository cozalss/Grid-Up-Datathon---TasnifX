"""p03 nihai: GERCEKCI taban (l1 tek asamali, uretim hattinin amaci) karsisinda
iki-asamali huber -- tohum tohum, soguk/sicak ayrimiyla, test bilesimine agirlikli."""

import json, math, os, sys, time
import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p03_tezgah as T
from p03_fikir_ortak import komsu_ozellik

B = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


SOGUK_TEST = 0.22159179949852242
tr, te = T.ortam()
Xe, ye, Xd, yd, hd, d_soguk = T.veri(tr)
sg = hd.tanim.isin(d_soguk).to_numpy()
poz = ye > 0
z = (ye == 0).astype(int)
P = dict(
    metric="l2",
    learning_rate=0.04,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    num_threads=8,
    verbose=-1,
)
PC = dict(P, objective="binary", metric="binary_logloss")
TUR = 600
HUB = dict(objective="huber", alpha=2.0, lambda_l2=20.0)

# komsu ozellikli surumler
e_gec = tr[(tr.tarih <= T.E_KESIM) & (tr.tarih >= T.E_GEC_BAS)]
e_hed = tr[(tr.tarih > T.E_KESIM) & (tr.tarih <= T.E_HED_SON)]
e_hed = e_hed[~e_hed.tanim.isin(d_soguk)].reset_index(drop=True)
d_gec = tr[(tr.tarih <= T.D_KESIM) & (tr.tarih >= T.D_GEC_BAS)]
ke = komsu_ozellik(
    e_gec[~e_gec.tanim.isin(d_soguk)], e_hed.drop_duplicates("tanim")[["tanim", "ilce", "idnum"]]
)
kd = komsu_ozellik(d_gec, hd.drop_duplicates("tanim")[["tanim", "ilce", "idnum"]])
Xe2 = Xe.assign(k_komsu=e_hed.tanim.map(ke).to_numpy())
Xd2 = Xd.assign(k_komsu=hd.tanim.map(kd).to_numpy())
log("hazir")


def olc(p):
    a = float(np.sqrt(np.mean((p - yd) ** 2)))
    s = float(np.sqrt(np.mean((p[sg] - yd[sg]) ** 2)))
    c = float(np.sqrt(np.mean((p[~sg] - yd[~sg]) ** 2)))
    w = math.sqrt((1 - SOGUK_TEST) * c * c + SOGUK_TEST * s * s)
    return {"rmsle": a, "soguk": s, "sicak": c, "test_agirlikli": w}


R = {"soguk_test_orani": SOGUK_TEST, "kosular": {}}
for ad, (Xtr_, Xte_, iki, ek) in {
    "A_tek_l1": (Xe, Xd, False, dict(objective="l1")),
    "B_iki_huber": (Xe, Xd, True, HUB),
    "C_iki_huber_kom": (Xe2, Xd2, True, HUB),
}.items():
    per = []
    for s in (7, 17, 27):
        if iki:
            p0 = lgb.train(dict(PC, seed=s), lgb.Dataset(Xtr_, z), num_boost_round=TUR).predict(
                Xte_
            )
            pp = lgb.train(
                dict(P, seed=s, **ek), lgb.Dataset(Xtr_[poz], ye[poz]), num_boost_round=TUR
            ).predict(Xte_)
            p = (1 - p0) * pp
        else:
            p = lgb.train(
                dict(P, seed=s, **ek), lgb.Dataset(Xtr_, ye), num_boost_round=TUR
            ).predict(Xte_)
        per.append(olc(p))
        log(f"{ad} tohum {s}: {per[-1]['rmsle']:.5f} (agirlikli {per[-1]['test_agirlikli']:.5f})")
    R["kosular"][ad] = {
        "tohumlar": per,
        "ort": {k: float(np.mean([q[k] for q in per])) for k in per[0]},
        "std": {k: float(np.std([q[k] for q in per])) for k in per[0]},
    }

a = R["kosular"]["A_tek_l1"]["ort"]
for ad in ("B_iki_huber", "C_iki_huber_kom"):
    o = R["kosular"][ad]["ort"]
    R["kosular"][ad]["kazanc_vs_A"] = {
        "yaz25": a["rmsle"] - o["rmsle"],
        "test_agirlikli": a["test_agirlikli"] - o["test_agirlikli"],
    }
json.dump(
    R, open(os.path.join(B, "p03_nihai.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False
)
print(json.dumps(R, indent=1, ensure_ascii=False))
