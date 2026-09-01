"""p35-c: rho'nun ANATOMISI + tavan.

1) Bayrakli kumede rho'yu DOGRU POZITIF / YANLIS POZITIF olarak ayristir.
2) Precision/recall egrisi boyunca rho.
3) TAVAN: (q,p) sigma-cebiri icinde en iyi yon -- kovalanmis E[r|q,p]
   DIGER iki bloktan ogrenilir, hedefte uygulanir (durust).
4) TAVAN-2: butun oznitelikler uzerinde blok-disi artik regresyonu
   (herhangi bir ogrenilebilir yonun durust rho tavani).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p27_ortak import agirlik, blok  # noqa: E402

CIK = os.path.dirname(os.path.abspath(__file__))
BLOKLAR = ("yaz25", "guz25", "kis26")
R = {}

D = {b: blok(b, soguk_harman="cat", son_islem=True) for b in BLOKLAR}
W = {b: agirlik(D[b]) for b in BLOKLAR}
_v = pd.unique(pd.concat([D[b].ilce_key.astype(str) for b in BLOKLAR]))
_kod = {v: i for i, v in enumerate(_v)}
for b in BLOKLAR:
    D[b] = D[b].copy()
    D[b]["ilce_kod"] = D[b].ilce_key.astype(str).map(_kod).astype("float32")

OZ = ["guc", "ufuk_gun", "soguk_mu", "t_sifir_orani", "t_kuyruk_sifir", "t_olu_mu",
      "t_son_kayit_yasi", "t_log_ort", "t_log_std", "t_log_medyan", "t_log_p10",
      "t_gun_sayisi", "t_doluluk", "t_yuk_faktoru", "t_trend",
      "t_log_son7", "t_log_son30", "t_log_son90", "t_son30_gun", "t_son90_gun",
      "t_gy_sifir_orani", "t_gy_log_ort", "g_guc_kova", "g_ilce_log_ort",
      "g_kova_log_ort", "ilce_kod", "guc_yuzdelik", "tatil_mi", "sicaklik_ort",
      "ozet_pencere_gun", "p_doluluk", "p_son_ofset", "yas"]
KUL = [c for c in OZ if c in D["yaz25"].columns]
PAR = dict(objective="binary", learning_rate=0.05, num_leaves=63, verbose=-1,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           min_data_in_leaf=20)

Q = {}
for hed in BLOKLAR:
    tr = pd.concat([D[b] for b in BLOKLAR if b != hed])
    m = lgb.train(dict(PAR, seed=1000),
                  lgb.Dataset(tr[KUL].astype("float32"),
                              (tr.tuketim.values <= 0).astype(int)),
                  num_boost_round=300)
    Q[hed] = m.predict(D[hed][KUL].astype("float32"))
    # ayni model diger bloklara da (kovalama icin egitim bloklarinda q lazim)
print("q hazir")


def rho_of(r, dl, w):
    k2 = float(np.sum(w * dl * dl) / np.sum(w))
    if k2 <= 0:
        return 0.0, 0.0
    kap = np.sqrt(k2)
    return float(np.sum(w * r * dl) / np.sum(w)) / kap, float(kap)


# ---------- 1+2) esik egrisi ve ayristirma ----------
egri = []
for b in BLOKLAR:
    d, w, q = D[b], W[b], Q[b]
    p, r = d.p.values, d.r.values
    z = d.tuketim.values <= 0
    sw = np.sum(w)
    for e in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99):
        f = q > e
        if f.sum() < 50:
            continue
        dl = -p * f
        rho, kap = rho_of(r, dl, w)
        # ayristirma: pay = E[r*delta]; DP(z) ve YP(~z) katkilari
        pay_dp = float(np.sum(w[f & z] * r[f & z] * dl[f & z]) / sw)
        pay_yp = float(np.sum(w[f & ~z] * r[f & ~z] * dl[f & ~z]) / sw)
        egri.append(dict(
            blok=b, esik=e,
            pay=round(float(np.sum(w * f) / sw), 5),
            kesinlik=round(float(np.sum(w[f] * z[f]) / np.sum(w[f])), 4),
            duyarlilik=round(float(np.sum(w[f & z]) / np.sum(w[z])), 4),
            ort_p_DP=round(float(np.average(p[f & z], weights=w[f & z])), 3),
            ort_p_YP=round(float(np.average(p[f & ~z], weights=w[f & ~z]))
                           if (f & ~z).sum() else np.nan, 3),
            ort_r_YP=round(float(np.average(r[f & ~z], weights=w[f & ~z]))
                           if (f & ~z).sum() else np.nan, 3),
            rho=round(rho, 5), kappa_tam=round(kap, 4),
            rho_pay_DP=round(pay_dp / kap, 5), rho_pay_YP=round(pay_yp / kap, 5),
        ))
print("\n=== esik egrisi: rho ve DP/YP ayristirmasi ===")
EG = pd.DataFrame(egri)
print(EG.to_string(index=False))
R["01_egri"] = egri

# ---------- 3) TAVAN: (q,p) kovalari, DURUST ----------
def kova_yonu(hed, nq=12, np_=6):
    tr = pd.concat([D[b] for b in BLOKLAR if b != hed])
    wtr = np.concatenate([W[b] for b in BLOKLAR if b != hed])
    # egitim bloklarinda q: kendi blok-disi modeliyle (Q[b] zaten oyle)
    qtr = np.concatenate([Q[b] for b in BLOKLAR if b != hed])
    qk = np.linspace(0, 1, nq + 1)[1:-1]
    pk = np.quantile(np.concatenate([D[b].p.values for b in BLOKLAR if b != hed]),
                     np.linspace(0, 1, np_ + 1)[1:-1])
    itr = np.digitize(qtr, qk) * 100 + np.digitize(tr.p.values, pk)
    rtr = tr.r.values
    df = pd.DataFrame(dict(k=itr, r=rtr, w=wtr))
    df["rw"] = df.r * df.w
    g = df.groupby("k").agg(rw=("rw", "sum"), w=("w", "sum"))
    mu = (g.rw / g.w)
    te = D[hed]
    ite = np.digitize(Q[hed], qk) * 100 + np.digitize(te.p.values, pk)
    return mu.reindex(ite).fillna(0.0).to_numpy()


tav = []
for hed in BLOKLAR:
    dl = kova_yonu(hed)
    rho, kap = rho_of(D[hed].r.values, dl, W[hed])
    tav.append(dict(blok=hed, yon="kova_E[r|q,p]_blokdisi",
                    rho=round(rho, 5), kappa_tam=round(kap, 4)))
    print(tav[-1])

# ---------- 4) TAVAN-2: tam artik regresyonu, blok-disi ----------
PARR = dict(objective="regression", learning_rate=0.05, num_leaves=63, verbose=-1,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            min_data_in_leaf=50)
for hed in BLOKLAR:
    tr = pd.concat([D[b] for b in BLOKLAR if b != hed])
    wtr = np.concatenate([W[b] for b in BLOKLAR if b != hed])
    m = lgb.train(dict(PARR, seed=7),
                  lgb.Dataset(tr[KUL].astype("float32"), tr.r.values, weight=wtr),
                  num_boost_round=400)
    dl = m.predict(D[hed][KUL].astype("float32"))
    rho, kap = rho_of(D[hed].r.values, dl, W[hed])
    tav.append(dict(blok=hed, yon="tam_artik_regresyon_blokdisi",
                    rho=round(rho, 5), kappa_tam=round(kap, 4)))
    print(tav[-1])
    # sadece SIFIR CEBI oznitelikleri (mevsim/ufuk yok) -- tezgah yapaylugindan arinik
    SF = [c for c in KUL if c.startswith(("t_", "p_", "g_")) or c in ("guc", "yas",
          "guc_yuzdelik", "ozet_pencere_gun", "soguk_mu")]
    m2 = lgb.train(dict(PARR, seed=7),
                   lgb.Dataset(tr[SF].astype("float32"), tr.r.values, weight=wtr),
                   num_boost_round=400)
    dl2 = m2.predict(D[hed][SF].astype("float32"))
    rho2, kap2 = rho_of(D[hed].r.values, dl2, W[hed])
    tav.append(dict(blok=hed, yon="artik_regresyon_yalniz_trafo_oz",
                    rho=round(rho2, 5), kappa_tam=round(kap2, 4)))
    print(tav[-1])
R["02_tavan"] = tav

with open(os.path.join(CIK, "p35_c.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=str)
print("\nyazildi p35_c.json")
