"""p27-06: SIFIR CEBI -- TRAFO DUZEYI "olu" tespiti + optimal yumusak buzme.

p27-05 bulgusu: blok sifirlarinin %68-84'u, o blokta 4 AY BOYUNCA HIC
uretmeyen trafolardan geliyor (186/139/178 trafo). Yani asil problem satir
duzeyi degil TRAFO duzeyi bir siniflandirma: "bu trafo onumuzdeki 4 ayda
tamamen sessiz mi kalacak?"

Cebir: trafo icin sessizlik olasiligi q_T ise, log tahmini (1-q_T) ile
carpmak MSE-optimaldir; sert esik gerekmez.

Butun egitimler BLOK-DISI (hedef blok disindaki iki blokta fit).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p27_ortak import HEDEF_SOGUK, blok, rmsle  # noqa: E402

CIK = os.path.dirname(os.path.abspath(__file__))
R = {}
BLOKLAR = ("yaz25", "guz25", "kis26")
D = {b: blok(b) for b in BLOKLAR}

OZ = ["guc", "ufuk_gun", "soguk_mu", "t_sifir_orani", "t_kuyruk_sifir", "t_olu_mu",
      "t_son_kayit_yasi", "t_log_ort", "t_log_std", "t_log_medyan", "t_log_p10",
      "t_gun_sayisi", "t_doluluk", "t_yuk_faktoru", "t_trend",
      "t_log_son7", "t_log_son30", "t_log_son90", "t_son30_gun",
      "t_gy_sifir_orani", "t_gy_log_ort", "g_guc_kova", "g_ilce_log_ort",
      "g_kova_log_ort", "ilce_kod", "guc_yuzdelik", "tatil_mi", "sicaklik_ort",
      "ozet_pencere_gun", "p_doluluk", "p_son_ofset", "yas"]

# ortak ilce kodlamasi
_v = pd.unique(pd.concat([D[b].ilce_key.astype(str) for b in BLOKLAR]))
_kod = {v: i for i, v in enumerate(_v)}
for b in BLOKLAR:
    D[b] = D[b].copy()
    D[b]["ilce_kod"] = D[b].ilce_key.astype(str).map(_kod).astype("float32")

KUL = [c for c in OZ if c in D["yaz25"].columns]
# trafo duzeyi ozellikler (satir icinde sabit olanlar)
T_KUL = [c for c in KUL if c not in
         ("ufuk_gun", "tatil_mi", "sicaklik_ort", "soguk_mu")]
print("satir oznitelik:", len(KUL), " trafo oznitelik:", len(T_KUL))


def bilesik(sic, sog):
    return float(np.sqrt(HEDEF_SOGUK * sog**2 + (1 - HEDEF_SOGUK) * sic**2))


def bil_of(d, p):
    sg = d.soguk_mu.values == 1
    return bilesik(rmsle(d.y.values[~sg], p[~sg]), rmsle(d.y.values[sg], p[sg]))


PAR = dict(objective="binary", learning_rate=0.05, num_leaves=31, verbose=-1,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           min_data_in_leaf=20)


def trafo_tablo(b):
    """her trafo icin: TAM sessiz mi + trafo duzeyi ozellikler (ilk satir)."""
    d = D[b]
    z = (d.tuketim.values <= 0)
    g = pd.DataFrame(dict(t=d.tanim.values, z=z, sg=d.soguk_mu.values))
    y = g.groupby("t").z.mean()
    ilk = d.groupby("tanim")[T_KUL].first()
    out = ilk.copy()
    out["hedef"] = (y.reindex(out.index) == 1.0).astype(int)
    out["sifir_orani"] = y.reindex(out.index)
    out["n_satir"] = g.groupby("t").size().reindex(out.index)
    return out


TT = {b: trafo_tablo(b) for b in BLOKLAR}
print({b: dict(n=len(TT[b]), poz=int(TT[b].hedef.sum())) for b in BLOKLAR})

sonuc, auc_tab = [], []
for hed in BLOKLAR:
    te = D[hed]
    taban = bil_of(te, te.p.values)

    # ---------- (A) TRAFO duzeyi q_T ----------
    trT = pd.concat([TT[b] for b in BLOKLAR if b != hed])
    teT = TT[hed]
    for tohum in (1000, 1001, 1002):
        pass
    qs = []
    for tohum in (1000, 1001, 1002):
        m = lgb.train(dict(PAR, seed=tohum),
                      lgb.Dataset(trT[T_KUL].astype("float32"), trT.hedef.values),
                      num_boost_round=250)
        qs.append(m.predict(teT[T_KUL].astype("float32")))
    qT = np.mean(qs, axis=0)
    qT_s = pd.Series(qT, index=teT.index)
    q_row_T = qT_s.reindex(te.tanim.values).to_numpy()

    # ---------- (B) SATIR duzeyi q ----------
    tr = pd.concat([D[b] for b in BLOKLAR if b != hed])
    mr = lgb.train(dict(PAR, num_leaves=63, seed=1000),
                   lgb.Dataset(tr[KUL].astype("float32"),
                               (tr.tuketim.values <= 0).astype(int)),
                   num_boost_round=300)
    q_row = mr.predict(te[KUL].astype("float32"))

    yte = (te.tuketim.values <= 0).astype(int)
    sg = te.soguk_mu.values == 1
    auc_tab.append(dict(
        blok=hed,
        trafo_AUC=round(float(roc_auc_score(teT.hedef.values, qT)), 4),
        trafo_poz=int(teT.hedef.sum()), trafo_n=int(len(teT)),
        satir_AUC=round(float(roc_auc_score(yte, q_row)), 4),
        satir_AUC_sicak=round(float(roc_auc_score(yte[~sg], q_row[~sg])), 4),
        satir_AUC_soguk=round(float(roc_auc_score(yte[sg], q_row[sg])), 4),
        satirda_trafoq_AUC=round(float(roc_auc_score(yte, q_row_T)), 4),
    ))

    def kaydet(ad, p2):
        sonuc.append(dict(hedef=hed, yontem=ad, kazanc=round(taban - bil_of(te, p2), 5)))

    for gam in (0.5, 1.0, 1.5, 2.0):
        kaydet(f"A_trafo_(1-qT)^{gam}", te.p.values * (1.0 - q_row_T) ** gam)
    for gam in (0.5, 1.0, 1.5, 2.0):
        kaydet(f"B_satir_(1-q)^{gam}", te.p.values * (1.0 - q_row) ** gam)
    for gam in (1.0, 1.5):
        kaydet(f"C_bilesik_(1-max)^{gam}",
               te.p.values * (1.0 - np.maximum(q_row, q_row_T)) ** gam)
    for esik in (0.3, 0.5, 0.7, 0.9):
        p2 = te.p.values.copy(); p2[q_row_T > esik] = 0.0
        kaydet(f"D_trafo_sert_{esik}", p2)
    # yalniz yuksek q_T'de yumusak, digerlerinde dokunma
    for esik in (0.2, 0.4):
        p2 = te.p.values.copy()
        m2 = q_row_T > esik
        p2[m2] = p2[m2] * (1.0 - q_row_T[m2])
        kaydet(f"E_trafo_kapili_{esik}", p2)

R["01_auc"] = auc_tab
R["02_sonuc"] = sonuc
print("\n1) BLOK-DISI AUC:")
for x in auc_tab:
    print("  ", x)
tb = pd.DataFrame(sonuc).pivot(index="yontem", columns="hedef", values="kazanc")
tb = tb[list(BLOKLAR)]
tb["ORT"] = tb.mean(axis=1)
tb["isaret"] = (tb[list(BLOKLAR)] > 0).sum(axis=1).astype(str) + "/3"
print("\n2) KOHORT AGIRLIKLI BILESIK KAZANC (+ = iyi):")
print(tb.to_string())
R["03_tablo"] = json.loads(tb.to_json())
R["04_lb_olcek"] = dict(
    olcek=0.93907,
    en_iyi={str(i): round(float(tb.loc[i, "ORT"]) * 0.93907, 5)
            for i in tb.index})
print("\n3) LB olcekli ORT kazanc:")
for k, v in sorted(R["04_lb_olcek"]["en_iyi"].items(), key=lambda x: -x[1]):
    print(f"   {k:28} {v:+.5f}")

with open(os.path.join(CIK, "p27_06.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=str)
print("\nyazildi p27_06.json")
