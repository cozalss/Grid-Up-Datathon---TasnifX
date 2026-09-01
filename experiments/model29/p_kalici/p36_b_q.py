"""p36-b: olu-trafo olasiligi q (blok-disi) + MSE-KUTLE duyarliligi + rho aileleri.

q modelleri hedef blok DISINDAKI iki blokta egitilir; oznitelikler yalniz
t_*/p_*/g_* ozetleri (hedef blogun BASINDAN ONCEKI pencere) + statik trafo
bilgisi.  Hedef blogun y'si hicbir yerde kullanilmaz.
"""
import json, os, sys, pickle
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
sys.path.insert(0, os.path.join(KOK, "experiments/model29/p_kalici"))
from p27_ortak import agirlik, blok, rmsle  # noqa

CIK = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"
BLOKLAR = ("yaz25", "guz25", "kis26")
R = {}

D = {b: blok(b, soguk_harman="cat", son_islem=True) for b in BLOKLAR}
W = {b: agirlik(D[b]) for b in BLOKLAR}
_v = pd.unique(pd.concat([D[b].ilce_key.astype(str) for b in BLOKLAR]))
_kod = {v: i for i, v in enumerate(_v)}
for b in BLOKLAR:
    D[b] = D[b].copy()
    D[b]["ilce_kod"] = D[b].ilce_key.astype(str).map(_kod).astype("float32")

# gorevde istenen oznitelikler + mevcut ozetler.  ufuk/takvim YOK (tezgah yapayligi).
OZ = ["guc", "yas", "guc_yuzdelik", "soguk_mu", "ozet_pencere_gun",
      "t_sifir_orani", "t_kuyruk_sifir", "t_olu_mu", "t_son_kayit_yasi",
      "t_log_ort", "t_log_std", "t_log_medyan", "t_log_p10", "t_log_p90",
      "t_gun_sayisi", "t_doluluk", "t_yuk_faktoru", "t_trend",
      "t_log_son7", "t_son7_gun", "t_log_son14", "t_son14_gun",
      "t_log_son30", "t_son30_gun", "t_log_son60", "t_son60_gun",
      "t_log_son90", "t_son90_gun",
      "t_gy_sifir_orani", "t_gy_log_ort", "t_gy_gun",
      "g_guc_kova", "g_ilce_log_ort", "g_kova_log_ort", "g_ilce_kova_ort",
      "ilce_kod", "ilce_trafo_sayisi", "p_doluluk", "p_son_ofset", "p_gun_sayisi"]
KUL = [c for c in OZ if c in D["yaz25"].columns]
print("oznitelik n =", len(KUL))

PAR = dict(objective="binary", learning_rate=0.05, num_leaves=63, verbose=-1,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           min_data_in_leaf=40)

Q = {}
auc = []
for hed in BLOKLAR:
    tr = pd.concat([D[b] for b in BLOKLAR if b != hed])
    ytr = (tr.tuketim.values <= 0).astype(int)
    te = D[hed]; yte = (te.tuketim.values <= 0).astype(int)
    qs = []
    for s in (1000, 1001, 1002):
        m = lgb.train(dict(PAR, seed=s),
                      lgb.Dataset(tr[KUL].astype("float32"), ytr), num_boost_round=300)
        qs.append(m.predict(te[KUL].astype("float32")))
    q = np.mean(qs, axis=0)
    # p^2 agirlikli varyant: MSE kutlesinin oldugu yere odaklan
    wtr2 = np.concatenate([D[b].p.values ** 2 + 0.25 for b in BLOKLAR if b != hed])
    mw = lgb.train(dict(PAR, seed=1000),
                   lgb.Dataset(tr[KUL].astype("float32"), ytr, weight=wtr2),
                   num_boost_round=300)
    qw = mw.predict(te[KUL].astype("float32"))
    Q[hed] = dict(q=q, qw=qw)
    sg = te.soguk_mu.values == 1
    # trafo duzeyi
    g = pd.DataFrame(dict(t=te.tanim.values, z=yte.astype(bool)))
    olu = (g.groupby("t").z.mean() == 1.0)
    qT = pd.Series(q, index=te.tanim.values).groupby(level=0).mean()
    auc.append(dict(blok=hed,
        satir_AUC=round(float(roc_auc_score(yte, q)), 4),
        satir_AUC_pw=round(float(roc_auc_score(yte, qw)), 4),
        satir_AUC_sicak=round(float(roc_auc_score(yte[~sg], q[~sg])), 4),
        satir_AUC_soguk=round(float(roc_auc_score(yte[sg], q[sg])), 4),
        trafo_AUC=round(float(roc_auc_score(olu.values.astype(int),
                                            qT.reindex(olu.index).values)), 4),
        trafo_poz=int(olu.sum()), trafo_n=int(len(olu))))
    print(auc[-1])
R["01_auc"] = auc

# ---------- ESAS: MSE-KUTLE duyarliligi ----------
print("\n=== SATIR duyarliligi vs MSE-KUTLE duyarliligi ===")
t = []
for b in BLOKLAR:
    d, w, q = D[b], W[b], Q[b]["q"]
    p = d.p.values; z = d.tuketim.values <= 0
    tot_z_mse = float(np.sum(w[z] * p[z] ** 2))
    tot_mse = float(np.sum(w * d.r.values ** 2))
    for e in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95):
        f = q > e
        if f.sum() < 50: continue
        t.append(dict(blok=b, esik=e,
            bayrak_pay=round(float(np.sum(w * f) / w.sum()), 5),
            kesinlik=round(float(np.sum(w[f] * z[f]) / np.sum(w[f])), 4),
            satir_duyarlilik=round(float(np.sum(w[f & z]) / np.sum(w[z])), 4),
            MSE_kutle_duyarlilik=round(float(np.sum(w[f & z] * p[f & z] ** 2) / tot_z_mse), 4),
            yakalanan_MSE_toplam_pay=round(float(np.sum(w[f & z] * p[f & z] ** 2) / tot_mse), 4),
            ort_p_DP=round(float(np.average(p[f & z], weights=w[f & z])), 3),
            ort_p_kacan_sifir=round(float(np.average(p[~f & z], weights=w[~f & z])), 3),
            rms_p_kacan_sifir=round(float(np.sqrt(np.average(p[~f & z] ** 2, weights=w[~f & z]))), 3),
            ort_y_YP=round(float(np.average(d.y.values[f & ~z], weights=w[f & ~z])), 3),
            YP_MSE_riski=round(float(np.sum(w[f & ~z] * p[f & ~z] ** 2) / tot_mse), 4)))
    print(pd.DataFrame(t[-7:]).to_string(index=False))
R["02_kutle"] = t
with open(os.path.join(CIK, "p36_b.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=str)
with open(os.path.join(CIK, "p36_q.pkl"), "wb") as f:
    pickle.dump({b: Q[b] for b in BLOKLAR}, f)
print("\nyazildi p36_b.json + p36_q.pkl")
