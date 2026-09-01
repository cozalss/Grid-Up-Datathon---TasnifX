"""p35-b: SIFIR YONU icin BIRIM YON BASINA rho -- esas olcum.

Cebir (p33/p34 cercevesi):  p_yeni = p + kappa*u,  u birim (agirlikli rms=1)
    MSE(kappa) = MSE0 - 2*kappa*rho + kappa^2,   rho = E_w[r*u]
Kazanc kappa*=rho'da maksimum ve MSE dususu = rho^2.

ONEMLI: onceki calisma (p27_06) KAZANC olcuyordu; tam sifirlama kappa'yi
cok buyutur ve rho>0 olsa bile zarar verir.  Burada SEKIL'i olcuyoruz,
olcegi degil.

Butun siniflandiricilar BLOK-DISI (hedef blok disindaki iki blokta egitilir).
Ozetler (t_*) hedef blogun BASINDAN ONCEKI pencereden gelir -- sizinti yok.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p27_ortak import agirlik, blok, rmsle  # noqa: E402

CIK = os.path.dirname(os.path.abspath(__file__))
BLOKLAR = ("yaz25", "guz25", "kis26")
R = {}

print("bloklar kuruluyor...")
D = {b: blok(b, soguk_harman="cat", son_islem=True) for b in BLOKLAR}
W = {b: agirlik(D[b]) for b in BLOKLAR}

# ortak ilce kodlamasi
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
T_KUL = [c for c in KUL if c not in ("ufuk_gun", "tatil_mi", "sicaklik_ort", "soguk_mu")]

PAR = dict(objective="binary", learning_rate=0.05, num_leaves=31, verbose=-1,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           min_data_in_leaf=20)


def trafo_tablo(b):
    d = D[b]
    z = (d.tuketim.values <= 0)
    g = pd.DataFrame(dict(t=d.tanim.values, z=z))
    y = g.groupby("t").z.mean()
    ilk = d.groupby("tanim")[T_KUL].first()
    out = ilk.copy()
    out["hedef"] = (y.reindex(out.index) == 1.0).astype(int)
    return out


TT = {b: trafo_tablo(b) for b in BLOKLAR}


# ---------------- rho cekirdegi ----------------
def rho_kappa(r, delta, w):
    """delta yonunun birim halinin rho'su ve tam-uygulama kappa'si."""
    k2 = float(np.sum(w * delta * delta) / np.sum(w))
    if k2 <= 0:
        return 0.0, 0.0
    kap = float(np.sqrt(k2))
    rd = float(np.sum(w * r * delta) / np.sum(w))
    return rd / kap, kap


SKOR = {}
auc_tab = []
for hed in BLOKLAR:
    te = D[hed]
    yte = (te.tuketim.values <= 0).astype(int)
    sg = te.soguk_mu.values == 1

    # (A) trafo duzeyi
    trT = pd.concat([TT[b] for b in BLOKLAR if b != hed])
    teT = TT[hed]
    qs = []
    for tohum in (1000, 1001, 1002):
        m = lgb.train(dict(PAR, seed=tohum),
                      lgb.Dataset(trT[T_KUL].astype("float32"), trT.hedef.values),
                      num_boost_round=250)
        qs.append(m.predict(teT[T_KUL].astype("float32")))
    qT = np.mean(qs, axis=0)
    q_row_T = pd.Series(qT, index=teT.index).reindex(te.tanim.values).to_numpy()

    # (B) satir duzeyi
    tr = pd.concat([D[b] for b in BLOKLAR if b != hed])
    mr = lgb.train(dict(PAR, num_leaves=63, seed=1000),
                   lgb.Dataset(tr[KUL].astype("float32"),
                               (tr.tuketim.values <= 0).astype(int)),
                   num_boost_round=300)
    q_row = mr.predict(te[KUL].astype("float32"))

    # (C) satir duzeyi, p^2 AGIRLIKLI (hatanin oldugu yere odaklan)
    wtr = np.concatenate([D[b].p.values ** 2 + 0.25 for b in BLOKLAR if b != hed])
    mw = lgb.train(dict(PAR, num_leaves=63, seed=1000),
                   lgb.Dataset(tr[KUL].astype("float32"),
                               (tr.tuketim.values <= 0).astype(int), weight=wtr),
                   num_boost_round=300)
    q_w = mw.predict(te[KUL].astype("float32"))

    SKOR[hed] = dict(qT=q_row_T, qrow=q_row, qw=q_w)
    auc_tab.append(dict(
        blok=hed,
        trafo_AUC=round(float(roc_auc_score(teT.hedef.values, qT)), 4),
        trafo_poz=int(teT.hedef.sum()), trafo_n=int(len(teT)),
        satir_AUC_qT=round(float(roc_auc_score(yte, q_row_T)), 4),
        satir_AUC_qrow=round(float(roc_auc_score(yte, q_row)), 4),
        satir_AUC_qw=round(float(roc_auc_score(yte, q_w)), 4),
        satir_AUC_qrow_sicak=round(float(roc_auc_score(yte[~sg], q_row[~sg])), 4),
        satir_AUC_qrow_soguk=round(float(roc_auc_score(yte[sg], q_row[sg])), 4),
    ))
    print(auc_tab[-1])
R["01_auc"] = auc_tab

# ---------------- yon ailesi ----------------
def yonlar(b):
    """ad -> delta (tahmine EKLENECEK yon; sifirlama icin negatif)."""
    d = D[b]
    p = d.p.values
    z = (d.tuketim.values <= 0)
    ks = np.nan_to_num(d.t_kuyruk_sifir.values.astype(np.float64), nan=-1.0)
    out = {}
    out["KAHIN_tum_sifir"] = -p * z
    out["KAHIN_gercek_artik"] = d.r.values.copy()  # mutlak tavan (rho=rms r)
    for e in (1, 3, 7, 14, 30):
        out[f"kuyruk>={e}"] = -p * (ks >= e)
    out["t_olu_mu"] = -p * (np.nan_to_num(d.t_olu_mu.values, nan=0.0) > 0.5)
    for ad in ("qT", "qrow", "qw"):
        q = SKOR[b][ad]
        for a in (0.5, 1.0, 2.0, 3.0):
            out[f"{ad}_p*q^{a}"] = -p * q ** a
        for e in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
            out[f"{ad}_p*1[q>{e}]"] = -p * (q > e)
        for e in (0.2, 0.5):
            out[f"{ad}_p*q*1[q>{e}]"] = -p * q * (q > e)
        # p'siz (duz sifir yonu): sadece bayrakli satirlarda sabit dusus
        out[f"{ad}_1[q>0.5]"] = -1.0 * (q > 0.5)
    # birlesik: max(qT, qrow)
    qm = np.maximum(SKOR[b]["qT"], SKOR[b]["qrow"])
    out["qmax_p*q"] = -d.p.values * qm
    out["qmax_p*q^2"] = -d.p.values * qm ** 2
    for e in (0.3, 0.5, 0.7):
        out[f"qmax_p*1[q>{e}]"] = -d.p.values * (qm > e)
    return out


sat = []
for b in BLOKLAR:
    d = D[b]
    w = W[b]
    r = d.r.values
    rms_r = float(np.sqrt(np.sum(w * r * r) / np.sum(w)))
    for ad, dl in yonlar(b).items():
        rho, kap = rho_kappa(r, dl, w)
        sat.append(dict(blok=b, yon=ad, rho=round(rho, 5), kappa_tam=round(kap, 4),
                        korelasyon=round(rho / rms_r, 5)))
    print(f"{b}: rms_r={rms_r:.4f}")

T = pd.DataFrame(sat)
P = T.pivot(index="yon", columns="blok", values="rho")[list(BLOKLAR)]
P["ORT"] = P.mean(axis=1)
P["MIN"] = P.min(axis=1)
P = P.sort_values("ORT", ascending=False)
print("\n=== rho (birim yon basina), blok basi ===")
print(P.to_string(float_format=lambda x: f"{x:.5f}"))
R["02_rho"] = json.loads(P.to_json())
K = T.pivot(index="yon", columns="blok", values="kappa_tam")[list(BLOKLAR)]
R["03_kappa_tam"] = json.loads(K.to_json())

# ---------------- DURUST BLOK-DISI SECIM ----------------
# aday sekiller = KAHIN olmayan hepsi; esik/sekil DIGER iki bloktan secilir
adaylar = [x for x in P.index if not x.startswith("KAHIN")]
durust = []
for hed in BLOKLAR:
    dis = [b for b in BLOKLAR if b != hed]
    skor = {a: np.mean([P.loc[a, b] for b in dis]) for a in adaylar}
    en = max(skor, key=skor.get)
    durust.append(dict(hedef=hed, secilen=en,
                       digerlerinde_rho=round(float(skor[en]), 5),
                       hedefte_rho=round(float(P.loc[en, hed]), 5),
                       hedefte_kappa_tam=round(float(K.loc[en, hed]), 4)))
    print("DURUST", durust[-1])
R["04_durust"] = durust
rho_durust = float(np.mean([x["hedefte_rho"] for x in durust]))
R["05_durust_ort_rho"] = round(rho_durust, 5)
print(f"\nDURUST blok-disi ORT rho = {rho_durust:.5f}")

# ---------------- kazanca cevir ----------------
MSE0 = float(np.mean([rmsle(D[b].y.values, D[b].p.values, W[b]) ** 2 for b in BLOKLAR]))
cv0 = float(np.sqrt(MSE0))
OLC = 0.93907
kaz = []
for ad, rho in [("durust", rho_durust)] + [(f"en_iyi_{a}", float(P.loc[a, "ORT"]))
                                           for a in list(P.index)[:6]]:
    yeni = float(np.sqrt(max(MSE0 - rho ** 2, 1e-9)))
    kaz.append(dict(ad=ad, rho=round(rho, 5), kappa_opt=round(rho, 5),
                    cv=round(cv0, 5), cv_yeni=round(yeni, 5),
                    cv_kazanc=round(cv0 - yeni, 5),
                    lb_kazanc=round((cv0 - yeni) * OLC, 5),
                    lb_yeni=round(1.00115 - (cv0 - yeni) * OLC, 5)))
    print(kaz[-1])
R["06_kazanc"] = kaz
R["07_gereken"] = dict(rho_gerekli_2sira=0.1144, cv_rms_r=round(cv0, 5))

with open(os.path.join(CIK, "p35_b.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=str)
print("\nyazildi p35_b.json")
