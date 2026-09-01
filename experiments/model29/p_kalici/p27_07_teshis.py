"""p27-07: NEDEN AUC 0.97 iken buzme KAYBEDIYOR? Kesinlik/duyarlilik teshisi."""
import json
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p27_ortak import HEDEF_SOGUK, blok, rmsle  # noqa: E402

CIK = os.path.dirname(os.path.abspath(__file__))
R = {}
BLOKLAR = ("yaz25", "guz25", "kis26")
D = {b: blok(b) for b in BLOKLAR}
_v = pd.unique(pd.concat([D[b].ilce_key.astype(str) for b in BLOKLAR]))
_kod = {v: i for i, v in enumerate(_v)}
for b in BLOKLAR:
    D[b] = D[b].copy()
    D[b]["ilce_kod"] = D[b].ilce_key.astype(str).map(_kod).astype("float32")

T_KUL = ["guc", "t_sifir_orani", "t_kuyruk_sifir", "t_olu_mu", "t_son_kayit_yasi",
         "t_log_ort", "t_log_std", "t_log_medyan", "t_log_p10", "t_gun_sayisi",
         "t_doluluk", "t_yuk_faktoru", "t_trend", "t_log_son7", "t_log_son30",
         "t_log_son90", "t_son30_gun", "t_gy_sifir_orani", "t_gy_log_ort",
         "g_guc_kova", "g_ilce_log_ort", "g_kova_log_ort", "ilce_kod",
         "guc_yuzdelik", "ozet_pencere_gun", "p_doluluk", "p_son_ofset", "yas"]
PAR = dict(objective="binary", learning_rate=0.05, num_leaves=31, verbose=-1,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           min_data_in_leaf=20)


def bilesik(sic, sog):
    return float(np.sqrt(HEDEF_SOGUK * sog**2 + (1 - HEDEF_SOGUK) * sic**2))


def bil_of(d, p):
    sg = d.soguk_mu.values == 1
    return bilesik(rmsle(d.y.values[~sg], p[~sg]), rmsle(d.y.values[sg], p[sg]))


def tablo(b):
    d = D[b]
    z = (d.tuketim.values <= 0)
    g = pd.DataFrame(dict(t=d.tanim.values, z=z))
    y = g.groupby("t").z.mean()
    ilk = d.groupby("tanim")[T_KUL + ["soguk_mu"]].first()
    out = ilk.copy()
    out["hedef"] = (y.reindex(out.index) == 1.0).astype(int)
    out["n"] = g.groupby("t").size().reindex(out.index)
    out["p_ort"] = d.groupby("tanim").p.mean().reindex(out.index)
    out["y_ort"] = d.groupby("tanim").y.mean().reindex(out.index)
    return out


TT = {b: tablo(b) for b in BLOKLAR}

# --- olu trafolarin kimligi
kim = []
for b in BLOKLAR:
    t = TT[b]
    olu = t[t.hedef == 1]
    kim.append(dict(blok=b, n_olu=int(len(olu)),
                    olu_soguk_payi=round(float(olu.soguk_mu.mean()), 3),
                    tum_soguk_payi=round(float(t.soguk_mu.mean()), 3),
                    olu_p_ort=round(float(olu.p_ort.mean()), 3),
                    canli_p_ort=round(float(t[t.hedef == 0].p_ort.mean()), 3),
                    canli_y_ort=round(float(t[t.hedef == 0].y_ort.mean()), 3),
                    olu_satir=int(olu.n.sum()),
                    olu_MSE_payi=None))
R["01_olu_kimligi"] = kim
print("1) OLU TRAFO KIMLIGI (p_ort = modelin ortalama log tahmini):")
for x in kim:
    print("  ", x)

# --- esik merdiveni
mer = []
for hed in BLOKLAR:
    te = D[hed]
    taban = bil_of(te, te.p.values)
    trT = pd.concat([TT[b] for b in BLOKLAR if b != hed])
    teT = TT[hed]
    qs = [lgb.train(dict(PAR, seed=s),
                    lgb.Dataset(trT[T_KUL].astype("float32"), trT.hedef.values),
                    num_boost_round=250).predict(teT[T_KUL].astype("float32"))
          for s in (1000, 1001, 1002)]
    q = np.mean(qs, axis=0)
    qs_ = pd.Series(q, index=teT.index)
    for esik in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        sec = q > esik
        tp = int(((teT.hedef.values == 1) & sec).sum())
        fp = int(((teT.hedef.values == 0) & sec).sum())
        m = qs_.reindex(te.tanim.values).to_numpy() > esik
        p2 = te.p.values.copy(); p2[m] = 0.0
        # yalniz TP'ler sifirlansa (kahin ust sinir, ayni secim kumesinde)
        hf = TT[hed].hedef.reindex(te.tanim.values).to_numpy()
        p3 = te.p.values.copy(); p3[m & (hf == 1)] = 0.0
        p4 = te.p.values.copy(); p4[m & (hf == 0)] = 0.0
        mer.append(dict(blok=hed, esik=esik, TP=tp, FP=fp,
                        kesinlik=round(tp / max(1, tp + fp), 3),
                        duyarlilik=round(tp / max(1, int(teT.hedef.sum())), 3),
                        kazanc_hepsi=round(taban - bil_of(te, p2), 5),
                        kazanc_yalniz_TP=round(taban - bil_of(te, p3), 5),
                        zarar_yalniz_FP=round(taban - bil_of(te, p4), 5),
                        FP_ort_gercek_y=round(float(np.mean(
                            te.y.values[m & (hf == 0)])), 3) if (m & (hf == 0)).sum() else None))
R["02_esik_merdiveni"] = mer
print("\n2) ESIK MERDIVENI (kazanc_yalniz_TP = dogru yakalananlarin degeri;")
print("   zarar_yalniz_FP = yanlis yakalananlarin bedeli):")
print(f"{'blok':7}{'esik':>6}{'TP':>5}{'FP':>5}{'kesin':>7}{'duyar':>7}"
      f"{'kaz_hepsi':>11}{'kaz_TP':>10}{'zarar_FP':>10}{'FP_y_ort':>10}")
for x in mer:
    print(f"{x['blok']:7}{x['esik']:>6}{x['TP']:>5}{x['FP']:>5}{x['kesinlik']:>7.3f}"
          f"{x['duyarlilik']:>7.3f}{x['kazanc_hepsi']:>+11.5f}{x['kazanc_yalniz_TP']:>+10.5f}"
          f"{x['zarar_yalniz_FP']:>+10.5f}"
          f"{(x['FP_ort_gercek_y'] if x['FP_ort_gercek_y'] is not None else 0):>10.3f}")

with open(os.path.join(CIK, "p27_07.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=str)
print("\nyazildi p27_07.json")
