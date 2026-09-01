"""En buyuk hata kaynagi = SIFIR/OLU satirlar. Duzeltmeyi kur ve OLC.

Butun parametreler guz25+kis26'dan; yaz25 hedefi yalnizca son olcumde.
"""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from p02_duzeltme import HEDEF_SOGUK, blok, skor

BURA = os.path.dirname(os.path.abspath(__file__))

OZ = [
    "t_sifir_orani",
    "t_kuyruk_sifir",
    "t_olu_mu",
    "t_son_kayit_yasi",
    "t_log_ort",
    "t_log_std",
    "t_log_son7",
    "t_log_son30",
    "t_log_son90",
    "t_gun_sayisi",
    "t_doluluk",
    "t_yayilma",
    "t_trend",
    "t_gy_sifir_orani",
    "yas",
    "guc",
    "soguk_mu",
    "ufuk_gun",
    "p_doluluk",
    "tk_hafta_sonu",
    "tatil_mi",
]


def main():
    yaz = blok("yaz25")
    dis = pd.concat([blok("guz25"), blok("kis26")], ignore_index=True)
    t0, t0w = skor(yaz, yaz.p.values)
    R = dict(taban=dict(rmsle=round(t0, 5), rmsle_test_bilesimi=round(t0w, 5)))

    z = (dis.tuketim.values <= 0).astype(int)
    m = lgb.train(
        dict(
            objective="binary",
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=200,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            verbose=-1,
            seed=7,
            num_threads=8,
        ),
        lgb.Dataset(dis[OZ], z),
        500,
    )
    qd, qy = m.predict(dis[OZ]), m.predict(yaz[OZ])
    zy = (yaz.tuketim.values <= 0).astype(int)
    try:
        from sklearn.metrics import roc_auc_score

        R["q_auc"] = dict(dis=round(roc_auc_score(z, qd), 4), yaz=round(roc_auc_score(zy, qy), 4))
    except Exception:
        R["q_auc"] = None
    R["q_kalibrasyon"] = dict(
        dis_ort_q=round(float(qd.mean()), 4),
        dis_gercek=round(float(z.mean()), 4),
        yaz_ort_q=round(float(qy.mean()), 4),
        yaz_gercek=round(float(zy.mean()), 4),
    )

    D = {}
    # S1: yapisal sifir-sisme duzeltmesi, serbest parametresiz
    D["S1_carpansal_(1-q)"] = (1.0 - qy) * yaz.p.values
    # S2: ussu, gamma dis blokta secilir
    gs = np.arange(0.1, 2.01, 0.1)
    sc = [float(((dis.y.values - (1 - qd) ** g * dis.p.values) ** 2).mean()) for g in gs]
    g_ = float(gs[int(np.argmin(sc))])
    D[f"S2_carpansal_gamma={g_:.1f}"] = (1 - qy) ** g_ * yaz.p.values
    # S3: r ~ q, q*p dogrusal (dis OLS)
    X = np.c_[np.ones(len(dis)), qd, qd * dis.p.values]
    c = np.linalg.lstsq(X, dis.r.values, rcond=None)[0]
    D["S3_dogrusal_q"] = yaz.p.values + c[0] + c[1] * qy + c[2] * qy * yaz.p.values
    # S4: S3 ama kuresel sabit atilir (blok seviyesi transfer etmiyor)
    D["S4_dogrusal_q_sabitsiz"] = yaz.p.values + c[1] * qy + c[2] * qy * yaz.p.values
    # S5: sert esik -- q > t ise sifira cek (t dis blokta secilir)
    ts = np.arange(0.3, 0.96, 0.05)
    sc = []
    for t in ts:
        pp = np.where(qd > t, 0.0, dis.p.values)
        sc.append(float(((dis.y.values - pp) ** 2).mean()))
    t_ = float(ts[int(np.argmin(sc))])
    D[f"S5_sert_esik_t={t_:.2f}"] = np.where(qy > t_, 0.0, yaz.p.values)
    # S6: artik-yigini -- dis blokta artigi ozelliklerden ogren, buzerek uygula
    mr = lgb.train(
        dict(
            objective="l2",
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=500,
            feature_fraction=0.8,
            verbose=-1,
            seed=7,
            num_threads=8,
        ),
        lgb.Dataset(dis[OZ], dis.r.values),
        400,
    )
    rd, ry = mr.predict(dis[OZ]), mr.predict(yaz[OZ])
    for lam in (0.25, 0.5, 1.0):
        D[f"S6_artik_yigin_lam={lam}"] = yaz.p.values + lam * (ry - ry.mean())

    R["duzeltmeler"] = {}
    for ad, pp in D.items():
        s, sw = skor(yaz, pp)
        R["duzeltmeler"][ad] = dict(
            rmsle=round(s, 5),
            kazanc=round(t0 - s, 5),
            rmsle_test_bilesimi=round(sw, 5),
            kazanc_test_bilesimi=round(t0w - sw, 5),
        )
        print(f"{ad:28s} RMSLE={s:.5f} kazanc={t0 - s:+.5f}  (agirlikli {sw:.5f} {t0w - sw:+.5f})")

    json.dump(
        R,
        open(os.path.join(BURA, "p04_sifir.json"), "w", encoding="utf-8"),
        indent=1,
        ensure_ascii=False,
    )
    print(json.dumps({k: R[k] for k in ("q_auc", "q_kalibrasyon")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
