"""p06: SOGUGA OZEL SIFIR SINIFLANDIRICISI.

NEDEN. p05_sifir_tavani.json: uretimdeki sifir siniflandiricisi (P0) esik
0,80'de soguk RMSLE'yi 1,43592'den HIC oynatmiyor -- yani soguk satirlarda
ANMA'si sifir. Sebep acik: P0 t_* (trafo gecmisi) kolonlarina dayaniyor,
soguk satirlarda hepsi NaN.

Ama p06_soguk_anatomi.json: SOGUK satirlarda sifirlar satirlarin %4'u,
KARE HATANIN %57,3'u. Kahin: soguk 1,43592 -> 0,93874.

Bu betik soguga OZEL bir q modeli egitir: yalnizca soguk satirlarda mevcut
olan oznitelikler (t_* HARIC), guz25+kis26'nin SOGUK satirlarinda egitilir,
yaz25'in soguk satirlarina uygulanir.

SIZINTI KONTROLU
  - Egitim etiketi guz25+kis26 hedefi; yaz25 hedefi hicbir yerde girmiyor.
  - Trafo kumeleri AYRIK: yaz25-soguk bir trafonun Nis-Tem 2025'te satiri
    vardir, dolayisiyla guz25/kis26'da SOGUK olamaz (ayrica denetleniyor).
  - Butun ayar (agac sayisi, gamma) dis bloklarin kendi ic bolunmesinden
    ya da sabit secilir; yaz25 skoruna gore secim YAPILMAZ. Yine de her
    gamma icin yaz25 skoru RAPORLANIR ki ayrisma gorulsun.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p02_duzeltme import DN, blok, skor  # noqa: E402

E = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
ATLA = {"tanim", "tarih", "tuketim", "lokasyon", "_blok", "p", "y", "r", "ay", "soguk_mu"}


def kolonlar():
    ok = []
    for c in E.columns:
        if c in ATLA or c.startswith("t_"):
            continue
        if E[c].dtype.kind in "ifbu":
            ok.append(c)
    return ok


def rmsle(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def auc(y, s):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n1 = float(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    import lightgbm as lgb

    KOL = kolonlar()
    yaz = blok("yaz25")
    dis = pd.concat([blok("guz25"), blok("kis26")], ignore_index=False)
    ys = yaz[yaz.soguk_mu == 1]
    ds = dis[dis.soguk_mu == 1]
    ort = set(ys.tanim) & set(ds.tanim)
    assert not ort, f"TRAFO SIZINTISI: {len(ort)} ortak trafo"
    print(f"soguk egitim {len(ds):,} satir / {ds.tanim.nunique():,} trafo   "
          f"yaz25 {len(ys):,} / {ys.tanim.nunique():,}   ozellik {len(KOL)}")

    Xd = E.loc[ds.index, KOL]
    Xy = E.loc[ys.index, KOL]
    zd = (ds.tuketim.values <= 0).astype(int)
    zy = (ys.tuketim.values <= 0).astype(int)

    PK = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
              num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              num_threads=8, verbose=-1)
    # agac sayisi: dis bloklarin KENDI ic bolunmesinden secilir (guz25 -> kis26)
    g_i = ds._blok.values == "guz25" if "_blok" in ds.columns else None
    gm = ds.index.isin(E.index[E._blok == "guz25"])
    ic = {}
    for tur in (150, 300, 600):
        m = lgb.train(dict(PK, seed=7), lgb.Dataset(Xd[gm], zd[gm]), num_boost_round=tur)
        ic[tur] = auc(zd[~gm], m.predict(Xd[~gm]))
    TUR = max(ic, key=ic.get)
    print("ic secim (guz25->kis26 AUC):", {k: round(v, 4) for k, v in ic.items()}, "->", TUR)

    q = np.mean([lgb.train(dict(PK, seed=s), lgb.Dataset(Xd, zd),
                           num_boost_round=TUR).predict(Xy) for s in (7, 17, 27)], axis=0)
    R = dict(
        n_egitim=int(len(ds)), n_yaz=int(len(ys)), ozellik=len(KOL),
        agac=TUR, ic_secim={str(k): round(v, 4) for k, v in ic.items()},
        auc_yaz25_soguk=round(auc(zy, q), 4),
        q_ort=round(float(q.mean()), 4), gercek_oran=round(float(zy.mean()), 4),
        taban_soguk=round(rmsle(ys.r), 5),
    )
    print(json.dumps(R, indent=1))

    # kalibrasyon tablosu
    ken = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
    kv = np.digitize(q, ken)
    R["kalibrasyon"] = [
        dict(kova=int(k), n=int((kv == k).sum()),
             q_ort=round(float(q[kv == k].mean()), 4),
             gercek=round(float(zy[kv == k].mean()), 4),
             p_ort=round(float(ys.p.values[kv == k].mean()), 3),
             kare_pay=round(float((ys.r.values[kv == k] ** 2).sum() / (ys.r.values**2).sum()), 4))
        for k in sorted(set(kv))
    ]
    print(pd.DataFrame(R["kalibrasyon"]).to_string(index=False))

    # --- DUZELTME: p' = (1-q)^gamma * p  (gamma sabit adaylar)
    pb = ys.p.values
    yv = ys.y.values
    R["duzeltme"] = {}
    for gmm in (0.25, 0.5, 0.75, 1.0, 1.5):
        p2 = pb * (1 - q) ** gmm
        R["duzeltme"][f"carpansal_g={gmm}"] = round(rmsle(yv - p2), 5)
    # dis blokta EN IYI gamma (sizintisiz secim)
    qd = np.mean([lgb.train(dict(PK, seed=s), lgb.Dataset(Xd[gm], zd[gm]),
                            num_boost_round=TUR).predict(Xd[~gm]) for s in (7, 17, 27)], axis=0)
    pdz, ydz = ds.p.values[~gm], ds.y.values[~gm]
    gg = np.linspace(0, 3, 61)
    sk = [rmsle(ydz - pdz * (1 - qd) ** g) for g in gg]
    g_sec = float(gg[int(np.argmin(sk))])
    R["gamma_dis_secim"] = dict(gamma=round(g_sec, 3),
                                dis_taban=round(rmsle(ydz - pdz), 5),
                                dis_duzeltilmis=round(min(sk), 5))
    p2 = pb * (1 - q) ** g_sec
    R["SECILEN"] = dict(gamma=round(g_sec, 3), yaz25_soguk_rmsle=round(rmsle(yv - p2), 5))
    print(json.dumps({k: R[k] for k in ("duzeltme", "gamma_dis_secim", "SECILEN")}, indent=1))

    np.save(os.path.join(BURA, "p06_q_yaz25_soguk.npy"), q)
    with open(os.path.join(BURA, "p06_soguk_q.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
