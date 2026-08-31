"""p06: DOGAL SOGUK uzmani -- uretim soguk modelinin YAPISAL alternatifi.

GEREKCE. Uretimin soguk uzmani (scripts/uret_soguk_tahmin.py) YAPAY olarak
sogutulmus satirlarla egitiliyor: maske orani 1,00 ile TUM trafolarin t_*
kolonlari NaN yapiliyor. Yani model "gecmisi silinmis OLGUN trafo"nun
seviyesini ogreniyor. Testteki soguk trafolar ise GERCEKTEN YENI
(host: "yeni devreye alinan trafolari temsil etmektedir"). Bu iki dagilim
ayni degil -- yaz25 soguk satirlarinda yanlilik +0,133, kis26'da +0,327:
model soguk trafolari SISTEMATIK olarak dusuk tahmin ediyor.

DENEY. Dis bloklarin (guz25+kis26) DOGAL soguk satirlarinda egitilmis bir
uzman, yaz25 soguk satirlarinda uretim tahmininden iyi mi? Harmani iyi mi?
Ayrica p03'un KOMSU SEVIYESI ozelligi burada ayri bir eksen olarak olculur.

SIZINTI KONTROLU
  - Egitim etiketi yalnizca guz25+kis26; yaz25 hedefi hicbir yerde girmez.
  - Trafo kumeleri AYRIK (assert ile denetleniyor).
  - Agac sayisi ve harman agirligi DIS bloklarin ic bolunmesinden secilir
    (guz25 -> kis26); yaz25 skoru yalnizca RAPORLANIR.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p02_duzeltme import DN, blok  # noqa: E402
from p06_soguk_tani import hazirla  # noqa: E402

E = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
ATLA = {"tanim", "tarih", "tuketim", "lokasyon", "_blok", "p", "y", "r", "ay", "soguk_mu"}
KOL = [c for c in E.columns if c not in ATLA and not c.startswith("t_") and E[c].dtype.kind in "ifbu"]


def rmsle(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def main():
    import lightgbm as lgb

    yaz = hazirla("yaz25")
    guz = hazirla("guz25")
    kis = hazirla("kis26")
    ys = yaz[yaz.soguk_mu == 1]
    gs = guz[guz.soguk_mu == 1]
    ks = kis[kis.soguk_mu == 1]
    assert not (set(ys.tanim) & (set(gs.tanim) | set(ks.tanim))), "TRAFO SIZINTISI"

    def X(d, komsu):
        x = E.loc[d.index, KOL].copy()
        if komsu:
            x["k_komsu"] = d.capa.values
        return x

    def y_ofs(d):
        return d.y.values - np.log1p(d.guc.values.astype("float64"))

    PK = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, num_threads=8, verbose=-1)
    TOHUM = (7, 17, 27)

    def egit(dtr, dte, tur, komsu):
        Xtr, Xte = X(dtr, komsu), X(dte, komsu)
        ps = [lgb.train(dict(PK, seed=s), lgb.Dataset(Xtr, y_ofs(dtr)),
                        num_boost_round=tur).predict(Xte) for s in TOHUM]
        return np.mean(ps, axis=0) + np.log1p(dte.guc.values.astype("float64"))

    R = dict(taban=dict(yaz25_soguk=round(rmsle(ys.r), 5),
                        guz25_soguk=round(rmsle(gs.r), 5),
                        kis26_soguk=round(rmsle(ks.r), 5)),
             n=dict(guz=int(len(gs)), kis=int(len(ks)), yaz=int(len(ys))),
             ozellik=len(KOL))

    # --- ic secim: guz25'te egit, kis26'da olc (agac sayisi + komsu ekseni + harman)
    ic = {}
    en = None
    for komsu in (False, True):
        for tur in (200, 400, 800):
            pk = egit(gs, ks, tur, komsu)
            v = rmsle(ks.y.values - pk)
            ic[f"komsu={komsu}_tur={tur}"] = round(v, 5)
            if en is None or v < en[0]:
                en = (v, tur, komsu, pk)
    print("ic secim (guz25 -> kis26 soguk RMSLE):", json.dumps(ic, indent=1), flush=True)
    _, TUR, KOMSU, pk_kis = en
    R["ic_secim"] = ic
    R["secilen"] = dict(agac=TUR, komsu=KOMSU)

    # harman agirligi da IC bolunmeden
    ww = np.linspace(0, 1, 21)
    sk = [rmsle(ks.y.values - ((1 - w) * ks.p.values + w * pk_kis)) for w in ww]
    W = float(ww[int(np.argmin(sk))])
    R["harman_agirligi_ic"] = dict(w=round(W, 3), kis26_taban=round(rmsle(ks.r), 5),
                                   kis26_harman=round(min(sk), 5),
                                   kis26_yalniz_dogal=round(rmsle(ks.y.values - pk_kis), 5))
    print(json.dumps(R["harman_agirligi_ic"], indent=1), flush=True)

    # --- UYGULA: guz25+kis26 dogal soguk -> yaz25 soguk
    dtr = pd.concat([gs, ks])
    py = egit(dtr, ys, TUR, KOMSU)
    R["yaz25"] = dict(
        taban=round(rmsle(ys.r), 5),
        yalniz_dogal=round(rmsle(ys.y.values - py), 5),
        harman_w_ic=round(rmsle(ys.y.values - ((1 - W) * ys.p.values + W * py)), 5),
    )
    for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        R["yaz25"][f"harman_w={w}"] = round(rmsle(ys.y.values - ((1 - w) * ys.p.values + w * py)), 5)
    print(json.dumps(R["yaz25"], indent=1))

    np.save(os.path.join(BURA, "p06_dogal_yaz25_soguk.npy"), py)
    with open(os.path.join(BURA, "p06_soguk_dogal.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
