"""p06: SOGUK satirlarin hata ANATOMISI + uygulanabilir capa arayisi.

Yalnizca TANI. yaz25 hedefi olcum icin okunur; hicbir uretim parametresi
buradan kestirilmez.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p02_duzeltme import DN, blok  # noqa: E402
from p06_soguk_tani import HAM, OZET, hazirla  # noqa: E402

KOK = os.path.dirname(os.path.dirname(BURA))


def rmsle(r):
    return float(np.sqrt(np.mean(np.asarray(r) ** 2)))


def kesit(d, kol):
    tk = float((d.r**2).sum())
    out = []
    for k, g in d.groupby(kol, observed=True):
        out.append(
            dict(
                seviye=str(k),
                n=int(len(g)),
                n_pay=round(len(g) / len(d), 4),
                kare_pay=round(float((g.r**2).sum()) / tk, 4),
                yanlilik=round(float(g.r.mean()), 4),
                rmsle=round(rmsle(g.r), 4),
            )
        )
    out.sort(key=lambda x: -x["kare_pay"])
    return out


def main():
    R = {}
    yaz = hazirla("yaz25")
    s = yaz[yaz.soguk_mu == 1].copy()
    R["taban_soguk"] = dict(n=int(len(s)), trafo=int(s.tanim.nunique()), rmsle=round(rmsle(s.r), 5))

    # --- 1. SIFIRLAR
    s["sinif"] = np.where(
        s.tuketim <= 0,
        "sifir",
        np.where(s.tuketim < 10, "0-10", np.where(s.tuketim < 100, "10-100", "100+")),
    )
    R["sifir_kesiti"] = kesit(s, "sinif")

    # --- 2. TAVANLAR (sizintili teshis)
    tav = {}
    tav["trafo_sabiti"] = round(
        rmsle(s.y - s.groupby("tanim", observed=True).y.transform("mean")), 5
    )
    p2 = s.p + s.groupby("tanim", observed=True).r.transform("mean")
    tav["trafo_ofseti_p_uzerine"] = round(rmsle(s.y - p2), 5)
    yy = s.y.values.copy()
    pp = s.p.values.copy()
    m0 = s.tuketim.values <= 0
    pp2 = pp.copy()
    pp2[m0] = 0.0
    tav["sifirlari_mukemmel_bil"] = round(rmsle(yy - pp2), 5)
    tav["kuresel_ofset"] = round(rmsle(s.r - s.r.mean()), 5)
    # p'nin en iyi afin donusumu
    X = np.c_[np.ones(len(s)), pp]
    tav["afin_p"] = round(rmsle(yy - X @ np.linalg.lstsq(X, yy, rcond=None)[0]), 5)
    R["tavanlar_sizintili"] = tav

    # --- 3. TRAFO SEVIYE SAPMASI: neye bagli?
    tb = (
        s.groupby("tanim", observed=True)
        .agg(
            b=("r", "mean"),
            n=("r", "size"),
            guc=("guc", "first"),
            capa=("capa", "first"),
            pm=("p", "mean"),
            ym=("y", "mean"),
            ilk=("p_ilk_ofset", "first"),
            gun=("p_gun_sayisi", "first"),
            yay=("p_yayilma", "first"),
            dol=("p_doluluk", "first"),
        )
        .dropna()
    )
    R["trafo_sapmasi"] = dict(
        n_trafo=int(len(tb)),
        std_b=round(float(tb.b.std()), 4),
        ort_b=round(float(tb.b.mean()), 4),
        korr={
            k: round(float(np.corrcoef(tb[k], tb.b)[0, 1]), 4)
            for k in ("guc", "capa", "pm", "ilk", "gun", "yay", "dol")
        },
    )

    # --- 4. Uretim modelinin zaten sahip oldugu grup ozellikleri ne kadar isliyor?
    E = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
    kol = [
        c
        for c in ("g_ilce_log_ort", "g_ilce_kova_ort", "g_kova_log_ort", "tanim_num")
        if c in E.columns
    ]
    sv = E.loc[s.index, kol]
    R["mevcut_grup_ozellikleri"] = {
        c: dict(
            korr_y=round(float(np.corrcoef(sv[c].fillna(sv[c].median()), s.y)[0, 1]), 4),
            korr_artik=round(float(np.corrcoef(sv[c].fillna(sv[c].median()), s.r)[0, 1]), 4),
        )
        for c in kol
    }

    # --- 5. ZAMAN icinde: bloklar arasi soguk yanlilik kararli mi?
    blk = {}
    for bad in ("yaz25", "guz25", "kis26"):
        d = hazirla(bad)
        ss = d[d.soguk_mu == 1]
        blk[bad] = dict(
            n=int(len(ss)),
            rmsle=round(rmsle(ss.r), 5),
            yanlilik=round(float(ss.r.mean()), 4),
            sifir_pay=round(float((ss.tuketim <= 0).mean()), 4),
            p_ort=round(float(ss.p.mean()), 4),
            y_ort=round(float(ss.y.mean()), 4),
            std_artik=round(float(ss.r.std()), 4),
        )
    R["bloklar"] = blk

    with open(os.path.join(BURA, "p06_soguk_anatomi.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    print(json.dumps(R, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
