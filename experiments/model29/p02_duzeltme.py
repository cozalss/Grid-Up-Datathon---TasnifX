"""yaz25 geri-testinde DUZELTME denemeleri.

SIZINTI KURALI: her duzeltmenin TUM parametreleri yalnizca guz25+kis26
artiklarindan kestirilir. yaz25 hedefi sadece nihai RMSLE'yi OLCMEK icin
okunur. Egitim verisi olarak da yaz25 hicbir yerde kullanilmaz.
"""

import json
import os

import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AO = os.path.join(KOK, "data/interim/aile_onbellek")
DN = os.path.join(KOK, "data/interim/deney")
BURA = os.path.dirname(os.path.abspath(__file__))
HEDEF_SOGUK = 0.2216  # gercek test'teki soguk trafo payi

E = pd.read_parquet(os.path.join(DN, "egitim.parquet"))


def blok(bad):
    blk = E[E._blok == bad]
    sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
    P = [
        np.load(os.path.join(AO, f"{bad}_{t}_{a}_uretim.npy")).astype(np.float64)
        for t in (1000, 1001, 1002)
        for a in ("cat", "xgb", "lgbm")
        if os.path.exists(os.path.join(AO, f"{bad}_{t}_{a}_uretim.npy"))
    ]
    z = np.load(os.path.join(DN, f"soguk_tahmin_{bad}.npz"))
    idx = np.concatenate([sic.index.values, sog.index.values])
    p = np.concatenate([np.mean(P, axis=0), np.mean([z[k] for k in z.files], axis=0)])
    d = E.loc[idx].copy()
    d["p"] = p
    d["y"] = np.log1p(d.tuketim.values.astype(np.float64))
    d["r"] = d.y - d.p
    d["ay"] = pd.to_datetime(d.tarih).dt.month
    return d


def skor(d, p):
    r = d.y.values - p
    sg = d.soguk_mu.values.astype(np.float64)
    pay = sg.mean()
    w = np.where(sg == 1, HEDEF_SOGUK / pay, (1 - HEDEF_SOGUK) / (1 - pay))
    w = w / w.mean()
    return float(np.sqrt(np.mean(r * r))), float(np.sqrt(np.mean(w * r * r)))


def binle(x, kenarlar):
    return np.clip(np.digitize(x, kenarlar), 0, len(kenarlar))


def ofset_tablo(x_egt, r_egt, x_uyg, kenarlar, min_n=200):
    """Kesit ortalamasini disaridan kestirip uygula (bos kova -> 0)."""
    be, bu = binle(x_egt, kenarlar), binle(x_uyg, kenarlar)
    o = np.zeros(len(kenarlar) + 1)
    for k in range(len(o)):
        m = be == k
        if m.sum() >= min_n:
            o[k] = r_egt[m].mean()
    return o[bu], o


def main():
    yaz = blok("yaz25")
    dis = pd.concat([blok("guz25"), blok("kis26")], ignore_index=True)
    R = {}
    t0, t0w = skor(yaz, yaz.p.values)
    R["taban"] = dict(rmsle=round(t0, 5), rmsle_test_bilesimi=round(t0w, 5))
    print(f"TABAN yaz25 RMSLE={t0:.5f}  (test bilesimi agirlikli {t0w:.5f})")

    yol = {}

    # --- D1: kuresel ofset (dis bloklarin ortalama artigi)
    a = float(dis.r.mean())
    yol["D1_kuresel_ofset"] = (yaz.p.values + a, dict(a=round(a, 5)))

    # --- D2: ufuk gunune gore ofset tablosu
    KEN = [8, 15, 22, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    off, tbl = ofset_tablo(dis.ufuk_gun.values, dis.r.values, yaz.ufuk_gun.values, KEN)
    yol["D2_ufuk_ofset"] = (yaz.p.values + off, dict(tablo=[round(v, 4) for v in tbl]))

    # --- D3: tahmin seviyesine gore afin kalibrasyon (dis bloklarda OLS)
    pm = float(dis.p.mean())
    X = np.c_[np.ones(len(dis)), dis.p.values - pm]
    cf = np.linalg.lstsq(X, dis.r.values, rcond=None)[0]
    yol["D3_afin_seviye"] = (
        yaz.p.values + cf[0] + cf[1] * (yaz.p.values - pm),
        dict(alfa=round(float(cf[0]), 5), beta=round(float(cf[1]), 5), p_ort=round(pm, 4)),
    )

    # --- D4: SIFIR RISKI. Kesim aninda bilinen ozelliklerden sifir olasiligi
    #     kestirilir, artik ortalamasi o olasiligin kovalarinda disaridan olculur.
    import lightgbm as lgb

    OZ = [
        "t_sifir_orani",
        "t_kuyruk_sifir",
        "t_olu_mu",
        "t_son_kayit_yasi",
        "t_log_ort",
        "t_log_son7",
        "t_log_son30",
        "t_gun_sayisi",
        "t_doluluk",
        "t_yayilma",
        "yas",
        "guc",
        "soguk_mu",
        "ufuk_gun",
        "p_doluluk",
        "t_gy_sifir_orani",
    ]
    ztr = (dis.tuketim.values <= 0).astype(int)
    m = lgb.train(
        dict(
            objective="binary",
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=200,
            feature_fraction=0.8,
            verbose=-1,
            seed=7,
            num_threads=8,
        ),
        lgb.Dataset(dis[OZ], ztr),
        400,
    )
    q_dis = m.predict(dis[OZ])
    q_yaz = m.predict(yaz[OZ])
    QK = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95]
    offq, tblq = ofset_tablo(q_dis, dis.r.values, q_yaz, QK)
    yol["D4_sifir_riski"] = (
        yaz.p.values + offq,
        dict(
            tablo=[round(v, 4) for v in tblq],
            yaz_sifir_orani=round(float((yaz.tuketim <= 0).mean()), 4),
            dis_sifir_orani=round(float(ztr.mean()), 4),
            auc_yaz=None,
        ),
    )

    # --- D5: sifir riski + ufuk birlikte (once q, sonra kalan artikta ufuk)
    r2 = dis.r.values - ofset_tablo(q_dis, dis.r.values, q_dis, QK)[0]
    off2, tbl2 = ofset_tablo(dis.ufuk_gun.values, r2, yaz.ufuk_gun.values, KEN)
    yol["D5_sifir_arti_ufuk"] = (
        yaz.p.values + offq + off2,
        dict(ufuk_tablo=[round(v, 4) for v in tbl2]),
    )

    # --- D6: D5 uzerine afin seviye kalibrasyonu
    r3 = r2 - ofset_tablo(dis.ufuk_gun.values, r2, dis.ufuk_gun.values, KEN)[0]
    X3 = np.c_[np.ones(len(dis)), dis.p.values - pm]
    c3 = np.linalg.lstsq(X3, r3, rcond=None)[0]
    yol["D6_hepsi"] = (
        yaz.p.values + offq + off2 + c3[0] + c3[1] * (yaz.p.values - pm),
        dict(alfa=round(float(c3[0]), 5), beta=round(float(c3[1]), 5)),
    )

    R["duzeltmeler"] = {}
    for ad, (pp, par) in yol.items():
        s, sw = skor(yaz, pp)
        R["duzeltmeler"][ad] = dict(
            rmsle=round(s, 5),
            kazanc=round(t0 - s, 5),
            rmsle_test_bilesimi=round(sw, 5),
            kazanc_test_bilesimi=round(t0w - sw, 5),
            parametre=par,
        )
        print(
            f"{ad:22s} RMSLE={s:.5f}  kazanc={t0 - s:+.5f}   (agirlikli {sw:.5f} {t0w - sw:+.5f})"
        )

    # --- TAVAN olcumleri (duzeltilebilirligin ust siniri, SIZINTILI -- sadece teshis)
    tav = {}
    for ad, kol in [("trafo_sabiti", "tanim"), ("gun_sabiti", "tarih"), ("ay_sabiti", "ay")]:
        o = yaz.groupby(kol, observed=True).r.transform("mean")
        s, sw = skor(yaz, yaz.p.values + o.values)
        tav[ad] = dict(rmsle=round(s, 5), kazanc=round(t0 - s, 5))
    o = yaz.groupby(
        pd.cut(yaz.ufuk_gun, [0, 15, 30, 45, 60, 75, 90, 105, 122]), observed=True
    ).r.transform("mean")
    s, _ = skor(yaz, yaz.p.values + o.values)
    tav["ufuk_sabiti"] = dict(rmsle=round(s, 5), kazanc=round(t0 - s, 5))
    msk = yaz.tuketim.values <= 0
    pp = yaz.p.values.copy()
    pp[msk] = 0.0
    s, _ = skor(yaz, pp)
    tav["sifirlari_mukemmel_bil"] = dict(rmsle=round(s, 5), kazanc=round(t0 - s, 5))
    R["tavan_sizintili_teshis"] = tav
    print("\nTAVAN (sizintili, sadece teshis):", json.dumps(tav, ensure_ascii=False))

    with open(os.path.join(BURA, "p02_duzeltme.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
