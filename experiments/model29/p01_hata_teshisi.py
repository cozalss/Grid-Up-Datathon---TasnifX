"""yaz25 geri-testinde hatanin AYRISTIRILMASI.

Tahminler zaten uretilmis ve onbellekte:
  data/interim/aile_onbellek/yaz25_{tohum}_{algo}_uretim.npy   (SICAK satirlar)
  data/interim/deney/soguk_tahmin_yaz25.npz                    (SOGUK satirlar)
Birlestirme m148_demet_plani.py 95-113 ile BIREBIR aynidir.

SIZINTI: yaz25 hedefi yalnizca OLCUM icin okunur; hicbir tahmin/kalibrasyon
adiminda kullanilmaz.
"""

import json
import os

import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AO = os.path.join(KOK, "data/interim/aile_onbellek")
DN = os.path.join(KOK, "data/interim/deney")
BURA = os.path.dirname(os.path.abspath(__file__))


def yukle():
    e = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
    blk = e[e._blok == "yaz25"]
    sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
    P = [
        np.load(os.path.join(AO, f"yaz25_{t}_{aa}_uretim.npy")).astype(np.float64)
        for t in (1000, 1001, 1002)
        for aa in ("cat", "xgb", "lgbm")
        if os.path.exists(os.path.join(AO, f"yaz25_{t}_{aa}_uretim.npy"))
    ]
    z = np.load(os.path.join(DN, "soguk_tahmin_yaz25.npz"))
    idx = np.concatenate([sic.index.values, sog.index.values])
    pb = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])
    bf = e.loc[idx].copy()
    bf["p"] = pb
    bf["y"] = np.log1p(bf.tuketim.values.astype(np.float64))
    bf["r"] = bf.y - bf.p
    bf["ay"] = pd.to_datetime(bf.tarih).dt.month
    return bf, len(P), list(z.files)


def rmsle(r):
    return float(np.sqrt(np.mean(np.asarray(r) ** 2)))


def kesit(bf, kol, etiket=None):
    """Bir kesit degiskeninin her seviyesi icin pay/yanlilik/rmsle."""
    tk = float((bf.r**2).sum())
    g = bf.groupby(kol, observed=True)
    out = []
    for k, d in g:
        s = float((d.r**2).sum())
        out.append(
            dict(
                seviye=str(k),
                n=int(len(d)),
                n_pay=round(len(d) / len(bf), 4),
                kare_pay=round(s / tk, 4),
                yanlilik=round(float(d.r.mean()), 4),
                rmsle=round(rmsle(d.r), 4),
                # yanliligin kare hataya katkisi (bias^2 / toplam ort kare)
                bias2_pay=round(len(d) * float(d.r.mean()) ** 2 / tk, 4),
            )
        )
    out.sort(key=lambda x: -x["kare_pay"])
    return dict(kol=etiket or kol, seviyeler=out)


def kova(bf, kol, kenarlar, etiket):
    b = bf.copy()
    b["_k"] = pd.cut(b[kol], kenarlar, include_lowest=True)
    r = kesit(b, "_k", etiket)
    r["seviyeler"].sort(key=lambda x: x["seviye"])
    return r


def main():
    bf, nmodel, sogfiles = yukle()
    R = dict(
        meta=dict(
            n=int(len(bf)),
            n_trafo=int(bf.tanim.nunique()),
            sicak_model_sayisi=nmodel,
            soguk_model_sayisi=len(sogfiles),
            tarih=[str(bf.tarih.min().date()), str(bf.tarih.max().date())],
        )
    )
    R["taban"] = dict(
        rmsle=round(rmsle(bf.r), 5),
        yanlilik=round(float(bf.r.mean()), 5),
        std=round(float(bf.r.std()), 5),
        # yanliligin toplam kare hatadaki payi
        bias2_pay=round(float(bf.r.mean()) ** 2 / float((bf.r**2).mean()), 4),
    )

    # --- 1. SOGUK / SICAK
    R["soguk"] = kesit(bf, "soguk_mu")

    # --- 2. AY
    R["ay"] = kesit(bf, "ay")
    R["ay"]["seviyeler"].sort(key=lambda x: int(x["seviye"]))

    # --- 3. UFUK (tahmin ufkunun kacinci gunu)
    R["ufuk"] = kova(bf, "ufuk_gun", [0, 15, 30, 45, 60, 75, 90, 105, 122], "ufuk_gun")

    # --- 4. TUKETIM BUYUKLUGU (gercek log seviye)
    R["seviye"] = kova(bf, "y", [-0.01, 0.7, 2, 4, 6, 7, 8, 9, 20], "log1p(tuketim)")

    # --- 5. TAHMIN SEVIYESI (uygulanabilir kesit -- hedef kullanmaz)
    R["tahmin_seviyesi"] = kova(bf, "p", [-5, 0.7, 2, 4, 6, 7, 8, 9, 20], "tahmin log1p")

    # --- 6. SIFIR / COK DUSUK TUKETIM
    bf["sifir_sinif"] = np.where(
        bf.tuketim <= 0, "sifir", np.where(bf.tuketim < 1, "0-1", np.where(bf.tuketim < 10, "1-10", "10+"))
    )
    R["sifir"] = kesit(bf, "sifir_sinif")

    # --- 7. TRAFO YASI
    R["yas"] = kova(bf, "yas", [-1, 30, 90, 180, 365, 500], "yas_gun")

    # --- 8. TRAFO BAZINDA YOGUNLASMA
    g = bf.groupby("tanim", observed=True).r.agg(n="size", ss=lambda x: float((x**2).sum()), ort="mean")
    g = g.sort_values("ss", ascending=False)
    tk = g.ss.sum()
    cum = g.ss.cumsum() / tk
    R["trafo"] = dict(
        n_trafo=int(len(g)),
        # kare hatanin %50'sini kac trafo olusturuyor
        yuzde50_trafo=int((cum <= 0.5).sum() + 1),
        yuzde50_trafo_pay=round((int((cum <= 0.5).sum() + 1)) / len(g), 4),
        yuzde80_trafo=int((cum <= 0.8).sum() + 1),
        en_kotu20=[
            dict(tanim=str(i), n=int(r.n), kare_pay=round(r.ss / tk, 5), yanlilik=round(r.ort, 3))
            for i, r in g.head(20).iterrows()
        ],
    )
    # trafo yanliligi sistematik mi: trafo ici ortalama artigin kare payi
    tra_bias2 = float((g.n * g.ort**2).sum())
    R["trafo"]["sabit_kayma_payi"] = round(tra_bias2 / tk, 4)

    # --- 9. SISTEMATIK MI: gun-ici ortak sok (tarih ortalamasi) payi
    d = bf.groupby("tarih", observed=True).r.agg(n="size", ort="mean")
    R["gun_ortak_sok_payi"] = round(float((d.n * d.ort**2).sum()) / tk, 4)

    with open(os.path.join(BURA, "p01_hata_teshisi.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, ensure_ascii=False)
    print(json.dumps(R, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
