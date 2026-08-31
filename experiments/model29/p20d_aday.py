"""p20d: kazancin SEVIYE / YAPI ayrismasi + TEST adayinin uretilmesi.

NEDEN AYRISMA
-------------
Uretim son islemi (son_islem.py) buzmeyi tahminin KENDI ortalamasina yapar,
yani harman degisince SEVIYE de degisir. Gonderim dosyasi ise uzerine
LB'den cozulmus seviye duzeltmeleri (m111 kappa, span a0+r_hat) tasiyor.
Seviye kazanci CIFT SAYILABILIR. Bu yuzden kazanc ikiye ayrilir:

    SEVIYE : yalnizca soguk satirlarin ortalama ofsetinin kaymasi
    YAPI   : ortalama sabit tutuldugunda kalan kesitsel/zamansal degisim

TEST ADAYI
----------
p_kalici/aday_csv/p06_test_soguk_aile.npy -- test soguk satirlari icin
(cat, xgb, lgbm) log tahminleri, tohum 1000-1002 ortalamasi, URETIM soguk
ayariyla (maske 1.00, cat depth 7). Yani EGITIM GEREKMIYOR.

Gonderim dosyasinin soguk satirlari ham model ciktisi DEGIL: uzerinde son
islem (buzme) ve gonderim-uzayi cebiri var. Bu zincir soguk satirlarda
AFFINE'dir ve egimi dosyadan OLCULEBILIR (p18_harman_hukmu.py yontemi,
v27/v30 uzerinde kalibre edildi). Aday su sekilde kurulur:

    log1p(yeni) = log1p(eski) + s * (r_harman - r_cat)          [V2 seviyeli]
    log1p(yeni) = log1p(eski) + s * (r_harman - r_cat - ort)    [V1 seviyesiz]

s = dosyanin kendi soguk satirlarindan olculen egim (kova merkezli).
V1 SEVIYEYI KORUR -- LB'den cozulmus seviye katmaniyla cift saymaz.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))
sys.path.insert(0, BURA)

from p20_harman import (  # noqa: E402
    AILE,
    BLOKLAR,
    HARMANLAR,
    KVA_KENAR,
    SOGUK_PAY,
    TOHUMLAR,
    olcutler,
    veri_kur,
)

BETA = 0.60
TABAN = "URETIM_cat"
ADAYLAR = ("ESIT", "ESKI_3_1_1")
CIK = (
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad"
)
KAYNAK = "tuketim_YP_seviye.csv"


def buz(r, beta):
    o = float(r.mean())
    return o + beta * (r - o)


def main() -> None:
    B, _ = veri_kur()
    R: dict = {}

    # ---------- 1) SEVIYE / YAPI AYRISMASI (blok CV'sinde)
    ayr: dict = {}
    for ad in ADAYLAR:
        ayr[ad] = {}
        for b in BLOKLAR:
            bb = B[b]
            p_t = sum(w * bb["P"][a].mean(axis=0) for w, a in zip(HARMANLAR[TABAN], AILE))
            p_a = sum(w * bb["P"][a].mean(axis=0) for w, a in zip(HARMANLAR[ad], AILE))
            r_t, r_a = p_t - bb["lguc"], p_a - bb["lguc"]
            # seviyesi TABANA esitlenmis aday
            r_a0 = r_a - r_a.mean() + r_t.mean()
            e = {}
            for etiket, r in (("taban", r_t), ("aday_tam", r_a), ("aday_seviyesiz", r_a0)):
                rr = buz(r, BETA)
                e[etiket] = olcutler(bb, bb["lgy"] - np.maximum(rr + bb["lguc"], 0.0))["agr"]
            ayr[ad][b] = {
                "dMSE_TOPLAM": round(e["taban"] - e["aday_tam"], 6),
                "dMSE_YAPI(seviyesiz)": round(e["taban"] - e["aday_seviyesiz"], 6),
                "dMSE_SEVIYE(fark)": round(e["aday_seviyesiz"] - e["aday_tam"], 6),
                "ort_ofset_taban": round(float(r_t.mean()), 5),
                "ort_ofset_aday": round(float(r_a.mean()), 5),
            }
        for k in ("dMSE_TOPLAM", "dMSE_YAPI(seviyesiz)", "dMSE_SEVIYE(fark)"):
            v = [ayr[ad][b][k] for b in BLOKLAR]
            ayr[ad][f"ORT_{k}"] = round(float(np.mean(v)), 6)
            ayr[ad][f"isaret_{k}"] = f"{sum(x > 0 for x in v)}/3"
    R["seviye_yapi_ayrismasi"] = ayr

    # ---------- 2) TEST ADAYI
    A = np.load(os.path.join(BURA, "p_kalici", "aday_csv", "p06_test_soguk_aile.npy")).astype(
        "float64"
    )
    T = pd.read_parquet(os.path.join(KOK, "data/interim/deney/test.parquet"))
    soguk = (T["soguk_mu"] == 1).to_numpy()
    ids_par = T["id"].to_numpy()
    guc_s = T["guc"].to_numpy("float64")[soguk]
    ham = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"), usecols=["id"])
    IDS = ham["id"].to_numpy()
    pos = pd.Index(IDS).get_indexer(ids_par)  # parquet -> csv sirasi
    assert (pos >= 0).all() and len(set(pos)) == len(pos)

    sub = pd.read_csv(os.path.join(KOK, "submissions", KAYNAK))
    assert np.array_equal(sub["id"].to_numpy(), IDS), "kaynak id sirasi test.csv ile ayni degil"
    L = np.log1p(sub["tuketim"].to_numpy("float64"))
    L_s = L[pos][soguk]

    # egim: kova merkezli regresyon (p18_harman_hukmu yontemi)
    kova = np.digitize(guc_s, KVA_KENAR) - 1

    def merkez(v):
        s = pd.Series(v)
        return (s - s.groupby(kova).transform("mean")).to_numpy()

    X = np.c_[[merkez(A[:, i]) for i in range(3)]].T
    bc, *_ = np.linalg.lstsq(X, merkez(L_s), rcond=None)
    s_egim = float(bc.sum())
    R["kaynak_dosya"] = {
        "dosya": KAYNAK,
        "LB": 1.00115,
        "olculen_egim_s": round(s_egim, 4),
        "aile_paylari": [round(float(x / s_egim), 4) for x in bc],
    }

    lguc_s = np.log1p(guc_s)
    r_cat = A[:, 0] - lguc_s
    uretilen = []
    for ad in ADAYLAR:
        r_a = A @ np.array(HARMANLAR[ad]) - lguc_s
        d = r_a - r_cat
        for var, dd in (("V1_seviyesiz", d - d.mean()), ("V2_seviyeli", d)):
            yeni_log = L.copy()
            tam = np.zeros(len(L))
            tam[pos[soguk]] = s_egim * dd
            yeni_log = yeni_log + tam
            y = np.clip(np.expm1(yeni_log), 0.0, None)
            out = pd.DataFrame({"id": IDS, "tuketim": y})
            adad = f"p20_harman_{ad}_{var}.csv"
            yol = os.path.join(CIK, adad)
            out.to_csv(yol, index=False)
            g = pd.read_csv(yol)
            ok = (
                len(g) == len(IDS)
                and np.array_equal(g["id"].to_numpy(), IDS)
                and int(g["tuketim"].isna().sum()) == 0
                and int((g["tuketim"] < 0).sum()) == 0
                and bool(np.isfinite(g["tuketim"].to_numpy()).all())
            )
            if not ok:
                raise SystemExit(f"DUR: {adad} dogrulamadan gecmedi")
            deg = int((np.abs(tam) > 1e-12).sum())
            uretilen.append(
                {
                    "dosya": adad,
                    "harman": ad,
                    "varyant": var,
                    "degisen_satir": deg,
                    "degisen_pay": round(deg / len(L), 4),
                    "log_kayma_ort": round(float(tam[pos[soguk]].mean()), 5),
                    "log_kayma_std": round(float(tam[pos[soguk]].std()), 5),
                    "log_kayma_min": round(float(tam.min()), 4),
                    "log_kayma_max": round(float(tam.max()), 4),
                    "dogrulama": "GECTI (714688 satir, id sirasi, NaN yok, negatif yok, sonlu)",
                }
            )
            print(
                f"{adad:42} degisen {deg:,}  kayma ort {tam[pos[soguk]].mean():+.5f} "
                f"std {tam[pos[soguk]].std():.5f}"
            )
    R["uretilen_dosyalar"] = uretilen
    R["cikti_dizini"] = CIK
    R["NOT"] = (
        "submissions/ altina YAZILMADI, GONDERIM YAPILMADI. "
        "V1 seviyeyi korur (LB'den cozulmus seviye katmaniyla cift saymaz); "
        "V2 blok CV'sinde olculen kazancin TAMAMINI tasir ama seviye "
        "duzeltmesiyle cakisma riski vardir."
    )

    yol = os.path.join(BURA, "p_kalici", "p20_aday.json")
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)

    print("\nSEVIYE / YAPI AYRISMASI (agr dMSE, beta=0.60)")
    for ad in ADAYLAR:
        print(
            f"  {ad}: TOPLAM {ayr[ad]['ORT_dMSE_TOPLAM']:+.5f} ({ayr[ad]['isaret_dMSE_TOPLAM']})"
            f" = YAPI {ayr[ad]['ORT_dMSE_YAPI(seviyesiz)']:+.5f} "
            f"({ayr[ad]['isaret_dMSE_YAPI(seviyesiz)']})"
            f" + SEVIYE {ayr[ad]['ORT_dMSE_SEVIYE(fark)']:+.5f} "
            f"({ayr[ad]['isaret_dMSE_SEVIYE(fark)']})"
        )
        for b in BLOKLAR:
            v = ayr[ad][b]
            print(
                f"    {b:6} toplam {v['dMSE_TOPLAM']:+.5f}  yapi "
                f"{v['dMSE_YAPI(seviyesiz)']:+.5f}  seviye {v['dMSE_SEVIYE(fark)']:+.5f}"
            )
    print(f"\nkayit: {yol}")


if __name__ == "__main__":
    main()
