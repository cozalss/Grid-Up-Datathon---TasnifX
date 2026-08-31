"""p20: SOGUK HARMAN EKSENI -- URETIM SON ISLEMI ALTINDA, KOHORT AGIRLIKLI.

SORU
----
Uretim soguk uzmani YALNIZ CAT ({"cat": 1.0}, tuketim_model.py:990-995).
Bu secim 2026-08-23'te (d04243f) YALNIZCA kis26'da yapildi. Test donemi
Nisan-Temmuz, yani yaz25 analogu -- ve orada cat-tekil, esit harmandan
0,14 KOTU (son islem ONCESI olcum, p18_harman_bilesim.json).

Bu betik o eksenin uretim son islemi ALTINDA, testin kohort dagilimina
AGIRLIKLANDIRILMIS olcumunu yapar.

URETIM SON ISLEMI (scripts/son_islem.py, beta=0.60)
---------------------------------------------------
    r  = log1p(tahmin) - log1p(guc)          # kapasite ofsetli uzay
    r' = ort(r_soguk) + beta * (r - ort)     # YALNIZ soguk satirlarda
    yeni = expm1(r' + log1p(guc))
Docstring: "URETIMDE KULLANILAN SON ISLEM BUDUR. LB'DE UC KEZ DOGRULANDI."
son_islem_gun.py (gun korumali) LB'de CURUTULDU ve kullanilmiyor.

DIKKAT -- BUZME HARMANI KAYIRIR
-------------------------------
Buzme yayilmayi kirpar. Esit harman zaten ortalama alarak yayilmayi
dusurmustur, cat-tekil ise daha yayilmistir. Sabit bir beta iki adaya
AYNI seyi yapmaz. Bu yuzden beta her harman icin BLOK-DISI secilir
(hedef blogun verisi kullanilmadan, diger iki bloktan).

OLCUTLER (uc tanesi de raporlanir, KARAR agirlikli olana gore)
--------------------------------------------------------------
ham  : soguk satirlarda duz MSE
agr  : testin soguk (p_gun_sayisi x kVA kovasi x ufuk kovasi) ortak hucre
       dagilimina yeniden agirliklandirilmis MSE
pg   : p_gun_sayisi in (75,90] alt kumesi -- testin soguklarinin %82,3'u

KAPILAR (docs/79 EK)
--------------------
(a) gercek blok CV'si mi  -> EVET, vekil yok
(b) blok-disi secim mi    -> beta blok-disi; harman adaylari 0-PARAMETRELI
(c) isaret tutarli mi     -> raporlanir
(d) tohum sayisiyla ayakta mi -> tohum tohum raporlanir
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BURA = os.path.dirname(os.path.abspath(__file__))
DN = os.path.join(KOK, "data", "interim", "deney")
AO = os.path.join(KOK, "data", "interim", "aile_onbellek")

BLOKLAR = ("yaz25", "guz25", "kis26")
TOHUMLAR = (1000, 1001, 1002)
AILE = ("cat", "xgb", "lgbm")
SOGUK_PAY = 0.2216  # test'teki soguk satir payi (158.369 / 714.688)
BETA_IZGARA = np.round(np.arange(0.20, 1.0001, 0.05), 3)

#: 0 PARAMETRELI harman adaylari (izgara aramasi YOK -- p06 tam orada kirildi)
HARMANLAR: dict[str, tuple[float, float, float]] = {
    "URETIM_cat": (1.0, 0.0, 0.0),
    "ESIT": (1 / 3, 1 / 3, 1 / 3),
    "ESKI_3_1_1": (0.6, 0.2, 0.2),
    "CATSIZ_xgb_lgbm": (0.0, 0.5, 0.5),
    "yalniz_xgb": (0.0, 1.0, 0.0),
    "yalniz_lgbm": (0.0, 0.0, 1.0),
}

KVA_KENAR = [0, 60, 110, 175, 260, 410, 650, 1050, 1e9]
PG_KENAR = [30, 60, 75, 90, 105]
UFUK_KENAR = [30, 60, 90]


def hucre(pg: np.ndarray, guc: np.ndarray, ufuk: np.ndarray) -> np.ndarray:
    a = np.digitize(pg, PG_KENAR)
    b = np.digitize(guc, KVA_KENAR) - 1
    c = np.digitize(ufuk, UFUK_KENAR)
    return a * 100 + b * 10 + c


# --------------------------------------------------------------- veri


def veri_kur() -> tuple[dict, dict]:
    E = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
    T = pd.read_parquet(os.path.join(DN, "test.parquet"))
    ts = T[T["soguk_mu"] == 1]
    test_h = hucre(
        ts["p_gun_sayisi"].to_numpy("float64"),
        ts["guc"].to_numpy("float64"),
        ts["ufuk_gun"].to_numpy("float64"),
    )
    test_pay = pd.Series(test_h).value_counts(normalize=True).to_dict()

    B: dict = {}
    for blok in BLOKLAR:
        d = E[E["_blok"] == blok]
        dg = d[d["soguk_mu"] == 1]
        z = np.load(os.path.join(DN, f"soguk_tahmin_{blok}.npz"))
        P = {a: np.stack([z[f"{t}_{a}"].astype("float64") for t in TOHUMLAR]) for a in AILE}
        guc = dg["guc"].to_numpy("float64")
        lgy = np.log1p(np.clip(dg["tuketim"].to_numpy("float64"), 0, None))
        h = hucre(dg["p_gun_sayisi"].to_numpy("float64"), guc, dg["ufuk_gun"].to_numpy("float64"))
        blok_pay = pd.Series(h).value_counts(normalize=True).to_dict()
        w = np.array([test_pay.get(k, 0.0) / blok_pay[k] for k in h], dtype="float64")
        kapsam = float(sum(v for k, v in test_pay.items() if k in blok_pay))
        w = w / w.mean()
        pgm = (dg["p_gun_sayisi"].to_numpy("float64") > 75) & (
            dg["p_gun_sayisi"].to_numpy("float64") <= 90
        )
        # SICAK taraf -- URETIM harmani (3/1/1/1.4), degismiyor
        ds = d[d["soguk_mu"] == 0]
        W = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 1.4}
        fam = {
            a: np.mean(
                [
                    np.load(os.path.join(AO, f"{blok}_{t}_{a}_uretim.npy")).astype("float64")
                    for t in TOHUMLAR
                ],
                axis=0,
            )
            for a in W
        }
        ps = sum(W[a] * fam[a] for a in W) / sum(W.values())
        ys = np.log1p(np.clip(ds["tuketim"].to_numpy("float64"), 0, None))
        e = ys - np.maximum(ps, 0.0)
        B[blok] = dict(
            P=P,
            lgy=lgy,
            lguc=np.log1p(guc),
            w=w,
            pgm=pgm,
            kapsam=kapsam,
            tanim=dg["tanim"].to_numpy(),
            n=len(dg),
            sicak_mse=float((e * e).mean()),
            n_sicak=len(ds),
            hucre_sayisi=int(len(blok_pay)),
            guc=guc,
            pg=dg["p_gun_sayisi"].to_numpy("float64"),
            sifir_pay=float((dg["tuketim"].to_numpy("float64") <= 0).mean()),
        )
    B["_test"] = dict(
        hucre_sayisi=int(len(test_pay)),
        guc_medyan=float(ts["guc"].median()),
        pg_75_90_pay=float(((ts["p_gun_sayisi"] > 75) & (ts["p_gun_sayisi"] <= 90)).mean()),
        kova12_pay=float((np.digitize(ts["guc"].to_numpy("float64"), KVA_KENAR) - 1 >= 6).mean()),
    )
    return B, test_pay


# ------------------------------------------------------------ olcum


def buz(r: np.ndarray, beta: float) -> np.ndarray:
    o = float(r.mean())
    return o + beta * (r - o)


def hatalar(
    b: dict, w_harman: tuple[float, float, float], beta: float, tohum: int | None = None
) -> np.ndarray:
    """Harman + son islem -> log uzayinda hata vektoru."""
    if tohum is None:
        p = sum(wa * b["P"][a].mean(axis=0) for wa, a in zip(w_harman, AILE))
    else:
        i = TOHUMLAR.index(tohum)
        p = sum(wa * b["P"][a][i] for wa, a in zip(w_harman, AILE))
    r = p - b["lguc"]
    if beta is not None and beta != 1.0:
        r = buz(r, beta)
    return b["lgy"] - np.maximum(r + b["lguc"], 0.0)


def olcutler(b: dict, e: np.ndarray) -> dict:
    e2 = e * e
    return {
        "ham": float(e2.mean()),
        "agr": float((b["w"] * e2).mean()),
        "pg": float(e2[b["pgm"]].mean()),
    }


def main() -> None:
    B, test_pay = veri_kur()
    R: dict = {
        "son_islem": "scripts/son_islem.py -- r' = ort + beta*(r-ort), YALNIZ soguk satirlarda",
        "tohumlar": list(TOHUMLAR),
        "harmanlar": {k: list(v) for k, v in HARMANLAR.items()},
        "test_kohortu": B["_test"],
        "blok_kohort_kapsami": {b: round(B[b]["kapsam"], 4) for b in BLOKLAR},
    }

    # --- 1) BETA BLOK-DISI SECIMI (her harman icin, her olcut icin)
    beta_sec: dict = {}
    for ad, wv in HARMANLAR.items():
        beta_sec[ad] = {}
        for olcut in ("ham", "agr", "pg"):
            beta_sec[ad][olcut] = {}
            for hedef in BLOKLAR:
                dis = [x for x in BLOKLAR if x != hedef]
                en_iyi, en_iyi_v = None, np.inf
                for beta in BETA_IZGARA:
                    v = float(
                        np.mean([olcutler(B[x], hatalar(B[x], wv, beta))[olcut] for x in dis])
                    )
                    if v < en_iyi_v:
                        en_iyi, en_iyi_v = float(beta), v
                beta_sec[ad][olcut][hedef] = en_iyi
    R["beta_blok_disi"] = beta_sec

    # --- 2) ANA TABLO
    tablo: dict = {}
    for ad, wv in HARMANLAR.items():
        tablo[ad] = {}
        for blok in BLOKLAR:
            sat: dict = {}
            for etiket, beta in (("son_islem_YOK", 1.0), ("beta060_URETIM", 0.60)):
                sat[etiket] = olcutler(B[blok], hatalar(B[blok], wv, beta))
            for olcut in ("ham", "agr", "pg"):
                bb = beta_sec[ad][olcut][blok]
                sat.setdefault("beta_blok_disi", {})[olcut] = {
                    "beta": bb,
                    "mse": olcutler(B[blok], hatalar(B[blok], wv, bb))[olcut],
                }
            # tohum tohum (beta=0.60)
            sat["tohum_bazinda_beta060"] = {
                str(t): olcutler(B[blok], hatalar(B[blok], wv, 0.60, t)) for t in TOHUMLAR
            }

            tablo[ad][blok] = sat
    R["tablo"] = tablo

    # --- 3) KARSILASTIRMA: her aday vs URETIM_cat
    TABAN = "URETIM_cat"
    kars: dict = {}
    for ad in HARMANLAR:
        if ad == TABAN:
            continue
        kars[ad] = {}
        for etiket in ("son_islem_YOK", "beta060_URETIM"):
            d = {}
            for olcut in ("ham", "agr", "pg"):
                dm = {
                    b: tablo[TABAN][b][etiket][olcut] - tablo[ad][b][etiket][olcut] for b in BLOKLAR
                }
                d[olcut] = {
                    **{b: round(dm[b], 6) for b in BLOKLAR},
                    "ORT_dMSE": round(float(np.mean(list(dm.values()))), 6),
                    "isaret_3_3": all(v > 0 for v in dm.values())
                    or all(v < 0 for v in dm.values()),
                    "kazanan_blok": int(sum(v > 0 for v in dm.values())),
                    "test_dMSE": round(float(np.mean(list(dm.values()))) * SOGUK_PAY, 6),
                }
            kars[ad][etiket] = d
        # blok-disi beta ile
        d = {}
        for olcut in ("ham", "agr", "pg"):
            dm = {
                b: tablo[TABAN][b]["beta_blok_disi"][olcut]["mse"]
                - tablo[ad][b]["beta_blok_disi"][olcut]["mse"]
                for b in BLOKLAR
            }
            d[olcut] = {
                **{b: round(dm[b], 6) for b in BLOKLAR},
                "ORT_dMSE": round(float(np.mean(list(dm.values()))), 6),
                "kazanan_blok": int(sum(v > 0 for v in dm.values())),
                "test_dMSE": round(float(np.mean(list(dm.values()))) * SOGUK_PAY, 6),
            }
        kars[ad]["beta_blok_disi"] = d
        # tohum bazinda (agr, beta=0.60)
        kars[ad]["tohum_bazinda_agr_dMSE"] = {
            b: {
                str(t): round(
                    tablo[TABAN][b]["tohum_bazinda_beta060"][str(t)]["agr"]
                    - tablo[ad][b]["tohum_bazinda_beta060"][str(t)]["agr"],
                    6,
                )
                for t in TOHUMLAR
            }
            for b in BLOKLAR
        }
    R["karsilastirma_vs_URETIM_cat"] = kars

    # --- 4) BILESIK RMSLE (uretim sicak tarafi sabit)
    bil: dict = {}
    for ad, wv in HARMANLAR.items():
        bil[ad] = {}
        for etiket, beta in (("son_islem_YOK", 1.0), ("beta060_URETIM", 0.60)):
            v = {}
            for b in BLOKLAR:
                mc = tablo[ad][b][etiket]["ham"]
                v[b] = round(
                    float(np.sqrt(SOGUK_PAY * mc + (1 - SOGUK_PAY) * B[b]["sicak_mse"])), 5
                )
            v["ORT"] = round(float(np.mean([v[b] for b in BLOKLAR])), 5)
            bil[ad][etiket] = v
    R["bilesik_RMSLE_ham"] = bil

    yol = os.path.join(BURA, "p_kalici", "p20_harman.json")
    os.makedirs(os.path.dirname(yol), exist_ok=True)
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)

    # --- ekran
    print(
        f"test kohort hucresi {B['_test']['hucre_sayisi']}  blok kapsami {R['blok_kohort_kapsami']}"
    )
    for etiket in ("son_islem_YOK", "beta060_URETIM", "beta_blok_disi"):
        print(f"\n=== {etiket} === (dMSE = URETIM_cat - aday; POZITIF = aday IYI)")
        print(
            f"{'aday':18}{'olcut':6}"
            + "".join(f"{b:>11}" for b in BLOKLAR)
            + f"{'ORT':>11}{'testdMSE':>11}{'kazanan':>9}"
        )
        for ad in HARMANLAR:
            if ad == TABAN:
                continue
            for olcut in ("ham", "agr", "pg"):
                d = kars[ad][etiket][olcut]
                print(
                    f"{ad:18}{olcut:6}"
                    + "".join(f"{d[b]:>+11.5f}" for b in BLOKLAR)
                    + f"{d['ORT_dMSE']:>+11.5f}{d['test_dMSE']:>+11.5f}"
                    + f"{d['kazanan_blok']:>9}/3"
                )
    print(f"\nkayit: {yol}")


if __name__ == "__main__":
    main()
