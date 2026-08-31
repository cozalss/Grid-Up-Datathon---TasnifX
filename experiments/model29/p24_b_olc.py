"""p24-B: SICAK HARMAN DENETIMI -- soguktaki uretim hatasinin ikizi var mi?

SORU
----
Uretim sicak harmani (cat 3, xgb 1, lgbm 1, sinir_agi 1.4):
- cat'in 3x agirligi, tuketim_model.py yorumlarindaki kendi olcumlerinin
  "sicakta xgb en iyi aile" bulgusuyla celisiyor;
- sinir_agi 1.4 (harmanin %21.9'u) HICBIR olcum kaydinda gecmiyor
  (aile_onbellegi.py:15-17).

DUZENEK
-------
data/interim/aile_onbellek/{blok}_{tohum}_{aile}_uretim.npy (float32,
yalniz SICAK satirlar, egitim.parquet blok-ici sirasi; {blok}_gercek.npy
ile satir eslesmesi p24_a_kesif'te BIREBIR dogrulandi, maxfark 0.0).
CEKINCE (docs/80 §8): bu onbellek bugunku aile_onbellegi.py ile birebir
yeniden uretilemedi (maxabs 0.325) -- "yaklasik uretim". Ama butun adaylar
AYNI onbellekten harmanlandigindan KARSILASTIRMA gecerli.

Harman: log uzayinda agirlikli ortalama, agirlik toplamina bolunmus
(tuketim_model.py:1287-1305 ile ayni cebir). Sicak tarafta son islem yok;
tek kirpim max(p, 0).

Bilesik skor: soguk taraf SABIT = uretim sogugu (yalniz cat, tohum
1000-1002 ortalamasi, son islem beta=0.60), MSE uzayinda
soguk %22.16 / sicak %77.84.

Kohort: sicak satirlar test'in (p_gun_sayisi x kVA x ufuk) hucre
dagilimina agirliklandirilir (agr); ham da raporlanir. p24_a_kesif:
kapsam 0.993-1.000, TV 0.14-0.16 -- soguktaki kadar bozuk DEGIL, ama
pg(75,90] payi test %14.8 vs blok %1.4-3.5 oldugu icin agr ana olcut.

Kapilar: (a) gercek blok CV, (b) 0 parametre, (c) uc blok isareti,
(d) 3x3 tohum hucresi. Trafo-kumeli onyukleme 500, GA95.
Kirpma merdiveni K in {0,5,10,25} (trafo bazinda en uc |katki|lar atilir).
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
AILELER = ("cat", "xgb", "lgbm", "sinir_agi")
SOGUK_PAY = 0.2216
SICAK_PAY = 1 - SOGUK_PAY
BETA_SOGUK = 0.60
N_ONY = 500
MEVCUT_LB = 1.00115

#: 0-PARAMETRELI adaylar (cat, xgb, lgbm, sinir_agi)
ADAYLAR: dict[str, tuple[float, float, float, float]] = {
    "URETIM": (3.0, 1.0, 1.0, 1.4),
    "ESIT_4": (1.0, 1.0, 1.0, 1.0),
    "SINIRSIZ": (3.0, 1.0, 1.0, 0.0),
    "SINIRSIZ_ESIT": (1.0, 1.0, 1.0, 0.0),
    "XGB_AGIR": (1.0, 3.0, 1.0, 1.4),
    "XGB_AGIR_SINIRSIZ": (1.0, 3.0, 1.0, 0.0),
    "YALNIZ_XGB": (0.0, 1.0, 0.0, 0.0),
    "YALNIZ_CAT": (1.0, 0.0, 0.0, 0.0),
}
TABAN = "URETIM"

KVA_KENAR = [0, 60, 110, 175, 260, 410, 650, 1050, 1e9]
PG_KENAR = [30, 60, 75, 90, 105]
UFUK_KENAR = [30, 60, 90]


def hucre(pg, guc, ufuk):
    a = np.digitize(pg, PG_KENAR)
    b = np.digitize(guc, KVA_KENAR) - 1
    c = np.digitize(ufuk, UFUK_KENAR)
    return a * 100 + b * 10 + c


def veri_kur() -> dict:
    E = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
    T = pd.read_parquet(os.path.join(DN, "test.parquet"))
    tsic = T[T["soguk_mu"] == 0]
    th = hucre(
        tsic["p_gun_sayisi"].to_numpy("float64"),
        tsic["guc"].to_numpy("float64"),
        tsic["ufuk_gun"].to_numpy("float64"),
    )
    test_pay = pd.Series(th).value_counts(normalize=True).to_dict()
    del T, tsic

    B: dict = {}
    for blok in BLOKLAR:
        d = E[E["_blok"] == blok]
        ds = d[d["soguk_mu"] == 0]
        n = len(ds)
        # aile tahminleri: {aile: (3, n)}
        P = {
            a: np.stack(
                [
                    np.load(os.path.join(AO, f"{blok}_{t}_{a}_uretim.npy")).astype("float64")
                    for t in TOHUMLAR
                ]
            )
            for a in AILELER
        }
        assert all(P[a].shape == (3, n) for a in AILELER), blok
        g = np.load(os.path.join(AO, f"{blok}_gercek.npy")).astype("float64")
        assert float(np.max(np.abs(g - ds["tuketim"].to_numpy("float64")))) == 0.0
        y = np.log1p(np.clip(g, 0, None))
        h = hucre(
            ds["p_gun_sayisi"].to_numpy("float64"),
            ds["guc"].to_numpy("float64"),
            ds["ufuk_gun"].to_numpy("float64"),
        )
        blok_pay = pd.Series(h).value_counts(normalize=True).to_dict()
        w = np.array([test_pay.get(k, 0.0) / blok_pay[k] for k in h], dtype="float64")
        w = w / w.mean()
        # SOGUK taraf SABIT: uretim (yalniz cat, tohum ort, beta=0.60)
        dg = d[d["soguk_mu"] == 1]
        z = np.load(os.path.join(DN, f"soguk_tahmin_{blok}.npz"))
        pc = np.mean([z[f"{t}_cat"].astype("float64") for t in TOHUMLAR], axis=0)
        lguc = np.log1p(dg["guc"].to_numpy("float64"))
        r = pc - lguc
        r = float(r.mean()) + BETA_SOGUK * (r - float(r.mean()))
        yg = np.log1p(np.clip(dg["tuketim"].to_numpy("float64"), 0, None))
        eg = yg - np.maximum(r + lguc, 0.0)
        B[blok] = dict(
            P=P,
            y=y,
            w=w,
            n=n,
            tanim=ds["tanim"].to_numpy(),
            soguk_mse=float((eg * eg).mean()),
            n_soguk=len(dg),
        )
    return B


def harman(b: dict, wv, tohum=None) -> np.ndarray:
    top = sum(wv)
    if tohum is None:
        p = sum(x * b["P"][a].mean(axis=0) for x, a in zip(wv, AILELER)) / top
    else:
        i = TOHUMLAR.index(tohum)
        p = sum(x * b["P"][a][i] for x, a in zip(wv, AILELER)) / top
    return np.maximum(p, 0.0)


def hata(b, wv, tohum=None, seviyesiz=False, p_taban=None):
    p = harman(b, wv, tohum)
    if seviyesiz:
        p = p - float(p.mean()) + float(p_taban.mean())
        p = np.maximum(p, 0.0)
    return b["y"] - p


def olc(b, e):
    e2 = e * e
    return {"ham": float(e2.mean()), "agr": float((b["w"] * e2).mean())}


def main() -> None:
    B = veri_kur()
    R: dict = {
        "00_SORU": "Uretim sicak harmani (3,1,1,1.4) 0-parametreli adaylara yenik mi?",
        "01_CEKINCE": (
            "onbellek bugunku aile_onbellegi.py ile birebir degil (docs/80 §8, "
            "maxabs 0.325) -- mutlak seviyeler yaklasik, KARSILASTIRMA gecerli"
        ),
        "adaylar": {k: list(v) for k, v in ADAYLAR.items()},
        "soguk_sabit": {b: round(B[b]["soguk_mse"], 6) for b in BLOKLAR},
        "paylar": {"soguk": SOGUK_PAY, "sicak": SICAK_PAY},
    }

    # --- 1) ANA TABLO: aday x blok, ham/agr MSE + bilesik RMSLE
    tablo: dict = {}
    for ad, wv in ADAYLAR.items():
        tablo[ad] = {}
        for b in BLOKLAR:
            m = olc(B[b], hata(B[b], wv))
            m["bilesik_rmsle_ham"] = round(
                float(np.sqrt(SOGUK_PAY * B[b]["soguk_mse"] + SICAK_PAY * m["ham"])), 5
            )
            m["bilesik_rmsle_agr"] = round(
                float(np.sqrt(SOGUK_PAY * B[b]["soguk_mse"] + SICAK_PAY * m["agr"])), 5
            )
            m["tohum"] = {str(t): olc(B[b], hata(B[b], wv, t)) for t in TOHUMLAR}
            tablo[ad][b] = m
    R["tablo"] = tablo

    # --- 2) KARSILASTIRMA vs URETIM (pozitif = aday IYI) + YAPI/SEVIYE ayrismasi
    kars: dict = {}
    for ad, wv in ADAYLAR.items():
        if ad == TABAN:
            continue
        kars[ad] = {}
        for olcut in ("ham", "agr"):
            dm = {b: tablo[TABAN][b][olcut] - tablo[ad][b][olcut] for b in BLOKLAR}
            ort = float(np.mean(list(dm.values())))
            kars[ad][olcut] = {
                **{b: round(dm[b], 6) for b in BLOKLAR},
                "ORT_dMSE": round(ort, 6),
                "isaret": f"{sum(v > 0 for v in dm.values())}/3",
                "test_dMSE": round(ort * SICAK_PAY, 6),
            }
        # tohum 9 hucre (agr)
        th = {
            b: {
                str(t): round(
                    tablo[TABAN][b]["tohum"][str(t)]["agr"] - tablo[ad][b]["tohum"][str(t)]["agr"],
                    6,
                )
                for t in TOHUMLAR
            }
            for b in BLOKLAR
        }
        kars[ad]["tohum_agr"] = th
        kars[ad]["tohum_isareti"] = (
            f"{sum(1 for b in BLOKLAR for t in TOHUMLAR if th[b][str(t)] > 0)}/9"
        )
        # YAPI (seviyesiz): aday ortalamasi TABANA esitlenmis
        yapi = {}
        for b in BLOKLAR:
            pt = harman(B[b], ADAYLAR[TABAN])
            et = B[b]["y"] - pt
            ey = hata(B[b], wv, seviyesiz=True, p_taban=pt)
            yapi[b] = round(float((B[b]["w"] * (et * et - ey * ey)).mean()), 6)
        kars[ad]["YAPI_agr"] = {
            **yapi,
            "ORT": round(float(np.mean(list(yapi.values()))), 6),
            "isaret": f"{sum(v > 0 for v in yapi.values())}/3",
        }
    R["karsilastirma_vs_URETIM"] = kars

    # --- 3) ONYUKLEME (trafo kumeli, 500) + KIRPMA MERDIVENI (agr)
    ony: dict = {}
    for ad, wv in ADAYLAR.items():
        if ad == TABAN:
            continue
        ony[ad] = {}
        hepsi = []
        for b in BLOKLAR:
            et = hata(B[b], ADAYLAR[TABAN])
            ea = hata(B[b], wv)
            d2 = (et * et - ea * ea) * B[b]["w"]
            s = pd.Series(d2).groupby(B[b]["tanim"])
            top, adet = s.sum().to_numpy(), s.size().to_numpy().astype("float64")
            rng = np.random.default_rng(240 + BLOKLAR.index(b))
            k = len(top)
            idx = rng.integers(0, k, size=(N_ONY, k))
            o = top[idx].sum(axis=1) / adet[idx].sum(axis=1)
            hepsi.append(o)
            ss = s.sum().sort_values()
            kirp = {}
            for K in (0, 5, 10, 25):
                at = set(ss.index[:K]) | set(ss.index[len(ss) - K :]) if K else set()
                m = ~pd.Series(B[b]["tanim"]).isin(at).to_numpy()
                kirp[f"K={K}"] = round(float(d2[m].sum() / m.sum()), 6)
            ony[ad][b] = {
                "dMSE_agr": round(float(top.sum() / adet.sum()), 6),
                "GA95": [
                    round(float(np.percentile(o, 2.5)), 6),
                    round(float(np.percentile(o, 97.5)), 6),
                ],
                "P_pozitif": round(float((o > 0).mean()), 4),
                "kirpma": kirp,
                "trafo": int(k),
            }
        oo = np.mean(hepsi, axis=0)
        v = [ony[ad][b]["dMSE_agr"] for b in BLOKLAR]
        g = float(np.mean(v))
        kk = {
            f"K={K}": round(float(np.mean([ony[ad][b]["kirpma"][f"K={K}"] for b in BLOKLAR])), 6)
            for K in (0, 5, 10, 25)
        }
        ony[ad]["OZET"] = {
            "ORT_dMSE_agr": round(g, 6),
            "isaret": f"{sum(x > 0 for x in v)}/3",
            "GA95_blokici": [
                round(float(np.percentile(oo, 2.5)), 6),
                round(float(np.percentile(oo, 97.5)), 6),
            ],
            "P_pozitif": round(float((oo > 0).mean()), 4),
            "kirpma_ORT": kk,
            "test_dMSE_agr": round(g * SICAK_PAY, 6),
            "beklenen_LB_oran1.0": round(float(np.sqrt(max(MEVCUT_LB**2 - g * SICAK_PAY, 0))), 5),
            "beklenen_LB_oran0.5": round(
                float(np.sqrt(max(MEVCUT_LB**2 - 0.5 * g * SICAK_PAY, 0))), 5
            ),
            "MUHAFAZAKAR_K25_test_dMSE": round(kk["K=25"] * SICAK_PAY, 6),
        }
    R["onyukleme"] = ony

    yol = os.path.join(BURA, "p_kalici", "p24_sicak_harman.json")
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)

    # --- ekran
    print("=== SICAK HARMAN (dMSE = URETIM - aday; POZITIF = aday IYI) ===")
    print(
        f"{'aday':20}{'olcut':6}"
        + "".join(f"{b:>11}" for b in BLOKLAR)
        + f"{'ORT':>11}{'testdMSE':>11}{'isrt':>6}{'tohum':>7}"
    )
    for ad in ADAYLAR:
        if ad == TABAN:
            continue
        for olcut in ("ham", "agr"):
            d = kars[ad][olcut]
            print(
                f"{ad:20}{olcut:6}"
                + "".join(f"{d[b]:>+11.5f}" for b in BLOKLAR)
                + f"{d['ORT_dMSE']:>+11.5f}{d['test_dMSE']:>+11.5f}"
                + f"{d['isaret']:>6}{kars[ad]['tohum_isareti']:>7}"
            )
        yv = kars[ad]["YAPI_agr"]
        print(
            f"{'':20}{'YAPI':6}"
            + "".join(f"{yv[b]:>+11.5f}" for b in BLOKLAR)
            + f"{yv['ORT']:>+11.5f}{'':>11}{yv['isaret']:>6}"
        )
    print("\n=== ONYUKLEME (trafo kumeli 500, agr) ===")
    for ad in ADAYLAR:
        if ad == TABAN:
            continue
        o = ony[ad]["OZET"]
        print(
            f"{ad:20} ORT {o['ORT_dMSE_agr']:>+9.5f} {o['isaret']} "
            f"GA95 [{o['GA95_blokici'][0]:+.5f},{o['GA95_blokici'][1]:+.5f}] "
            f"P(+)={o['P_pozitif']:.3f}  kirpma "
            + " ".join(f"{k}={x:+.4f}" for k, x in o["kirpma_ORT"].items())
        )
    print("\nkayit:", yol)


if __name__ == "__main__":
    main()
