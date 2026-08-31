"""p24-A: SICAK HARMAN DENETIMI -- kesif ve hizalama dogrulamasi.

1. aile_onbellek/{blok}_{tohum}_{aile}_uretim.npy sekilleri sicak satir
   sayisiyla tutuyor mu (sinir_agi DAHIL)?
2. {blok}_gercek.npy sicak satirlarin log1p(tuketim)'i mi (satir eslesme)?
3. Sicak satirlarin pencere/ufuk kohort dagilimi test'e ne kadar uyuyor
   (kapsam, ESS, TV mesafesi) -- soguktaki kadar bozuk mu?
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

KVA_KENAR = [0, 60, 110, 175, 260, 410, 650, 1050, 1e9]
PG_KENAR = [30, 60, 75, 90, 105]
UFUK_KENAR = [30, 60, 90]


def hucre(pg, guc, ufuk):
    a = np.digitize(pg, PG_KENAR)
    b = np.digitize(guc, KVA_KENAR) - 1
    c = np.digitize(ufuk, UFUK_KENAR)
    return a * 100 + b * 10 + c


def main() -> None:
    E = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
    T = pd.read_parquet(os.path.join(DN, "test.parquet"))
    tsic = T[T["soguk_mu"] == 0]
    test_h = hucre(
        tsic["p_gun_sayisi"].to_numpy("float64"),
        tsic["guc"].to_numpy("float64"),
        tsic["ufuk_gun"].to_numpy("float64"),
    )
    test_pay = pd.Series(test_h).value_counts(normalize=True)

    R: dict = {
        "test_sicak_satir": int(len(tsic)),
        "test_sicak_hucre": int(test_pay.size),
        "test_pg_75_90_pay": float(((tsic.p_gun_sayisi > 75) & (tsic.p_gun_sayisi <= 90)).mean()),
        "bloklar": {},
    }

    for blok in BLOKLAR:
        d = E[E["_blok"] == blok]
        ds = d[d["soguk_mu"] == 0]
        n = len(ds)
        sat: dict = {"n_sicak": n, "n_trafo": int(ds["tanim"].nunique())}

        # 1) sekiller
        sek = {}
        for a in AILELER:
            for t in TOHUMLAR:
                yol = os.path.join(AO, f"{blok}_{t}_{a}_uretim.npy")
                if not os.path.exists(yol):
                    sek[f"{t}_{a}"] = "YOK"
                else:
                    arr = np.load(yol, mmap_mode="r")
                    sek[f"{t}_{a}"] = list(arr.shape) if arr.shape != (n,) else "OK"
        sat["sekil"] = {k: v for k, v in sek.items() if v != "OK"} or "HEPSI OK (12/12)"

        # 2) gercek.npy eslesmesi
        g = np.load(os.path.join(AO, f"{blok}_gercek.npy")).astype("float64")
        y = np.log1p(np.clip(ds["tuketim"].to_numpy("float64"), 0, None))
        sat["gercek_sekil"] = list(g.shape)
        if len(g) == n:
            sat["gercek_maxfark_log1p"] = float(np.max(np.abs(g - y)))
        else:
            sat["gercek_maxfark_log1p"] = "SEKIL FARKLI"

        # 3) kohort uyumu
        h = hucre(
            ds["p_gun_sayisi"].to_numpy("float64"),
            ds["guc"].to_numpy("float64"),
            ds["ufuk_gun"].to_numpy("float64"),
        )
        blok_pay = pd.Series(h).value_counts(normalize=True)
        ortak = test_pay.index.intersection(blok_pay.index)
        kapsam = float(test_pay.loc[ortak].sum())
        tv = 0.5 * float(
            (
                test_pay.reindex(test_pay.index.union(blok_pay.index), fill_value=0)
                - blok_pay.reindex(test_pay.index.union(blok_pay.index), fill_value=0)
            )
            .abs()
            .sum()
        )
        w = np.array([test_pay.get(k, 0.0) / blok_pay[k] for k in h])
        w = w / w.mean()
        ess = float(w.sum() ** 2 / (w * w).sum())
        sat["kohort"] = {
            "blok_hucre": int(blok_pay.size),
            "kapsam_test_payi": round(kapsam, 4),
            "tv_mesafesi": round(tv, 4),
            "ess_oran": round(ess / n, 4),
            "w_max": round(float(w.max()), 2),
            "pg_75_90_pay": round(
                float(((ds.p_gun_sayisi > 75) & (ds.p_gun_sayisi <= 90)).mean()), 4
            ),
        }

        # ornek: aile korelasyonlari (tohum 1000)
        P = {
            a: np.load(os.path.join(AO, f"{blok}_1000_{a}_uretim.npy")).astype("float64")
            for a in AILELER
        }
        sat["aile_rmsle_t1000"] = {
            a: round(float(np.sqrt(np.mean((y - np.maximum(P[a], 0)) ** 2))), 5) for a in AILELER
        }
        R["bloklar"][blok] = sat
        print(blok, json.dumps(sat, ensure_ascii=False, default=str)[:600])

    # soguk npz anahtarlari (uretim soguk tarafi icin)
    z = np.load(os.path.join(DN, "soguk_tahmin_yaz25.npz"))
    R["soguk_npz_anahtar_ornek"] = sorted(z.files)[:12]

    yol = os.path.join(BURA, "p_kalici", "p24_kesif.json")
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    print("kayit:", yol)


if __name__ == "__main__":
    main()
