"""p20b: SOGUK HARMAN EKSENI -- onyukleme (trafo kumeli) ve kis26 teshisi.

1) ONYUKLEME: trafo bazinda kumeli, 500 tekrar, kohort agirlikli dMSE icin
   GA95. Trafo kumeli olmasi sart -- ayni trafonun ~100 gunu bagimsiz degil.
2) kis26 TESHISI: kis26 neden tek basina cat-tekili seviyor? Ayristirmalar:
   sifir/sifirdisi, kVA kovasi, ay, p_gun_sayisi kovasi, trafo bazinda
   yogunlasma.
3) TEST-BENZERI ALT KUME: her blogu testin baskin kohortuna (kova>=650 kVA
   VE p_gun in (75,90]) daraltip yeniden olc.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)

from p20_harman import (  # noqa: E402
    AILE,
    BLOKLAR,
    HARMANLAR,
    KVA_KENAR,
    SOGUK_PAY,
    TOHUMLAR,
    hatalar,
    veri_kur,
)

BETA = 0.60
N_ONY = 500
ADAYLAR = ("ESIT", "ESKI_3_1_1", "CATSIZ_xgb_lgbm")
TABAN = "URETIM_cat"
MEVCUT_LB = 1.00115


def main() -> None:
    B, _ = veri_kur()
    E = pd.read_parquet(
        os.path.join(
            os.path.dirname(os.path.dirname(BURA)), "data", "interim", "deney", "egitim.parquet"
        )
    )
    R: dict = {"beta": BETA, "n_onyukleme": N_ONY, "tohumlar": list(TOHUMLAR)}

    # hata vektorleri
    HAT = {ad: {b: hatalar(B[b], HARMANLAR[ad], BETA) for b in BLOKLAR} for ad in HARMANLAR}

    # ---------------- 1) ONYUKLEME (trafo kumeli)
    ony: dict = {}
    for ad in ADAYLAR:
        ony[ad] = {}
        blok_ort = []
        for b in BLOKLAR:
            w, tan = B[b]["w"], B[b]["tanim"]
            d2 = (HAT[TABAN][b] ** 2 - HAT[ad][b] ** 2) * w  # POZITIF = aday IYI
            s = pd.Series(d2).groupby(tan)
            top = s.sum().to_numpy()
            adet = s.size().to_numpy().astype("float64")
            n = float(adet.sum())
            gozlem = float(top.sum() / n)
            rng = np.random.default_rng(20 + BLOKLAR.index(b))
            k = len(top)
            idx = rng.integers(0, k, size=(N_ONY, k))
            ornek = top[idx].sum(axis=1) / adet[idx].sum(axis=1)
            ony[ad][b] = {
                "dMSE_agr": round(gozlem, 6),
                "GA95": [
                    round(float(np.percentile(ornek, 2.5)), 6),
                    round(float(np.percentile(ornek, 97.5)), 6),
                ],
                "P_pozitif": round(float((ornek > 0).mean()), 4),
                "trafo": int(k),
            }
            blok_ort.append(ornek)
        ort = np.mean(blok_ort, axis=0)
        gozlem_ort = float(np.mean([ony[ad][b]["dMSE_agr"] for b in BLOKLAR]))
        ony[ad]["UC_BLOK_ORT"] = {
            "dMSE_agr": round(gozlem_ort, 6),
            "GA95_blokici_gurultu": [
                round(float(np.percentile(ort, 2.5)), 6),
                round(float(np.percentile(ort, 97.5)), 6),
            ],
            "P_pozitif": round(float((ort > 0).mean()), 4),
            "UYARI": "Bu GA yalnizca BLOK ICI ornekleme gurultusunu tasir. "
            "Bloklar arasi sacilim (yaz +, kis -) ondan COK daha buyuk; "
            "asagidaki bloklar_arasi_se ile birlikte okunmali.",
        }
        v = [ony[ad][b]["dMSE_agr"] for b in BLOKLAR]
        ony[ad]["bloklar_arasi"] = {
            "sd": round(float(np.std(v, ddof=1)), 6),
            "se_ort": round(float(np.std(v, ddof=1) / np.sqrt(3)), 6),
            "isaret": f"{sum(x > 0 for x in v)}/3",
        }
        # LB cevrimi
        td = gozlem_ort * SOGUK_PAY
        ony[ad]["LB_cevrimi"] = {
            "test_dMSE": round(td, 6),
            "dRMSLE_oran1.0": round(MEVCUT_LB - float(np.sqrt(max(MEVCUT_LB**2 - td, 0))), 5),
            "beklenen_LB_oran1.0": round(float(np.sqrt(max(MEVCUT_LB**2 - td, 0))), 5),
            "beklenen_LB_oran0.5": round(float(np.sqrt(max(MEVCUT_LB**2 - 0.5 * td, 0))), 5),
        }
    R["onyukleme"] = ony

    # ---------------- 2) kis26 TESHISI
    tes: dict = {}
    for b in BLOKLAR:
        d = E[(E["_blok"] == b) & (E["soguk_mu"] == 1)]
        y = d["tuketim"].to_numpy("float64")
        guc = B[b]["guc"]
        kova = np.digitize(guc, KVA_KENAR) - 1
        ay = pd.to_datetime(d["tarih"]).dt.month.to_numpy()
        w = B[b]["w"]
        d2 = (HAT[TABAN][b] ** 2 - HAT["ESIT"][b] ** 2) * w
        pay = {}
        m0 = y <= 0
        pay["sifir_satir_payi"] = round(float(m0.mean()), 4)
        pay["dMSE_sifir_satirlardan"] = round(float(d2[m0].sum() / len(d2)), 6)
        pay["dMSE_sifirdisi_satirlardan"] = round(float(d2[~m0].sum() / len(d2)), 6)
        pay["kova_bazinda_dMSE_payi"] = {
            int(k): round(float(d2[kova == k].sum() / len(d2)), 6) for k in sorted(set(kova))
        }
        pay["kova>=6(650+kVA)_satir_payi"] = round(float((kova >= 6).mean()), 4)
        pay["ay_bazinda_dMSE_payi"] = {
            int(a): round(float(d2[ay == a].sum() / len(d2)), 6) for a in sorted(set(ay))
        }
        # trafo yogunlasmasi
        s = pd.Series(d2).groupby(B[b]["tanim"]).sum().sort_values()
        tp = float(s.sum())
        pay["en_kotu_trafo_dMSE_payi"] = round(float(s.iloc[0] / tp), 4) if tp != 0 else None
        pay["en_kotu_5_trafo_dMSE_payi"] = (
            round(float(s.iloc[:5].sum() / tp), 4) if tp != 0 else None
        )
        pay["trafo_sayisi"] = int(len(s))
        pay["kazanan_trafo_orani"] = round(float((s > 0).mean()), 4)
        tes[b] = pay
    R["kis26_teshisi"] = tes

    # ---------------- 3) TEST-BENZERI ALT KUME
    alt: dict = {}
    for b in BLOKLAR:
        guc, pg = B[b]["guc"], B[b]["pg"]
        m = (guc >= 650) & (pg > 75) & (pg <= 90)
        alt[b] = {"n": int(m.sum()), "pay": round(float(m.mean()), 4)}
        if m.sum() > 500:
            for ad in (TABAN,) + ADAYLAR:
                alt[b][ad] = round(float((HAT[ad][b][m] ** 2).mean()), 5)
            for ad in ADAYLAR:
                alt[b][f"dMSE_{ad}"] = round(alt[b][TABAN] - alt[b][ad], 5)
    R["test_benzeri_alt_kume_kova650+_pg(75,90]"] = alt

    yol = os.path.join(BURA, "p_kalici", "p20_onyukleme.json")
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)

    print("ONYUKLEME (trafo kumeli, 500) -- dMSE_agr, POZITIF = aday IYI")
    for ad in ADAYLAR:
        print(f"\n{ad}")
        for b in BLOKLAR:
            v = ony[ad][b]
            print(
                f"  {b:6} {v['dMSE_agr']:>+9.5f}  GA95 [{v['GA95'][0]:+.5f},{v['GA95'][1]:+.5f}]"
                f"  P(+)={v['P_pozitif']:.3f}  trafo={v['trafo']}"
            )
        o = ony[ad]["UC_BLOK_ORT"]
        ba = ony[ad]["bloklar_arasi"]
        lb = ony[ad]["LB_cevrimi"]
        print(
            f"  ORT    {o['dMSE_agr']:>+9.5f}  bloklar arasi sd={ba['sd']:.5f} "
            f"se={ba['se_ort']:.5f} isaret={ba['isaret']}"
        )
        print(
            f"  -> test dMSE {lb['test_dMSE']:+.5f} | LB oran1.0 {lb['beklenen_LB_oran1.0']:.5f}"
            f" | oran0.5 {lb['beklenen_LB_oran0.5']:.5f}"
        )
    print(f"\nkayit: {yol}")


if __name__ == "__main__":
    main()
