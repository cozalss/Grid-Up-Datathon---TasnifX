"""p20c: kis26 tersligi GERCEK mi, birkac trafonun eseri mi? + test-benzeri
kohortun onyuklemesi.

p20b iki isaret buldu:
  1. kis26'nin ESIT aleyhine dMSE'sinin %104'u EN KOTU 5 TRAFOdan geliyor.
     (Projenin kendi curutme deseni: d04243f soguk-grup-kolon fikrini
      "kazancin %116'si tek trafodan" diye reddetmisti. Ayni mercek burada
      TERS yone, yani cat-tekil lehine isliyor.)
  2. Testin BASKIN kohortunda (>=650 kVA VE p_gun in (75,90]) kis26'nin
     isareti DONUYOR (+0,090 ESIT / +0,066 3/1/1) ve 3/1/1 UC BLOKTA DA
     pozitif. Ama alt kume kucuk -- onyukleme sart.

Bu betik ikisini de sinar.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)

from p20_harman import BLOKLAR, HARMANLAR, SOGUK_PAY, TOHUMLAR, hatalar, veri_kur  # noqa: E402

BETA = 0.60
N_ONY = 500
TABAN = "URETIM_cat"
ADAYLAR = ("ESIT", "ESKI_3_1_1")
MEVCUT_LB = 1.00115


def kumeli_onyukleme(d2: np.ndarray, tan: np.ndarray, tohum: int) -> dict:
    s = pd.Series(d2).groupby(tan)
    top, adet = s.sum().to_numpy(), s.size().to_numpy().astype("float64")
    n = float(adet.sum())
    rng = np.random.default_rng(tohum)
    k = len(top)
    idx = rng.integers(0, k, size=(N_ONY, k))
    o = top[idx].sum(axis=1) / adet[idx].sum(axis=1)
    return {
        "dMSE": round(float(top.sum() / n), 6),
        "GA95": [round(float(np.percentile(o, 2.5)), 6), round(float(np.percentile(o, 97.5)), 6)],
        "P_pozitif": round(float((o > 0).mean()), 4),
        "trafo": int(k),
        "n": int(n),
    }, o


def main() -> None:
    B, _ = veri_kur()
    HAT = {ad: {b: hatalar(B[b], HARMANLAR[ad], BETA) for b in BLOKLAR} for ad in HARMANLAR}
    R: dict = {"beta": BETA, "n_onyukleme": N_ONY}

    # ---- 1) TRAFO KIRPMA MERDIVENI (kohort agirlikli, tam blok)
    kirp: dict = {}
    for ad in ADAYLAR:
        kirp[ad] = {}
        for b in BLOKLAR:
            w, tan = B[b]["w"], B[b]["tanim"]
            d2 = (HAT[TABAN][b] ** 2 - HAT[ad][b] ** 2) * w
            s = pd.Series(d2).groupby(tan).sum().sort_values()
            sat = {}
            for K in (0, 1, 3, 5, 10, 25):
                at = set(s.index[:K]) | set(s.index[len(s) - K :]) if K else set()
                m = ~pd.Series(tan).isin(at).to_numpy()
                sat[f"K={K}"] = round(float(d2[m].sum() / m.sum()), 6)
            kirp[ad][b] = sat
    R["trafo_kirpma_merdiveni"] = kirp

    # ---- 2) TEST-BENZERI KOHORT, ONYUKLEMELI
    tb: dict = {}
    ony_ort: dict = {ad: [] for ad in ADAYLAR}
    for ad in ADAYLAR:
        tb[ad] = {}
        for b in BLOKLAR:
            guc, pg = B[b]["guc"], B[b]["pg"]
            m = (guc >= 650) & (pg > 75) & (pg <= 90)
            d2 = HAT[TABAN][b][m] ** 2 - HAT[ad][b][m] ** 2
            v, o = kumeli_onyukleme(d2, B[b]["tanim"][m], 40 + BLOKLAR.index(b))
            tb[ad][b] = v
            ony_ort[ad].append(o)
        oo = np.mean(ony_ort[ad], axis=0)
        g = float(np.mean([tb[ad][b]["dMSE"] for b in BLOKLAR]))
        vv = [tb[ad][b]["dMSE"] for b in BLOKLAR]
        td = g * SOGUK_PAY
        tb[ad]["UC_BLOK_ORT"] = {
            "dMSE": round(g, 6),
            "GA95_blokici": [
                round(float(np.percentile(oo, 2.5)), 6),
                round(float(np.percentile(oo, 97.5)), 6),
            ],
            "P_pozitif": round(float((oo > 0).mean()), 4),
            "isaret": f"{sum(x > 0 for x in vv)}/3",
            "bloklar_arasi_se": round(float(np.std(vv, ddof=1) / np.sqrt(3)), 6),
            "test_dMSE": round(td, 6),
            "beklenen_LB_oran1.0": round(float(np.sqrt(max(MEVCUT_LB**2 - td, 0))), 5),
            "beklenen_LB_oran0.5": round(float(np.sqrt(max(MEVCUT_LB**2 - 0.5 * td, 0))), 5),
        }
    R["test_benzeri_kohort_onyuklemeli"] = tb

    # ---- 3) kis26 EN KOTU TRAFOLARIN KIMLIGI
    b = "kis26"
    w, tan = B[b]["w"], B[b]["tanim"]
    d2 = (HAT[TABAN][b] ** 2 - HAT["ESIT"][b] ** 2) * w
    s = pd.Series(d2).groupby(tan).sum().sort_values()
    E = pd.read_parquet(
        os.path.join(
            os.path.dirname(os.path.dirname(BURA)), "data", "interim", "deney", "egitim.parquet"
        )
    )
    d = E[(E["_blok"] == b) & (E["soguk_mu"] == 1)]
    ozet = d.groupby("tanim").agg(
        guc=("guc", "first"),
        n=("tuketim", "size"),
        sifir_pay=("tuketim", lambda v: float((v <= 0).mean())),
        ort=("tuketim", "mean"),
        lok=("lokasyon", "first"),
    )
    kot = []
    for t in s.index[:8]:
        o = ozet.loc[t]
        kot.append(
            {
                "tanim": str(t),
                "dMSE_katkisi": round(float(s.loc[t] / len(d2)), 6),
                "guc": float(o.guc),
                "gun": int(o.n),
                "sifir_pay": round(float(o.sifir_pay), 3),
                "ort_tuketim": round(float(o.ort), 1),
                "lokasyon": str(o.lok),
            }
        )
    R["kis26_en_kotu_8_trafo"] = kot
    R["kis26_toplam_negatif_dMSE"] = round(float(s[s < 0].sum() / len(d2)), 6)
    R["kis26_net_dMSE"] = round(float(s.sum() / len(d2)), 6)

    yol = os.path.join(BURA, "p_kalici", "p20_kis26.json")
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)

    print("1) TRAFO KIRPMA MERDIVENI (agr dMSE; K = her uctan atilan trafo)")
    for ad in ADAYLAR:
        print(f"  {ad}")
        for b2 in BLOKLAR:
            print(f"    {b2:6} " + "  ".join(f"{k}={v:+.5f}" for k, v in kirp[ad][b2].items()))
    print("\n2) TEST-BENZERI KOHORT (>=650 kVA & p_gun in (75,90]), onyuklemeli")
    for ad in ADAYLAR:
        print(f"  {ad}")
        for b2 in BLOKLAR:
            v = tb[ad][b2]
            print(
                f"    {b2:6} dMSE {v['dMSE']:>+9.5f} GA95 [{v['GA95'][0]:+.5f},"
                f"{v['GA95'][1]:+.5f}] P(+)={v['P_pozitif']:.3f} n={v['n']} trafo={v['trafo']}"
            )
        o = tb[ad]["UC_BLOK_ORT"]
        print(
            f"    ORT    dMSE {o['dMSE']:>+9.5f}  isaret {o['isaret']}  "
            f"P(+)={o['P_pozitif']:.3f}  -> LB oran1.0 {o['beklenen_LB_oran1.0']:.5f}"
            f" / oran0.5 {o['beklenen_LB_oran0.5']:.5f}"
        )
    print(
        f"\n3) kis26 net dMSE {R['kis26_net_dMSE']:+.5f}, "
        f"negatif trafolarin toplami {R['kis26_toplam_negatif_dMSE']:+.5f}"
    )
    for k in kot[:5]:
        print("   ", k)
    print(f"\nkayit: {yol}")


if __name__ == "__main__":
    main()
