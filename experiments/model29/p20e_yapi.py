"""p20e: SEVIYESIZ (V1) adayin onyuklemesi ve trafo kirpma merdiveni.

p20d bulgusu: kazanc SEVIYE ve YAPI diye ayrildiginda YAPI bileseni UC
BLOKTA DA POZITIF (kis26 dahil), butun terslik SEVIYE bileseninde ve
seviye UC BLOKTA DA zarar veriyor. Gonderim dosyasinin seviye katmani
zaten LB'den cozulmus; dolayisiyla dogru aday SEVIYESIZ (V1) olandir.

Bu betik V1'i docs/79 EK'in dort kapisindan gecirir.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)

from p20_harman import AILE, BLOKLAR, HARMANLAR, SOGUK_PAY, TOHUMLAR, veri_kur  # noqa: E402

BETA = 0.60
N_ONY = 500
TABAN = "URETIM_cat"
ADAYLAR = ("ESIT", "ESKI_3_1_1")
MEVCUT_LB = 1.00115


def buz(r, beta):
    o = float(r.mean())
    return o + beta * (r - o)


def hata_seviyesiz(b, ad, tohum=None):
    """V1: aday harmani, ortalama ofseti TABANA esitlenmis."""

    def harman(w, t):
        if t is None:
            return sum(x * b["P"][a].mean(axis=0) for x, a in zip(w, AILE))
        i = TOHUMLAR.index(t)
        return sum(x * b["P"][a][i] for x, a in zip(w, AILE))

    r_t = harman(HARMANLAR[TABAN], tohum) - b["lguc"]
    if ad == TABAN:
        r = r_t
    else:
        r_a = harman(HARMANLAR[ad], tohum) - b["lguc"]
        r = r_a - r_a.mean() + r_t.mean()
    rr = buz(r, BETA)
    return b["lgy"] - np.maximum(rr + b["lguc"], 0.0)


def main() -> None:
    B, _ = veri_kur()
    R: dict = {
        "varyant": "V1 SEVIYESIZ (ortalama ofset TABANA esitlendi)",
        "beta": BETA,
        "n_onyukleme": N_ONY,
    }
    H = {ad: {b: hata_seviyesiz(B[b], ad) for b in BLOKLAR} for ad in (TABAN,) + ADAYLAR}

    ony: dict = {}
    for ad in ADAYLAR:
        ony[ad] = {}
        hepsi = []
        for b in BLOKLAR:
            w, tan = B[b]["w"], B[b]["tanim"]
            d2 = (H[TABAN][b] ** 2 - H[ad][b] ** 2) * w
            s = pd.Series(d2).groupby(tan)
            top, adet = s.sum().to_numpy(), s.size().to_numpy().astype("float64")
            rng = np.random.default_rng(70 + BLOKLAR.index(b))
            k = len(top)
            idx = rng.integers(0, k, size=(N_ONY, k))
            o = top[idx].sum(axis=1) / adet[idx].sum(axis=1)
            hepsi.append(o)
            # kirpma merdiveni
            ss = pd.Series(d2).groupby(tan).sum().sort_values()
            kirp = {}
            for K in (0, 5, 10, 25, 50):
                at = set(ss.index[:K]) | set(ss.index[len(ss) - K :]) if K else set()
                m = ~pd.Series(tan).isin(at).to_numpy()
                kirp[f"K={K}"] = round(float(d2[m].sum() / m.sum()), 6)
            # tohum bazinda
            toh = {}
            for t in TOHUMLAR:
                et = hata_seviyesiz(B[b], TABAN, t) ** 2 - hata_seviyesiz(B[b], ad, t) ** 2
                toh[str(t)] = round(float((w * et).mean()), 6)
            ony[ad][b] = {
                "dMSE_agr": round(float(top.sum() / adet.sum()), 6),
                "GA95": [
                    round(float(np.percentile(o, 2.5)), 6),
                    round(float(np.percentile(o, 97.5)), 6),
                ],
                "P_pozitif": round(float((o > 0).mean()), 4),
                "kirpma": kirp,
                "tohum": toh,
                "trafo": int(k),
            }
        oo = np.mean(hepsi, axis=0)
        v = [ony[ad][b]["dMSE_agr"] for b in BLOKLAR]
        g = float(np.mean(v))
        kk = {
            f"K={K}": round(float(np.mean([ony[ad][b]["kirpma"][f"K={K}"] for b in BLOKLAR])), 6)
            for K in (0, 5, 10, 25, 50)
        }
        ony[ad]["OZET"] = {
            "ORT_dMSE_agr": round(g, 6),
            "isaret": f"{sum(x > 0 for x in v)}/3",
            "GA95_blokici": [
                round(float(np.percentile(oo, 2.5)), 6),
                round(float(np.percentile(oo, 97.5)), 6),
            ],
            "P_pozitif": round(float((oo > 0).mean()), 4),
            "bloklar_arasi_se": round(float(np.std(v, ddof=1) / np.sqrt(3)), 6),
            "kirpma_ORT": kk,
            "tohum_isareti_9_hucre": f"{sum(1 for b in BLOKLAR for t in TOHUMLAR if ony[ad][b]['tohum'][str(t)] > 0)}/9",
            "test_dMSE": round(g * SOGUK_PAY, 6),
            "beklenen_LB_oran1.0": round(float(np.sqrt(max(MEVCUT_LB**2 - g * SOGUK_PAY, 0))), 5),
            "beklenen_LB_oran0.5": round(
                float(np.sqrt(max(MEVCUT_LB**2 - 0.5 * g * SOGUK_PAY, 0))), 5
            ),
            "MUHAFAZAKAR_K25_test_dMSE": round(kk["K=25"] * SOGUK_PAY, 6),
            "MUHAFAZAKAR_K25_LB_oran0.5": round(
                float(np.sqrt(max(MEVCUT_LB**2 - 0.5 * kk["K=25"] * SOGUK_PAY, 0))), 5
            ),
        }
    R["onyukleme"] = ony
    R["KAPILAR"] = {
        ad: {
            "a_gercek_blok_CV": True,
            "b_blok_disi_secim": "0 PARAMETRE -- harman adayi hicbir bloktan secilmedi",
            "c_isaret_tutarli": ony[ad]["OZET"]["isaret"],
            "d_tohumla_ayakta": ony[ad]["OZET"]["tohum_isareti_9_hucre"],
        }
        for ad in ADAYLAR
    }

    yol = os.path.join(BURA, "p_kalici", "p20_yapi.json")
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)

    print("V1 SEVIYESIZ -- onyukleme (trafo kumeli, 500), kohort agirlikli")
    for ad in ADAYLAR:
        print(f"\n{ad}")
        for b in BLOKLAR:
            v = ony[ad][b]
            print(
                f"  {b:6} {v['dMSE_agr']:>+9.5f} GA95 [{v['GA95'][0]:+.5f},{v['GA95'][1]:+.5f}]"
                f" P(+)={v['P_pozitif']:.3f}  kirpma "
                + " ".join(f"{k}={x:+.4f}" for k, x in v["kirpma"].items())
                + "  tohum "
                + " ".join(f"{x:+.4f}" for x in v["tohum"].values())
            )
        o = ony[ad]["OZET"]
        print(
            f"  ORT {o['ORT_dMSE_agr']:>+9.5f} isaret {o['isaret']} "
            f"tohum {o['tohum_isareti_9_hucre']} P(+)={o['P_pozitif']:.3f} "
            f"se(bloklar)={o['bloklar_arasi_se']:.5f}"
        )
        print(f"  kirpma ORT: " + " ".join(f"{k}={x:+.5f}" for k, x in o["kirpma_ORT"].items()))
        print(
            f"  -> test dMSE {o['test_dMSE']:+.5f}  LB oran1.0 {o['beklenen_LB_oran1.0']:.5f}"
            f" / oran0.5 {o['beklenen_LB_oran0.5']:.5f}"
            f"  | K=25 muhafazakar oran0.5 {o['MUHAFAZAKAR_K25_LB_oran0.5']:.5f}"
        )
    print(f"\nkayit: {yol}")


if __name__ == "__main__":
    main()
