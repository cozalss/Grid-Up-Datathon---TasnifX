"""D14 -- GONDERIM ENVANTERI: hangi dosya kac puan alir (Kaggle'a HICBIR SEY GONDERMEZ).

Her aday dosya icin v102 tabanina gore d = lp(dosya) - lp(v102) ve Q = ||d||^2/n
hesaplanir. Skor iki ucla verilir:

    TEZ DOGRUYSA (delta = uygulanan adim, yani kappa=1 ve L=Q):
        MSE = m0 - Q            en iyi hal
    TEZ YANLISSA (delta = 0, yon gercekle hic hizali degil):
        MSE = m0 + Q            en kotu hal
    BASA BAS: gercek hizalanma kappa_gercek = 0.5 -> skor degismez

Olculmus dosyalarda gercek skor yazilir; tahmine gerek yok.
"""

from __future__ import annotations

import json

import numpy as np
from ortak import CIK, N_TEST, SUB, hizala, lp

TABAN = "tuketim_v102_kappa_optimum.csv"
M0 = 1.011091
IKINCI, LIDER = 1.00041, 0.99138

OLCULMUS = {
    "tuketim_v2.csv": 1.16143,
    "tuketim_v7.csv": 1.16922,
    "tuketim_v15.csv": 1.03910,
    "tuketim_v16.csv": 1.06605,
    "tuketim_v18.csv": 1.03370,
    "tuketim_v25_hedge.csv": 1.04820,
    "tuketim_v27_v18hedge.csv": 1.03362,
    "tuketim_v30_buzme.csv": 1.02639,
    "tuketim_v44_v27yeni.csv": 1.03053,
    "tuketim_v46_gun.csv": 1.02448,
    "tuketim_v47_eskison.csv": 1.01750,
    "tuketim_v50_nihai30.csv": 1.01686,
    "tuketim_v55_gunolcek.csv": 1.01591,
    "tuketim_v67_c1335_olay.csv": 1.01548,
    "tuketim_v73_soguk_gun160.csv": 1.01538,
    "tuketim_v79_S3.csv": 1.01556,
    "tuketim_v80_optimum.csv": 1.01341,
    "tuketim_v81_sicak08.csv": 1.01429,
    "tuketim_v83_sicak_optimum.csv": 1.01318,
    "tuketim_v101_hepsi.csv": 1.01614,
    "tuketim_v102_kappa_optimum.csv": 1.00553,
    "tuketim_v109_birlesik.csv": 1.01818,
}

# Degerlendirilecek HAZIR adaylar: (dosya, sinif, not)
ADAYLAR = [
    ("tuketim_p11_dalga_soguk.csv", "PROB", "05-11 dalgasi, train gecmisi YOK"),
    ("tuketim_p14_dalga_gecmisli.csv", "PROB", "05-11 dalgasi, train gecmisi VAR"),
    ("tuketim_v103_gram2.csv", "GUVENLI", "rank-17 Gram optimumu"),
    ("tuketim_v110_grupb_optimum.csv", "RISKLI", "grup B kappa 0.459->1.405"),
    ("tuketim_v111_donuscu.csv", "ZEHIRLI", "organik delta, %85 toplu kohort"),
    ("tuketim_v112_donuscu_yarim.csv", "ZEHIRLI", "ayni hata, yarim genlik"),
    ("tuketim_v113_toplu_prob.csv", "YEDEK", "dar toplu kohort (dalga probu daha iyi)"),
    ("tuketim_v114_organik_prob.csv", "YEDEK", "dar organik kohort"),
    ("tuketim_v89_genis_taban.csv", "ZEHIRLI", "olu trafo tezi curudu (s1)"),
    ("tuketim_v88_olu_taban.csv", "ZEHIRLI", "olu trafo tezi curudu"),
    ("tuketim_v87_olu_izole.csv", "ZEHIRLI", "olu trafo tezi curudu"),
    ("tuketim_sota_v1.csv", "ZEHIRLI", "olu trafo tezi curudu"),
]


def konum(r: float) -> str:
    if r < LIDER:
        return "1."
    if r < IKINCI:
        return "2."
    if r < 1.00553:
        return "3.+"
    return "3."


def main() -> int:
    taban = lp(hizala(TABAN))
    rap: dict = {"taban": TABAN, "m0": M0, "ikinci": IKINCI, "lider": LIDER, "adaylar": {}}

    print(f"taban {TABAN}  m0={M0:.6f}  RMSLE 1.00553")
    print(f"2. sira {IKINCI}   lider {LIDER}\n")
    print(
        f"{'dosya':34s} {'sinif':8s} {'Q':>9s} {'TEZ DOGRU':>10s} "
        f"{'TEZ YANLIS':>11s} {'basabas':>8s}"
    )
    print("-" * 88)
    for ad, sinif, aciklama in ADAYLAR:
        yol = SUB / ad
        if not yol.exists():
            print(f"{ad:34s} {sinif:8s}  DOSYA YOK")
            continue
        d = lp(hizala(ad)) - taban
        Q = float(d @ d / N_TEST)
        iyi = float(np.sqrt(max(M0 - Q, 1e-9)))  # kappa_gercek = 1
        kotu = float(np.sqrt(M0 + Q))  # kappa_gercek = 0
        nz = np.abs(d) > 1e-12
        rap["adaylar"][ad] = {
            "sinif": sinif,
            "aciklama": aciklama,
            "Q": Q,
            "degisen_satir": int(nz.sum()),
            "tez_dogruysa_RMSLE": iyi,
            "tez_dogruysa_konum": konum(iyi),
            "tez_yanlissa_RMSLE": kotu,
            "basa_bas_gercek_kappa": 0.5,
        }
        print(f"{ad:34s} {sinif:8s} {Q:9.6f} {iyi:10.5f} {kotu:11.5f} {0.5:8.2f}   {konum(iyi)}")

    print("\n" + "=" * 88)
    print("PLAN: p11 + p14 problari -> HAK3'te kappa* birlesimi")
    p11 = rap["adaylar"]["tuketim_p11_dalga_soguk.csv"]
    p14 = rap["adaylar"]["tuketim_p14_dalga_gecmisli.csv"]
    pay11 = p11["degisen_satir"] / N_TEST
    pay14 = p14["degisen_satir"] / N_TEST
    print(f"  pay_soguk {pay11:.5f}   pay_gecmisli {pay14:.5f}")
    print("\n  HAK3 sonucu = m0 - (d_soguk^2*pay_soguk + d_gecmisli^2*pay_gecmisli)")
    print(f"\n  {'|d_soguk|':>10s} {'|d_gecmisli|':>13s} {'kazanc':>9s} {'RMSLE':>9s}  konum")
    hak3 = {}
    for a in (0.0, 0.10, 0.15, 0.20, 0.25, 0.30):
        for b in (0.0, 0.10, 0.20, 0.30):
            kz = a * a * pay11 + b * b * pay14
            r = float(np.sqrt(max(M0 - kz, 1e-9)))
            hak3[f"{a:.2f}/{b:.2f}"] = {"kazanc": kz, "RMSLE": r, "konum": konum(r)}
            print(f"  {a:10.2f} {b:13.2f} {kz:9.6f} {r:9.5f}  {konum(r)}")
    rap["HAK3_senaryolari"] = hak3
    esik = float(np.sqrt((M0 - IKINCI**2) / (pay11 + pay14)))
    esik_l = float(np.sqrt((M0 - LIDER**2) / (pay11 + pay14)))
    rap["esikler"] = {"ikinci_icin_ortak_delta": esik, "lider_icin_ortak_delta": esik_l}
    print(f"\n  iki blokta da |delta| >= {esik:.4f} ise 2. SIRA")
    print(f"  iki blokta da |delta| >= {esik_l:.4f} ise 1. SIRA")

    rap["olculmus"] = OLCULMUS
    (CIK / "d14_envanter.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nyazildi: experiments/donuscu/d14_envanter.json")
    print("KAGGLE'A HICBIR SEY GONDERILMEDI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
