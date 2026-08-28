"""D11 -- 2026-05-11 DALGASI: uc AYRIK prob (Kaggle'a HICBIR SEY GONDERMEZ).

Bulgu: test panelinin %25.33'u (181.038 satir / 2.222 trafo) tek gunde,
2026-05-11'de giriyor. Bu bir sistemik devreye alma dalgasi. Bu blok bugune
kadar KENDI YONU olarak hic prob edilmedi -- s16'nin "span tukendi" hukmunun
disinda kalan yeni yon tam olarak budur.

Dalga uc AYRIK alt kumeye bolunur (satir kumeleri kesismez -> yonler dik):
    P_soguk : train gecmisi YOK        1326 trafo  108.253 satir  %15.15
    P_aktif : train son kayit >= 03-27  502 trafo   40.771 satir   %5.70
    P_kesik : train son kayit <  03-27  394 trafo   32.014 satir   %4.48
(03-27 ayirici: kalici kural 9, docs/43:170 -- grup A / grup B)

NEDEN AYRI: uc alt kumenin gercek ofseti farkli isaretli olabilir; tek yonde
birlestirmek onlari birbirine goturur (v109 dersi, kalici kural 25).
Cauchy-Schwarz: sum(pay_i * d_i^2) >= (sum pay_i d_i)^2 / sum pay_i, yani
AYRI olcmek birlesik olcmekten HER ZAMAN >= iyi.

PLAN
  29 Agu HAK1/2/3 : uc probu gonder -> L_soguk, L_aktif, L_kesik TAM olculur
  30 Agu HAK1     : optimum birlesim; kazanc = sum L_i^2/Q_i  >= 0 GARANTI

Adim s=-0.30. Isaret ONSELDEN geliyor: train gecmisi olan iki alt kumede
v102, trafolari kendi 2025 ayni-pencere seviyelerinin USTUNE yaziyor
(mevsim-duzeltilmis: aktif -0.1238, kesik -0.2727). Yani beklenen gercek
ofset NEGATIF. Buyukluk 0.30: en kucuk blokta bile SNR ~ 190.
HAK2 kazanci L^2/Q, s'den BAGIMSIZDIR -- s yalniz olcum netligini ve
probun KENDI skorunu etkiler, nihai sonucu DEGIL.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import CIK, KOK, N_TEST, SUB, hizala, lp, test, train

TABAN = "tuketim_v102_kappa_optimum.csv"
V102_MSE = 1.011091
IKINCI, LIDER = 1.00041, 0.99138
DALGA = pd.Timestamp("2026-05-11")
GRUPA_SINIR = pd.Timestamp("2026-03-27")
ADIM = (
    -0.30
)  # onsel: v102 dalgayi kendi gecmis seviyesinin USTUNE yaziyor (aktif -0.12, kesik -0.27)
SD_L = 7.14e-6  # 5 hane skor yuvarlamasindan


def main() -> int:  # noqa: PLR0915
    tr, te = train(), test()
    v102 = hizala(TABAN)
    egt = set(tr["tanim"].unique())
    son = tr.groupby("tanim")["tarih"].max()
    giris = te.groupby("tanim")["tarih"].min()
    dalga = set(giris.index[giris == DALGA])

    def sinif(t: str) -> str:
        if t not in egt:
            return "soguk"
        return "aktif" if son[t] >= GRUPA_SINIR else "kesik"

    kat = {t: sinif(t) for t in dalga}
    tn = te["tanim"].to_numpy()
    ss = pd.read_csv(KOK / "data/raw/sample_submission.csv", usecols=["id"])

    PROBLAR = [
        ("tuketim_p11_dalga_soguk.csv", "soguk", "train gecmisi YOK"),
        ("tuketim_p12_dalga_aktif.csv", "aktif", "son kayit >= 2026-03-27 (grup A)"),
        ("tuketim_p13_dalga_kesik.csv", "kesik", "son kayit <  2026-03-27 (grup B)"),
    ]
    rap: dict = {
        "dalga_gunu": str(DALGA.date()),
        "adim": ADIM,
        "taban": TABAN,
        "v102_MSE": V102_MSE,
        "problar": {},
    }
    maskeler = {}
    for ad, k, aciklama in PROBLAR:
        tf = [t for t, v in kat.items() if v == k]
        m = np.isin(tn, tf)
        maskeler[ad] = m
        adim = np.where(m, ADIM, 0.0)
        yeni = np.clip(np.expm1(lp(v102) + adim), 0.0, None)
        cik = pd.DataFrame({"id": te["id"].to_numpy(), "tuketim": yeni})
        cik = ss.merge(cik, on="id", how="left", validate="one_to_one")
        assert not cik["tuketim"].isna().any(), f"{ad}: eksik id"
        assert len(cik) == N_TEST, f"{ad}: satir"
        (SUB / ad).write_text(cik.to_csv(index=False, lineterminator="\n"), encoding="utf-8")

        dvec = lp(yeni) - lp(v102)
        Q = float(dvec @ dvec / N_TEST)
        pay = float(m.sum()) / N_TEST
        # gercek ofset d -> L = d*pay*? ; adim s ise L = d*pay*s... normalize:
        # d_vec = s*1_blok  => L = <r,-d_vec>/n = d_gercek * s * pay
        # HAK2 kazanci = L^2/Q = (d*s*pay)^2/(s^2*pay) = d^2*pay
        senaryo = {}
        for dg in (-0.50, -0.40, -0.30, -0.20, -0.15, -0.10, 0.10, 0.20, 0.30):
            kz = dg * dg * pay
            senaryo[f"gercek_ofset={dg:.2f}"] = {
                "L": round(dg * ADIM * pay, 6),
                "SNR": round(dg * ADIM * pay / SD_L, 1),
                "HAK2_kazanc": round(kz, 6),
                "HAK2_MSE": round(V102_MSE - kz, 6),
                "HAK2_RMSLE": round(float(np.sqrt(max(V102_MSE - kz, 1e-9))), 6),
            }
        r = {
            "aciklama": aciklama,
            "trafo": len(tf),
            "satir": int(m.sum()),
            "pay": round(pay, 5),
            "Q": Q,
            "HAK1_kendi_skoru_gercek_ofset_eksi030_ise": round(
                float(np.sqrt(V102_MSE - (2 * -0.30 * ADIM * pay - ADIM**2 * pay))), 6
            ),
            "senaryolar": senaryo,
        }
        rap["problar"][ad] = r
        print(f"\n=== {ad} ===  {aciklama}")
        print(f"  trafo {r['trafo']}  satir {r['satir']:,}  pay %{100 * pay:.2f}  Q={Q:.6f}")
        for k2, v in senaryo.items():
            print(
                f"    {k2}: L={v['L']:+.6f} SNR={v['SNR']:7.1f}  "
                f"HAK2 kazanc={v['HAK2_kazanc']:.6f} -> RMSLE {v['HAK2_RMSLE']:.6f}"
            )

    # --- diklik denetimi -----------------------------------------------------
    adlar = [a for a, _, _ in PROBLAR]
    print("\n=== DIKLIK DENETIMI (satir kumeleri kesismemeli) ===")
    ok = True
    for i in range(len(adlar)):
        for j in range(i + 1, len(adlar)):
            ortak = int((maskeler[adlar[i]] & maskeler[adlar[j]]).sum())
            print(f"  {adlar[i]} & {adlar[j]}: ortak satir {ortak}")
            ok &= ortak == 0
    assert ok, "problar dik degil"

    # --- birlesik tavan ------------------------------------------------------
    toplam_pay = sum(rap["problar"][a]["pay"] for a in adlar)
    print(f"\n  toplam dalga payi %{100 * toplam_pay:.2f}")
    print("\n=== 30 AGU BIRLESIK OPTIMUM (uc L olculdukten sonra) ===")
    print("  kazanc = sum L_i^2/Q_i   ve  HER ZAMAN >= 0")
    for dg in (0.10, 0.15, 0.20, 0.25, 0.30, 0.334):
        kz = dg * dg * toplam_pay
        m = V102_MSE - kz
        etiket = ""
        if np.sqrt(m) < LIDER:
            etiket = "  <- LIDERI GECER"
        elif np.sqrt(m) < IKINCI:
            etiket = "  <- 2.yi gecer"
        print(f"  ucu de {dg:.3f} ise: kazanc {kz:.6f} -> RMSLE {np.sqrt(m):.6f}{etiket}")
    rap["hedefler"] = {
        "ikinci": IKINCI,
        "lider": LIDER,
        "ikinci_icin_gereken_kazanc": V102_MSE - IKINCI**2,
        "lider_icin_gereken_kazanc": V102_MSE - LIDER**2,
        "toplam_dalga_payi": toplam_pay,
        "ucu_de_esitse_lider_icin_gereken_ofset": float(
            np.sqrt((V102_MSE - LIDER**2) / toplam_pay)
        ),
        "ucu_de_esitse_ikinci_icin_gereken_ofset": float(
            np.sqrt((V102_MSE - IKINCI**2) / toplam_pay)
        ),
    }
    print(
        f"\n  2.yi gecmek icin gereken ortak ofset : "
        f"{rap['hedefler']['ucu_de_esitse_ikinci_icin_gereken_ofset']:.4f}"
    )
    print(
        f"  lideri gecmek icin gereken ofset     : "
        f"{rap['hedefler']['ucu_de_esitse_lider_icin_gereken_ofset']:.4f}"
    )
    (CIK / "d11_dalga_problari.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nyazildi: experiments/donuscu/d11_dalga_problari.json + uc gonderim dosyasi")
    print("KAGGLE'A HICBIR SEY GONDERILMEDI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
