"""D16 -- ANA BLOK PROBU (Kaggle'a HICBIR SEY GONDERMEZ).

d11/d13 dalgayi (testin %25.33'u) prob etti. Ama blok haritasi cikarilinca
cok daha buyuk ve HIC SORULMAMIS bir blok gorundu:

    PANEL-BASI x aktif (grup A)   3.926 trafo  469.671 satir  %65.72

Kaldirac karsilastirmasi -- 2. sirayi tek basina almak icin gereken |delta|:
    ana blok      %65.72  ->  0.125
    dalga-soguk   %15.15  ->  0.260
Yarisi yetiyor.

NEDEN SIFIR OLMAYABILIR: LB ile cozulmus sicak sabiti (+0.024860) TUM sicak
rejim uzerinden cozuldu; bu blok sicagin ~%84'u, yani sabit buyuk olcude ona
gore ayarli ve artik kucuk beklenir. AMA bu kesit panel-giris gunu x grup
eksenindedir, rejim ekseniyle CAPRAZDIR -- rejim sabiti bu ayrimi goremez.
Blok o kadar buyuk ki delta=0.05 bile 0.00164 kazanc demek.

ADIM: s = -0.15 (dalga problarindaki -0.30 degil). Blok cok buyuk oldugu icin
kucuk adim da bol SNR veriyor (delta=0.05'te bile SNR ~ 690) ve kirpma riski
dusuyor. HAK2 kazanci L^2/Q, s'den BAGIMSIZDIR.

KIRPMA KORUMASI: log1p(v102) < |kappa*|*|s| olan satirlar cikarilir.
Esik 0.45 -> |kappa*| < 3.0 araliginda kirpma imkansiz.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import CIK, KOK, N_TEST, SUB, hizala, lp, test, train

TABAN = "tuketim_v102_kappa_optimum.csv"
M0 = 1.011091
IKINCI, LIDER = 1.00041, 0.99138
PANEL_BASI = pd.Timestamp("2026-04-01")
GRUPA_SINIR = pd.Timestamp("2026-03-27")
ADIM = -0.15
KIRPMA_ESIGI = 0.45  # |kappa*| < 3.0
CIKTI = "tuketim_p15_ana_blok.csv"
SD_L = 7.14e-6


def main() -> int:
    tr, te = train(), test()
    v102 = hizala(TABAN)
    egt = set(tr["tanim"].unique())
    son = tr.groupby("tanim")["tarih"].max()
    giris = te.groupby("tanim")["tarih"].min()

    hedef = [
        t for t in giris.index if giris[t] == PANEL_BASI and t in egt and son[t] >= GRUPA_SINIR
    ]
    tn = te["tanim"].to_numpy()
    lv = lp(v102)
    m_ham = np.isin(tn, hedef)
    m = m_ham & (lv >= KIRPMA_ESIGI)
    dislanan = int((m_ham & ~m).sum())
    print(f"hedef trafo {len(hedef)}   ham satir {int(m_ham.sum()):,}")
    print(
        f"kirpma korumasi: esik {KIRPMA_ESIGI} -> {dislanan} satir disarida "
        f"(%{100 * dislanan / m_ham.sum():.3f}); guvenli |kappa*| < {KIRPMA_ESIGI / abs(ADIM):.1f}"
    )

    adim = np.where(m, ADIM, 0.0)
    yeni = np.clip(np.expm1(lv + adim), 0.0, None)
    ss = pd.read_csv(KOK / "data/raw/sample_submission.csv", usecols=["id"])
    cik = pd.DataFrame({"id": te["id"].to_numpy(), "tuketim": yeni})
    cik = ss.merge(cik, on="id", how="left", validate="one_to_one")
    assert not cik["tuketim"].isna().any() and len(cik) == N_TEST
    (SUB / CIKTI).write_text(cik.to_csv(index=False, lineterminator="\n"), encoding="utf-8")

    d = lp(yeni) - lv
    Q = float(d @ d / N_TEST)
    nz = np.abs(d) > 1e-12
    pay = float(nz.sum()) / N_TEST
    kirpik = int((nz & (np.abs(d - ADIM) > 1e-9)).sum())
    print(f"\n=== {CIKTI} ===")
    print(f"  satir {int(nz.sum()):,}  pay %{100 * pay:.2f}  Q={Q:.6f}  adim {ADIM}")
    print(
        f"  kirpilan satir {kirpik} (0 olmali)  adim araligi [{d[nz].min():+.6f}, {d[nz].max():+.6f}]"
    )
    assert kirpik == 0, "kirpma var -- KIRPMA_ESIGI yukseltilmeli"

    # dalga problariyla diklik
    print("\n=== DIKLIK (dalga problariyla kesismemeli) ===")
    for baska in ("tuketim_p11_dalga_soguk.csv", "tuketim_p14_dalga_gecmisli.csv"):
        db = lp(hizala(baska)) - lv
        ortak = int((nz & (np.abs(db) > 1e-12)).sum())
        print(f"  {baska}: ortak satir {ortak}")
        assert ortak == 0, f"{baska} ile dik degil"

    print("\n=== HAK1 KARAR KARTI ===")
    print(f"  L=0 (blok olu) -> skor tam {np.sqrt(M0 + Q):.5f}")
    print(f"\n  {'gelen skor':>11s} {'delta':>7s} {'kazanc':>9s} {'HAK2 sonucu':>12s}  yorum")
    kart = {}
    for dg in (0.00, 0.03, 0.05, 0.08, 0.10, 0.125, 0.15, 0.20, 0.25):
        L = dg * abs(ADIM) * pay
        skor = float(np.sqrt(M0 + Q - 2 * L))
        kaz = L * L / Q
        r = float(np.sqrt(M0 - kaz))
        yorum = (
            "OLU"
            if kaz < 0.0015
            else ("zayif" if kaz < 0.004 else ("IYI" if r >= IKINCI else "2. SIRA"))
        )
        kart[f"{dg:.3f}"] = {"skor": skor, "kazanc": kaz, "HAK2_RMSLE": r, "yorum": yorum}
        print(f"  {skor:11.5f} {-dg:+7.3f} {kaz:9.6f} {r:12.5f}  {yorum}")

    rap = {
        "cikti": CIKTI,
        "adim": ADIM,
        "kirpma_esigi": KIRPMA_ESIGI,
        "trafo": len(hedef),
        "satir": int(nz.sum()),
        "dislanan_satir": dislanan,
        "pay": pay,
        "Q": Q,
        "L_sifir_skoru": float(np.sqrt(M0 + Q)),
        "ikinci_icin_gereken_delta": float(np.sqrt((M0 - IKINCI**2) / pay)),
        "lider_icin_gereken_delta": float(np.sqrt((M0 - LIDER**2) / pay)),
        "karar_karti": kart,
    }
    print(f"\n  2. sira icin gereken |delta|: {rap['ikinci_icin_gereken_delta']:.4f}")
    print(f"  1. sira icin gereken |delta|: {rap['lider_icin_gereken_delta']:.4f}")
    (CIK / "d16_ana_blok.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nyazildi: submissions/{CIKTI} + experiments/donuscu/d16_ana_blok.json")
    print("KAGGLE'A HICBIR SEY GONDERILMEDI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
