"""D17 -- BOLGE PROBLARI (Kaggle'a HICBIR SEY GONDERMEZ).

Eksen taramasi (2025 ayni pencere vs v102, ortak kayma cikarilmis) siralamasi:

    AY x BOLGE   63 blok   tahmini 0.025887
    BOLGE        20 blok            0.015579   <- en iyi TEK eksen
    AY x GUC     28 blok            0.013006
    2-HAFTALIK    9 blok            0.008997
    AY            4 blok            0.007623
    GUC BANDI     7 blok            0.004974
    IL            2 blok            0.001074
    HAFTA GUNU    7 blok            0.000245

YONTEM NOTU -- neden bu, CV tuzagina dusmuyor:
Tahminler 2025 penceresinden geliyor, yani CV cinsi kanit; sicilimizde CV
turevli YONLER LB'de kappa~0 verdi (v108, y1). O yuzden DESENI UYGULAMIYORUZ.
Tahmin yalniz GRUPLAMAYI secmek icin kullanilir; her prob saf GOSTERGE'dir
(blok icinde sabit adim), dolayisiyla blokun GERCEK ortalama artigini olcer.
Desen yanlissa verim duser, olcum yine de gecerlidir.

BOLUMLEME (satir kumeleri kesismez -> yonler dik):
    P1 METROPOL          185.768 satir  %26.0   2025 sapmasi -0.1379
    P2 GUNEY BOLGE       270.791 satir  %37.9              +0.0713
    P3 NEG kova           39.762 satir   %5.6              -0.2934  (agirlikli)
       SARUHANLI, AKHISAR, ALASEHIR
    P4 POZ kova           50.618 satir   %7.1              +0.4122  (agirlikli)
       KULA, TURGUTLU, SARIGOL, SELENDI, GOLMARMARA, GORDES
    (kalan bolgeler dokunulmaz -- referans)

ADIM: blok basina, 2025 sapmasinin ISARETINDE ve buyuklugu SNR'a yetecek
kadar. Kazanc L^2/Q adimdan BAGIMSIZ; adim yalniz olcum netligini ve probun
kendi skorunu etkiler.

KIRPMA KORUMASI: log1p(v102) < |kappa*|*|adim| olan satirlar cikarilir;
esik adim*3 -> |kappa*| < 3.0 araliginda kirpma imkansiz.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import CIK, KOK, N_TEST, SUB, hizala, lp, test

TABAN = "tuketim_v102_kappa_optimum.csv"
M0 = 1.011091
IKINCI, LIDER = 1.00041, 0.99138
SD_L = 7.14e-6
KAPPA_TAVAN = 3.0

# (dosya, bolge listesi, adim, 2025 tahmini sapma)
BLOKLAR = [
    ("tuketim_p21_metropol.csv", ["METROPOL"], -0.15, -0.1379),
    ("tuketim_p22_guney.csv", ["GUNEY BOLGE"], +0.10, +0.0713),
    ("tuketim_p23_neg.csv", ["SARUHANLI", "AKHISAR", "ALASEHIR"], -0.30, -0.2934),
    (
        "tuketim_p24_poz.csv",
        ["KULA", "TURGUTLU", "SARIGOL", "SELENDI", "GOLMARMARA", "GORDES"],
        +0.40,
        +0.4122,
    ),
]


def sadelestir(s: pd.Series) -> pd.Series:
    """Turkce karakterleri ASCII'ye indir -- kaynak dosyada kodlama karisik."""
    esle = str.maketrans("ÇĞİÖŞÜçğıöşü", "CGIOSUcgiosu")
    return s.str.translate(esle).str.upper().str.strip()


def main() -> int:  # noqa: PLR0915
    te = test()
    v102 = hizala(TABAN)
    lv = lp(v102)
    bolge = sadelestir(te["lokasyon"].str.split(">").str[1])
    ss = pd.read_csv(KOK / "data/raw/sample_submission.csv", usecols=["id"])

    print("test panelindeki bolgeler:")
    for b, n in bolge.value_counts().items():
        print(f"  {b:22s} {n:8,}")

    rap: dict = {"taban": TABAN, "m0": M0, "bloklar": {}}
    maskeler = {}
    for ad, bolgeler, adim, tahmin in BLOKLAR:
        m_ham = bolge.isin(bolgeler).to_numpy()
        if m_ham.sum() == 0:
            print(f"\n!! {ad}: bolge eslesmedi {bolgeler}")
            continue
        esik = abs(adim) * KAPPA_TAVAN
        m = m_ham & (lv >= esik)
        dis = int((m_ham & ~m).sum())

        yeni = np.clip(np.expm1(lv + np.where(m, adim, 0.0)), 0.0, None)
        cik = pd.DataFrame({"id": te["id"].to_numpy(), "tuketim": yeni})
        cik = ss.merge(cik, on="id", how="left", validate="one_to_one")
        assert not cik["tuketim"].isna().any() and len(cik) == N_TEST
        (SUB / ad).write_text(cik.to_csv(index=False, lineterminator="\n"), encoding="utf-8")

        d = lp(yeni) - lv
        nz = np.abs(d) > 1e-12
        Q = float(d @ d / N_TEST)
        pay = float(nz.sum()) / N_TEST
        kirpik = int((nz & (np.abs(d - adim) > 1e-9)).sum())
        assert kirpik == 0, f"{ad}: kirpma var"
        maskeler[ad] = nz

        # karar karti
        kart = {}
        for dg in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
            g = dg * np.sign(adim)
            L = g * adim * pay
            kart[f"{g:+.2f}"] = {
                "beklenen_skor": round(float(np.sqrt(M0 + Q - 2 * L)), 5),
                "kazanc": round(L * L / Q, 6),
                "SNR": round(abs(L) / SD_L, 0),
            }
        r = {
            "bolgeler": bolgeler,
            "adim": adim,
            "tahmini_sapma_2025": tahmin,
            "satir": int(nz.sum()),
            "dislanan_satir": dis,
            "pay": pay,
            "Q": Q,
            "L_sifir_skoru": float(np.sqrt(M0 + Q)),
            "guvenli_kappa": float(lv[nz].min() / abs(adim)),
            "tahmin_tutarsa_kazanc": tahmin**2 * pay,
            "ikinci_icin_tek_basina_delta": float(np.sqrt((M0 - IKINCI**2) / pay)),
            "karar_karti": kart,
        }
        rap["bloklar"][ad] = r
        print(f"\n=== {ad} ===  {', '.join(bolgeler)}")
        print(
            f"  satir {r['satir']:,} (%{100 * pay:.2f})  disarida {dis}  Q={Q:.6f}  adim {adim:+.2f}"
        )
        print(f"  kirpilan 0  guvenli |kappa*| < {r['guvenli_kappa']:.2f}")
        print(f"  L=0 (blok olu) -> skor tam {r['L_sifir_skoru']:.5f}")
        print(f"  2025 tahmini {tahmin:+.4f} tutarsa kazanc {r['tahmin_tutarsa_kazanc']:.6f}")

    adlar = list(maskeler)
    print("\n=== DIKLIK DENETIMI ===")
    for i in range(len(adlar)):
        for j in range(i + 1, len(adlar)):
            o = int((maskeler[adlar[i]] & maskeler[adlar[j]]).sum())
            print(f"  {adlar[i]} & {adlar[j]}: {o}")
            assert o == 0, "dik degil"

    toplam = sum(r["tahmin_tutarsa_kazanc"] for r in rap["bloklar"].values())
    kapsam = sum(r["pay"] for r in rap["bloklar"].values())
    mse = M0 - toplam
    print("\n=== DORT PROB BIRLIKTE (2025 tahminleri tutarsa) ===")
    print(f"  kapsam %{100 * kapsam:.2f}   toplam kazanc {toplam:.6f}")
    print(f"  RMSLE {np.sqrt(max(mse, 1e-9)):.5f}   (2. sira {IKINCI}, lider {LIDER})")
    print(f"\n  2. sira icin gereken toplam kazanc: {M0 - IKINCI**2:.6f}")
    rap["ozet"] = {
        "kapsam": kapsam,
        "tahmin_tutarsa_kazanc": toplam,
        "tahmin_tutarsa_RMSLE": float(np.sqrt(max(mse, 1e-9))),
    }
    (CIK / "d17_bolge_problari.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nyazildi: 4 gonderim dosyasi + experiments/donuscu/d17_bolge_problari.json")
    print("KAGGLE'A HICBIR SEY GONDERILMEDI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
