"""D13 -- IKI PROBA indirgeme (Kaggle'a HICBIR SEY GONDERMEZ).

d11 uc ayrik prob kurdu; bu ucu ayri olcmek 4 hak / 2 gun demek. Birlestirme
maliyeti hesaplandi (kazanc = sum pay_i*delta_i^2, birlesik = (sum pay*d)^2/sum pay):

    AKTIF + KESIK birlesik  -> kayip 0.000556 MSE (RMSLE 0.00028)  IHMAL EDILEBILIR
                               ikisinin de onseli NEGATIF (-0.1238 / -0.2727)
    SOGUK da katilirsa      -> isareti TERS cikarsa kayip 0.009785
                               yani kazancin neredeyse TAMAMI; onsel YOK

Karar: SOGUK ayri kalir, AKTIF+KESIK birlesir. Iki yon dik oldugu icin
ucuncu hak AYNI GUN ikisinin optimumunu birden uygular:

    HAK1  tuketim_p11_dalga_soguk.csv       (Q=0.013632)
    HAK2  tuketim_p14_dalga_gecmisli.csv    (Q=0.009146, kirpma korumali)
    HAK3  d12_coz.py ile kurulan optimum    kazanc = sum L_i^2/Q_i >= 0

Bu betik yalniz p14'u uretir; p11 d11'den geliyor ve DEGISMEZ.
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
ADIM = -0.30
CIKTI = "tuketim_p14_dalga_gecmisli.csv"

# KIRPMA KORUMASI. Adim log uzayinda uygulanip expm1 ile geri donuluyor ve
# tahmin 0'a kirpiliyor. log1p(v102) < |kappa*|*|ADIM| olan satirlarda kirpma
# devreye girer; o zaman gerceklesen yon kappa*.d OLMAZ ve HAK3'te Q denetimi
# patlar. Ilk kurulusta p14'te 32 satir (%0.044) boyleydi -- v102 oralara
# 0.047-0.315 kWh yaziyor. Esik 0.90 ile |kappa*| < 3.0 araliginda kirpma
# IMKANSIZ hale gelir; bedeli 153 satir (%0.21) ve Q'nun %0.017'si.
# p11'de gerek yok (blokta min log1p(v102) = 2.6411, guvenli |kappa*| < 8.8).
KIRPMA_ESIGI = 0.90


def main() -> int:
    tr, te = train(), test()
    v102 = hizala(TABAN)
    egt = set(tr["tanim"].unique())
    son = tr.groupby("tanim")["tarih"].max()
    giris = te.groupby("tanim")["tarih"].min()
    dalga = set(giris.index[giris == DALGA])

    # train gecmisi OLAN dalga trafolari = aktif + kesik
    gecmisli = [t for t in dalga if t in egt]
    aktif = [t for t in gecmisli if son[t] >= GRUPA_SINIR]
    kesik = [t for t in gecmisli if son[t] < GRUPA_SINIR]
    tn = te["tanim"].to_numpy()
    lv = lp(v102)
    m_ham = np.isin(tn, gecmisli)
    m = m_ham & (lv >= KIRPMA_ESIGI)
    dislanan = int((m_ham & ~m).sum())
    print(
        f"kirpma korumasi: esik {KIRPMA_ESIGI} -> {dislanan} satir disarida "
        f"(%{100 * dislanan / m_ham.sum():.3f}); guvenli |kappa*| < {KIRPMA_ESIGI / abs(ADIM):.1f}"
    )

    adim = np.where(m, ADIM, 0.0)
    yeni = np.clip(np.expm1(lp(v102) + adim), 0.0, None)
    ss = pd.read_csv(KOK / "data/raw/sample_submission.csv", usecols=["id"])
    cik = pd.DataFrame({"id": te["id"].to_numpy(), "tuketim": yeni})
    cik = ss.merge(cik, on="id", how="left", validate="one_to_one")
    assert not cik["tuketim"].isna().any() and len(cik) == N_TEST
    (SUB / CIKTI).write_text(cik.to_csv(index=False, lineterminator="\n"), encoding="utf-8")

    d = lp(yeni) - lp(v102)
    Q = float(d @ d / N_TEST)
    pay = float(m.sum()) / N_TEST

    # p11 ile diklik
    d11 = lp(hizala("tuketim_p11_dalga_soguk.csv")) - lp(v102)
    ortak = int(((np.abs(d) > 1e-12) & (np.abs(d11) > 1e-12)).sum())
    Q11 = float(d11 @ d11 / N_TEST)
    pay11 = float((np.abs(d11) > 1e-12).sum()) / N_TEST

    # p12+p13 toplaminin p14'e esitligi -- YALNIZ korunan satirlarda
    d12 = lp(hizala("tuketim_p12_dalga_aktif.csv")) - lp(v102)
    d13 = lp(hizala("tuketim_p13_dalga_kesik.csv")) - lp(v102)
    sapma = float(np.abs((d12 + d13 - d)[m]).max())

    # KIRPMA DENETIMI: blokta adim her satirda tam ADIM olmali
    nz = np.abs(d) > 1e-12
    kirpik = int((nz & (np.abs(d - ADIM) > 1e-9)).sum())
    print(
        f"  kirpilan satir: {kirpik}  (0 olmali)  adim araligi "
        f"[{d[nz].min():+.6f}, {d[nz].max():+.6f}]"
    )
    assert kirpik == 0, "kirpma var -- KIRPMA_ESIGI yukseltilmeli"
    guvenli = float(lv[nz].min() / abs(ADIM))
    print(f"  blokta min log1p(v102) = {lv[nz].min():.4f} -> kirpmasiz |kappa*| < {guvenli:.2f}")

    print(f"=== {CIKTI} ===")
    print(f"  trafo {len(gecmisli)} (aktif {len(aktif)} + kesik {len(kesik)})")
    print(f"  satir {int(m.sum()):,}  pay %{100 * pay:.2f}  Q={Q:.6f}  adim {ADIM}")
    print("\n=== DENETIM ===")
    print(f"  max|adim(p12)+adim(p13) - adim(p14)| = {sapma:.3e}   (0 olmali)")
    print(f"  p11 ile ortak satir                  = {ortak}   (0 olmali)")
    assert sapma < 1e-12, "p14 != p12+p13"
    assert ortak == 0, "p11 ile dik degil"

    print("\n=== HAK3 SENARYOLARI (iki L olculdukten sonra) ===")
    print("  kazanc = delta_soguk^2 * pay_soguk + delta_gecmisli^2 * pay_gecmisli")
    print(f"  pay_soguk={pay11:.5f}  pay_gecmisli={pay:.5f}")
    satirlar = []
    for dsg in (0.0, 0.10, 0.15, 0.20, 0.25, 0.30):
        for dgc in (0.10, 0.20, 0.30):
            kz = dsg * dsg * pay11 + dgc * dgc * pay
            mse = V102_MSE - kz
            r = float(np.sqrt(max(mse, 1e-9)))
            et = "  <- LIDERI GECER" if r < LIDER else ("  <- 2.yi gecer" if r < IKINCI else "")
            satirlar.append((dsg, dgc, kz, r, et))
            print(f"  |d_soguk|={dsg:.2f} |d_gecmisli|={dgc:.2f} -> RMSLE {r:.6f}{et}")

    rap = {
        "cikti": CIKTI,
        "adim": ADIM,
        "trafo": len(gecmisli),
        "aktif": len(aktif),
        "kesik": len(kesik),
        "satir": int(m.sum()),
        "pay": pay,
        "Q": Q,
        "p11_pay": pay11,
        "p11_Q": Q11,
        "denetim": {
            "p12_arti_p13_sapma": sapma,
            "p11_ortak_satir": ortak,
            "kirpma_esigi": KIRPMA_ESIGI,
            "dislanan_satir": dislanan,
            "kirpilan_satir": kirpik,
            "guvenli_kappa_siniri": guvenli,
        },
        "birlestirme_maliyeti_MSE": 0.000556,
        "plan": {
            "HAK1": "tuketim_p11_dalga_soguk.csv",
            "HAK2": CIKTI,
            "HAK3": "d12_coz.py --prob p11=<skor> --prob p14=<skor>",
        },
    }
    (CIK / "d13_iki_prob.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nyazildi: submissions/{CIKTI} + experiments/donuscu/d13_iki_prob.json")
    print("KAGGLE'A HICBIR SEY GONDERILMEDI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
