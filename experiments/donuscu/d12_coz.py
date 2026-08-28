"""D12 -- prob skorlari olculdukten SONRA optimum birlesimi kurar.

Kullanim (skorlar Kaggle'dan OKUNUR, gonderim YAPILMAZ):
    python d12_coz.py --prob tuketim_p11_dalga_soguk.csv=1.01234 \
                      --prob tuketim_p14_dalga_gecmisli.csv=1.00789

Her prob i icin d_i = lp(prob_i) - lp(v102), Q_i = ||d_i||^2/n (dosyadan KESIN),
ve olculmus skordan
    L_i = (m0 + Q_i - m_i) / 2
Yonler DIK olmali (satir kumeleri kesismemeli; betik DENETLER). O zaman
birlesik optimum ayrisir:
    kappa*_i = L_i / Q_i        kazanc = sum L_i^2 / Q_i  >= 0 HER ZAMAN

Cikti: submissions/tuketim_v120_dalga_optimum.csv + on kayitli skor.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from ortak import CIK, KOK, N_TEST, SUB, hizala, lp, test

TABAN = "tuketim_v102_kappa_optimum.csv"
V102_MSE = 1.011091
IKINCI, LIDER = 1.00041, 0.99138
CIKTI = "tuketim_v120_dalga_optimum.csv"
SD_L = 7.14e-6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prob",
        action="append",
        required=True,
        metavar="DOSYA=SKOR",
        help="prob dosyasi ve olculmus LB skoru; birden fazla kez verilebilir",
    )
    ap.add_argument("--kirp", type=float, default=None, help="kappa* icin mutlak ust sinir")
    a = ap.parse_args()

    te = test()
    taban_lp = lp(hizala(TABAN))
    toplam_adim = np.zeros(N_TEST)
    rap: dict = {"taban": TABAN, "m0": V102_MSE, "problar": {}}
    kazanc = 0.0
    yonler = []

    for girdi in a.prob:
        dosya, _, skor_s = girdi.partition("=")
        if not skor_s:
            raise SystemExit(f"bicim hatasi: {girdi!r} -- DOSYA=SKOR bekleniyor")
        skor = float(skor_s)
        d = lp(hizala(dosya)) - taban_lp
        Q = float(d @ d / N_TEST)
        if Q <= 0:
            raise SystemExit(f"{dosya}: Q=0, taban ile ayni dosya")
        L = (V102_MSE + Q - skor**2) / 2.0
        kap = L / Q
        if a.kirp is not None:
            kap = float(np.clip(kap, -a.kirp, a.kirp))
        nz = np.abs(d) > 1e-12
        s = float(d[nz][0])
        # KIRPMA KORUMASI: kappa*.d uygulandiktan sonra tahmin negatife dusmemeli.
        # log1p(v102) + kappa*.s >= 0 olmali; aksi halde gerceklesen yon kappa*.d
        # OLMAZ ve on kayitli skor tutmaz.
        guvenli = float((taban_lp[nz] / abs(s)).min())
        if abs(kap) >= guvenli:
            raise SystemExit(
                f"{dosya}: kappa*={kap:+.4f} kirpma sinirini asiyor (|kappa*| < {guvenli:.3f}). "
                "Prob KIRPMA_ESIGI yukseltilerek yeniden kurulmali."
            )
        katki = L * L / Q
        kazanc += katki
        toplam_adim += kap * d
        yonler.append((dosya, nz))
        rap["problar"][dosya] = {
            "LB_skoru": skor,
            "Q": Q,
            "pay": round(float(nz.sum()) / N_TEST, 5),
            "adim_s": round(s, 4),
            "L": L,
            "L_SNR": round(abs(L) / SD_L, 1),
            "kappa*": kap,
            "cozulen_gercek_ofset": round(kap * s, 4),
            "kazanc_L2_bolu_Q": katki,
        }
        print(
            f"{dosya:34s} skor {skor:.5f}  Q={Q:.6f}  L={L:+.6f} (SNR {abs(L) / SD_L:.0f})  "
            f"kappa*={kap:+.4f}  gercek ofset={kap * s:+.4f}  kazanc={katki:.6f}"
        )

    print("\n=== DIKLIK DENETIMI ===")
    for i in range(len(yonler)):
        for j in range(i + 1, len(yonler)):
            ortak = int((yonler[i][1] & yonler[j][1]).sum())
            print(f"  {yonler[i][0]} & {yonler[j][0]}: ortak satir {ortak}")
            if ortak:
                raise SystemExit("yonler DIK DEGIL -- ayrisik optimum gecersiz")

    mse = V102_MSE - kazanc
    rmsle = float(np.sqrt(max(mse, 1e-9)))
    nerede = "LIDERI GECER" if rmsle < LIDER else ("2.yi gecer" if rmsle < IKINCI else "3. sira")
    print(
        f"\nTOPLAM kazanc {kazanc:.6f}   ON KAYITLI MSE {mse:.6f}   RMSLE {rmsle:.6f}  -> {nerede}"
    )
    if kazanc < 0:
        raise SystemExit("kazanc negatif olamaz -- girdi skorlarini kontrol et")

    yeni = np.clip(np.expm1(taban_lp + toplam_adim), 0.0, None)
    ss = pd.read_csv(KOK / "data/raw/sample_submission.csv", usecols=["id"])
    cik = pd.DataFrame({"id": te["id"].to_numpy(), "tuketim": yeni})
    cik = ss.merge(cik, on="id", how="left", validate="one_to_one")
    assert not cik["tuketim"].isna().any() and len(cik) == N_TEST
    (SUB / CIKTI).write_text(cik.to_csv(index=False, lineterminator="\n"), encoding="utf-8")

    dd = lp(yeni) - taban_lp
    Q_kur = float(dd @ dd / N_TEST)
    Q_bek = float(sum(r["kappa*"] ** 2 * r["Q"] for r in rap["problar"].values()))
    print(f"Q denetimi: kurulan {Q_kur:.6f}  beklenen {Q_bek:.6f}  fark {abs(Q_kur - Q_bek):.3e}")
    if abs(Q_kur - Q_bek) > 1e-9:
        raise SystemExit("Q tutmadi -- kirpma veya dosya uyusmazligi")
    rap["cikti"] = {
        "dosya": CIKTI,
        "Q_kurulan": Q_kur,
        "Q_beklenen": Q_bek,
        "ON_KAYITLI_MSE": mse,
        "ON_KAYITLI_RMSLE": rmsle,
        "toplam_kazanc": kazanc,
        "konum": nerede,
    }
    (CIK / "d12_coz.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nyazildi: submissions/{CIKTI}")
    print("KAGGLE'A HICBIR SEY GONDERILMEDI -- gonderim icin kullanicidan onay al.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
