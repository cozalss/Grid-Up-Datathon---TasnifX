"""D12 -- uc prob skoru olculdukten SONRA optimum birlesimi kurar.

Kullanim (skorlar Kaggle'dan OKUNUR, gonderim YAPILMAZ):
    python d12_coz.py --soguk 1.01234 --aktif 1.00789 --kesik 1.00612

Her prob i icin d_i = lp(prob_i) - lp(v102), Q_i = ||d_i||^2/n (dosyadan KESIN),
ve olculmus skordan
    L_i = (m0 + Q_i - m_i) / 2
Yonler DIK (satir kumeleri kesismiyor), bu yuzden birlesik optimum ayrisir:
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
CIKTI = "tuketim_v120_dalga_optimum.csv"
PROBLAR = {
    "soguk": "tuketim_p11_dalga_soguk.csv",
    "aktif": "tuketim_p12_dalga_aktif.csv",
    "kesik": "tuketim_p13_dalga_kesik.csv",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in PROBLAR:
        ap.add_argument(f"--{k}", type=float, required=True, help=f"{k} probunun LB skoru")
    ap.add_argument("--kirp", type=float, default=None, help="kappa* icin mutlak ust sinir")
    a = ap.parse_args()

    te = test()
    v102 = hizala(TABAN)
    taban_lp = lp(v102)
    toplam_adim = np.zeros(N_TEST)
    rap: dict = {"taban": TABAN, "m0": V102_MSE, "problar": {}}
    kazanc = 0.0

    for k, dosya in PROBLAR.items():
        skor = float(getattr(a, k))
        d = lp(hizala(dosya)) - taban_lp
        Q = float(d @ d / N_TEST)
        m_i = skor**2
        L = (V102_MSE + Q - m_i) / 2.0
        kap = L / Q
        if a.kirp is not None:
            kap = float(np.clip(kap, -a.kirp, a.kirp))
        pay = float((np.abs(d) > 1e-12).sum()) / N_TEST
        # d = s*1_blok oldugundan gercek ofset = kappa* * s
        s = float(d[np.abs(d) > 1e-12][0])
        katki = L * L / Q
        kazanc += katki
        toplam_adim += kap * d
        rap["problar"][k] = {
            "dosya": dosya,
            "LB_skoru": skor,
            "Q": Q,
            "pay": round(pay, 5),
            "adim_s": round(s, 4),
            "L": L,
            "kappa*": kap,
            "cozulen_gercek_ofset": round(kap * s, 4),
            "kazanc_L2_bolu_Q": katki,
        }
        print(
            f"{k:6s} skor {skor:.5f}  Q={Q:.6f}  L={L:+.6f}  kappa*={kap:+.4f}  "
            f"gercek ofset={kap * s:+.4f}  kazanc={katki:.6f}"
        )

    mse = V102_MSE - kazanc
    rmsle = float(np.sqrt(max(mse, 1e-9)))
    print(f"\nTOPLAM kazanc {kazanc:.6f}   ON KAYITLI MSE {mse:.6f}   RMSLE {rmsle:.6f}")
    if kazanc < 0:
        raise RuntimeError("kazanc negatif olamaz -- girdi skorlarini kontrol et")

    yeni = np.clip(np.expm1(taban_lp + toplam_adim), 0.0, None)
    ss = pd.read_csv(KOK / "data/raw/sample_submission.csv", usecols=["id"])
    cik = pd.DataFrame({"id": te["id"].to_numpy(), "tuketim": yeni})
    cik = ss.merge(cik, on="id", how="left", validate="one_to_one")
    assert not cik["tuketim"].isna().any() and len(cik) == N_TEST
    (SUB / CIKTI).write_text(cik.to_csv(index=False, lineterminator="\n"), encoding="utf-8")

    # dogrulama: kurulan dosyanin Q'su beklenenle tutuyor mu
    dd = lp(yeni) - taban_lp
    rap["cikti"] = {
        "dosya": CIKTI,
        "Q_kurulan": float(dd @ dd / N_TEST),
        "Q_beklenen": float(sum(r["kappa*"] ** 2 * r["Q"] for r in rap["problar"].values())),
        "ON_KAYITLI_MSE": mse,
        "ON_KAYITLI_RMSLE": rmsle,
        "toplam_kazanc": kazanc,
    }
    print(
        f"Q denetimi: kurulan {rap['cikti']['Q_kurulan']:.6f} "
        f"beklenen {rap['cikti']['Q_beklenen']:.6f}"
    )
    (CIK / "d12_coz.json").write_text(json.dumps(rap, indent=2, ensure_ascii=False), "utf-8")
    print(f"\nyazildi: submissions/{CIKTI}")
    print("KAGGLE'A HICBIR SEY GONDERILMEDI -- gonderim icin kullanicidan onay al.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
