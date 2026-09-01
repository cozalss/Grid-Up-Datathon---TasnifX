"""p23-E: KAYMA ADAYLARI (Adim 4) + DURUST BEKLENTI (Adim 5).

Muhasebe:
  * Adim 3 kaymasi (Delta_cat) SAF cat soguk uzmanina gore olculdu.
  * Ama p21 zinciri parti satirlarini cat'ten delta_zincir kadar ZATEN
    asagi cekiyor (3/1/1 harmani + seviye katmanlari). Cift sayimi onlemek
    icin: kayma_net = Delta_cat - delta_zincir.
  * delta_zincir satir duzeyinde: diff = log_p21 - log_cat_soguk;
    tarih-eslesmeli (parti ort - diger-soguk ort), tarih >= 2026-05-11.

Uygulama: log1p(yeni) = log1p(p21) + t * kayma_net, yalniz SOGUK-PARTI
satirlarinda (108.253), t in {0.5, 0.75, 1.0}.

Kullanim: p23_e_kayma_aday.py --kayma <Delta_cat>
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
PK = os.path.join(KOK, "experiments/model29/p_kalici")
AC = os.path.join(PK, "aday_csv")
JSON_YOL = os.path.join(PK, "p23_parti.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kayma",
        type=float,
        required=True,
        help="Adim 3 kaymasi (kopru - kontrol, tarih-eslesmeli)",
    )
    ap.add_argument("--kayma_ga", type=float, nargs=2, default=None, help="kaymanin GA95 alt/ust")
    ap.add_argument("--kur", action="store_true", help="aday CSV'leri yaz")
    ar = ap.parse_args()

    test = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"), dtype={"tanim": str})
    p21 = pd.read_csv(os.path.join(AC, "p21_harman311_olu50.csv"))
    assert (p21["id"].to_numpy() == test["id"].to_numpy()).all()
    soguk = np.load(os.path.join(AC, "p06_test_soguk_maske.npy"))
    m_parti = np.load(os.path.join(AC, "p23_parti_soguk_maske.npy"))
    aile = np.load(os.path.join(AC, "p06_test_soguk_aile.npy"))

    log_p21 = np.log1p(p21["tuketim"].to_numpy())
    log_cat = np.full(len(test), np.nan)
    log_cat[soguk] = aile[:, 0]

    # --- delta_zincir: tarih-eslesmeli diff-in-diff (soguk satirlar, >= 05-11)
    S = pd.DataFrame(
        {
            "tarih": test["tarih"].to_numpy(),
            "diff": log_p21 - log_cat,
            "parti": m_parti,
        }
    )[soguk]
    S = S[S["tarih"] >= "2026-05-11"]
    kp = S[S["parti"]].groupby("tarih")["diff"].agg(["mean", "size"])
    dg = S[~S["parti"]].groupby("tarih")["diff"].mean()
    ortak = kp.index.intersection(dg.index)
    delta_gun = kp.loc[ortak, "mean"] - dg.loc[ortak]
    w = kp.loc[ortak, "size"]
    delta_zincir = float((delta_gun * w).sum() / w.sum())

    kayma_net = ar.kayma - delta_zincir
    pay = float(m_parti.sum()) / len(test)

    sonuc = {
        "girdi_kayma_Delta_cat": ar.kayma,
        "delta_zincir_p21_eksi_cat_tarih_eslesmeli": round(delta_zincir, 4),
        "kayma_net": round(kayma_net, 4),
        "parti_soguk_satir": int(m_parti.sum()),
        "parti_soguk_pay": round(pay, 4),
    }

    # --- durust beklenti: dMSE = pay * (2*c*kayma_net - c^2), c = t*kayma_net
    def beklenti(k):
        b = {}
        for t in (0.5, 0.75, 1.0):
            c = t * k
            dmse = pay * (2 * c * k - c * c)  # pozitif = iyilesme
            b[f"t{t:.2f}"] = {
                "satir_ici_dMSE": round(2 * c * k - c * c, 5),
                "test_dMSE": round(dmse, 5),
                "LB_delta_tasima_1.0": round(-dmse / 2, 5),
                "LB_delta_tasima_0.5": round(-dmse / 4, 5),
            }
        return b

    sonuc["beklenti_nokta"] = beklenti(kayma_net)
    if ar.kayma_ga:
        sonuc["beklenti_GA95"] = {
            "kayma_alt": beklenti(ar.kayma_ga[0] - delta_zincir),
            "kayma_ust": beklenti(ar.kayma_ga[1] - delta_zincir),
        }

    if ar.kur:
        dosyalar = []
        for t, ad in [(0.5, "t050"), (0.75, "t075"), (1.0, "t100")]:
            yeni_log = log_p21.copy()
            yeni_log[m_parti] = yeni_log[m_parti] + t * kayma_net
            yeni = np.expm1(yeni_log)
            assert np.isfinite(yeni).all() and (yeni >= 0).all()
            assert len(yeni) == 714688
            # parti-disi satirlar birebir ayni mi
            assert np.array_equal(yeni[~m_parti], p21["tuketim"].to_numpy()[~m_parti])
            cikti = os.path.join(AC, f"p23_parti_{ad}.csv")
            pd.DataFrame({"id": test["id"], "tuketim": yeni}).to_csv(cikti, index=False)
            dosyalar.append(f"aday_csv/p23_parti_{ad}.csv")
        sonuc["aday_dosyalar"] = dosyalar
        sonuc["dogrulama"] = (
            "714688 satir, id sirasi birebir, NaN/negatif yok, parti-disi degismedi"
        )

    R = {}
    if os.path.exists(JSON_YOL):
        with open(JSON_YOL, encoding="utf-8") as fh:
            R = json.load(fh)
    R["adim4_kayma_adayi"] = sonuc
    with open(JSON_YOL, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, ensure_ascii=False)
    print(json.dumps(sonuc, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
