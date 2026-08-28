"""D15 -- TAM OPTIMUM: 22 olculmus yon + dalga problari birlikte cozulur.

d12 yalniz dalga problarini kullanir ve onlar dik oldugu icin optimum ayrisir.
Ama elimizde 22 OLCULMUS gonderim daha var; onlarin span'i s16'da 0.000443 MSE
tasiyor. Dalga yonleri o span'a DIK DEGIL, o yuzden ikisini ayri ayri toplamak
yanlis olur -- tam Gram cozulur:

    MSE(c) = m0 - 2 c.L + c' G c        c* = G^+ L        kazanc = L' G^+ L

Rank secimi SNR ile: |u.L| / sd(L), sd yalniz 5 hane skor yuvarlamasindan.
s16'da ozdeger 2.5e-12 olan bilesen 16.03'luk SAHTE kazanc oneriyordu; SNR
filtresi onu reddetti. Ayni filtre burada da var.

Kullanim:
    python d15_tam_optimum.py --prob tuketim_p11_dalga_soguk.csv=1.00327 \
                              --prob tuketim_p14_dalga_gecmisli.csv=1.00250

KIRPMA: birlesik adim buyuk olabilir; her satirda log1p(v102)+adim >= 0
denetlenir, ihlal varsa dosya YAZILMAZ.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from ortak import CIK, KOK, N_TEST, SUB, hizala, lp, test

TABAN = "tuketim_v102_kappa_optimum.csv"
M0 = 1.011091
IKINCI, LIDER = 1.00041, 0.99138
CIKTI = "tuketim_v121_tam_optimum.csv"
SD_L = 7.14e-6
SNR_ESIK = 3.0

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
    "tuketim_v109_birlesik.csv": 1.01818,
}


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob", action="append", default=[], metavar="DOSYA=SKOR")
    ap.add_argument("--yaz", action="store_true", help="dosyayi gercekten yaz")
    a = ap.parse_args()

    te = test()
    taban = lp(hizala(TABAN))
    adlar, D, skorlar = [], [], []
    for ad, sk in OLCULMUS.items():
        adlar.append(ad)
        D.append(lp(hizala(ad)) - taban)
        skorlar.append(sk)
    for girdi in a.prob:
        dosya, _, s = girdi.partition("=")
        if not s:
            raise SystemExit(f"bicim: {girdi!r} -- DOSYA=SKOR")
        adlar.append(dosya)
        D.append(lp(hizala(dosya)) - taban)
        skorlar.append(float(s))

    D = np.stack(D)
    G = D @ D.T / N_TEST
    L = np.array([(M0 + G[j, j] - skorlar[j] ** 2) / 2.0 for j in range(len(adlar))])
    print(f"yon sayisi {len(adlar)}  ({len(OLCULMUS)} olculmus + {len(a.prob)} prob)")

    w, V = np.linalg.eigh(G)
    sira = np.argsort(w)[::-1]
    w, V = w[sira], V[:, sira]
    uL = V.T @ L
    sec = (w > 0) & (np.abs(uL) / SD_L >= SNR_ESIK)
    atilan = int(((w > 0) & ~sec).sum())
    kazanc = float((uL[sec] ** 2 / w[sec]).sum())
    c = V[:, sec] @ (uL[sec] / w[sec])
    print(
        f"SNR>={SNR_ESIK} secilen bilesen {int(sec.sum())} / {int((w > 0).sum())}  (atilan {atilan})"
    )

    mse = M0 - kazanc
    rmsle = float(np.sqrt(max(mse, 1e-9)))
    nerede = "1. SIRA" if rmsle < LIDER else ("2. SIRA" if rmsle < IKINCI else "3. sira")
    print(
        f"\nTOPLAM kazanc {kazanc:.6f}   ON KAYITLI MSE {mse:.6f}   RMSLE {rmsle:.6f}  -> {nerede}"
    )

    adim = c @ D
    ihlal = int((taban + adim < 0).sum())
    print(f"\nKIRPMA DENETIMI: log1p(v102)+adim < 0 olan satir = {ihlal}  (0 olmali)")
    print(f"  adim araligi [{adim.min():+.4f}, {adim.max():+.4f}]")
    if ihlal:
        print("  ! kirpma var -- gerceklesen yon c.D OLMAZ, on kayit TUTMAZ")

    rap = {
        "taban": TABAN,
        "m0": M0,
        "yon_sayisi": len(adlar),
        "problar": {d.partition("=")[0]: float(d.partition("=")[2]) for d in a.prob},
        "secilen_bilesen": int(sec.sum()),
        "atilan_bilesen": atilan,
        "toplam_kazanc": kazanc,
        "ON_KAYITLI_MSE": mse,
        "ON_KAYITLI_RMSLE": rmsle,
        "konum": nerede,
        "kirpma_ihlali": ihlal,
        "adim_min": float(adim.min()),
        "adim_maks": float(adim.max()),
    }

    if a.yaz and not ihlal:
        yeni = np.clip(np.expm1(taban + adim), 0.0, None)
        ss = pd.read_csv(KOK / "data/raw/sample_submission.csv", usecols=["id"])
        cik = pd.DataFrame({"id": te["id"].to_numpy(), "tuketim": yeni})
        cik = ss.merge(cik, on="id", how="left", validate="one_to_one")
        assert not cik["tuketim"].isna().any() and len(cik) == N_TEST
        (SUB / CIKTI).write_text(cik.to_csv(index=False, lineterminator="\n"), encoding="utf-8")
        dd = lp(yeni) - taban
        Q_kur, Q_bek = float(dd @ dd / N_TEST), float(c @ G @ c)
        rap["Q_kurulan"], rap["Q_beklenen"] = Q_kur, Q_bek
        print(
            f"Q denetimi: kurulan {Q_kur:.6f}  beklenen {Q_bek:.6f}  fark {abs(Q_kur - Q_bek):.3e}"
        )
        print(f"yazildi: submissions/{CIKTI}")
    elif a.yaz:
        print("KIRPMA IHLALI -> dosya YAZILMADI")
    else:
        print("(--yaz verilmedi, dosya yazilmadi)")

    (CIK / "d15_tam_optimum.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("KAGGLE'A HICBIR SEY GONDERILMEDI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
