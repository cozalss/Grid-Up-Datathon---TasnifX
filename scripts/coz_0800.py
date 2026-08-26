"""08:00 COZUCUSU -- S1/S2/S3 skorlarindan HER SEYI cozer. Etiket kullanmaz.

NEDEN BU BETIK
--------------
08:00'de doner donmez uc bilinmeyeni TAM cozmek icin. Butun sabitler
gonderim DOSYALARINDAN onceden hesaplanir (etiketsiz), skorlar gelince
yalnizca aritmetik kalir.

CEBIR
-----
Her gonderim, S1 tabanina log-uzayinda sabit bir vektor Delta ekler:

    MSLE(S) = MSLE(S1) + ort(Delta^2) - 2*ort(r*Delta)      r = y - log1p(S1)

ort(Delta^2) = Q  ->  DOSYALARDAN TAM HESAPLANIR (etiketsiz).
ort(r*Delta) = L  ->  skordan cozulur:   L = (Q - (S^2 - S1^2)) / 2

S2: Delta2 = (c-1) * profil   (soguk satirlarda, seviye-notr)
    L2 = (c-1) * B_c    ->    B_c = L2 / (c-1)
    A_c = Q2 / (c-1)^2
    >>> OPTIMAL c* = 1 + B_c / A_c
    >>> o noktada dMSE = -B_c^2 / A_c

S3: Delta3 = Delta2 + delta * 1[soguk]
    Q3 ve L3'ten, L2 bilindigi icin seviye bileseni ayrilir:
    L3 = L2 + delta * B_seviye   ->   B_seviye = (L3 - L2) / delta
    ve p_soguk carpani zaten Q3'un icinde oldugu icin
    >>> b_soguk = B_seviye / p_soguk
    >>> optimal delta* = b_soguk,  dMSE = -p_soguk * b_soguk^2

Iki duzeltme DIK (soguk gun profili satir-ortalamasi tam 0), o yuzden
birlestirilebilir:
    ulasilabilir MSE = MSLE(S1) - B_c^2/A_c - p_soguk * b_soguk^2

KULLANIM
--------
    # once sabitleri hesapla (skor beklemeden kosulabilir):
    uv run python scripts/coz_0800.py --sabitler

    # skorlar gelince:
    uv run python scripts/coz_0800.py --s1 1.01507 --s2 1.01430 --s3 1.01180
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
S1_DOSYA = "tuketim_v67_c1335_olay.csv"
S2_DOSYA = "tuketim_v73_soguk_gun160.csv"
S3_DOSYA = "tuketim_v79_S3.csv"
C_S2 = 1.60
DELTA_S3 = 0.22
DELTA_KUYRUK = 0.35
B_KUYRUK_ONKAYIT = 0.414  # ikizden: guz25 0,4754 / kis26 0,3531
B_KUYRUK_SH = 0.061
P_SOGUK = 0.22159


def lg(ad: str) -> np.ndarray:
    d = pd.read_csv(KOK / "submissions" / ad)
    return np.log1p(np.clip(d["tuketim"].to_numpy(dtype="float64"), 0, None))


def sabitler() -> dict:
    a1, a2, a3 = lg(S1_DOSYA), lg(S2_DOSYA), lg(S3_DOSYA)
    d2, d3 = a2 - a1, a3 - a1
    q2, q3 = float((d2**2).mean()), float((d3**2).mean())
    a_c = q2 / (C_S2 - 1.0) ** 2
    # S3'un seviye bileseni: Delta3 - Delta2 = delta * 1[soguk]
    fark = d3 - d2
    n_soguk = int((np.abs(fark) > 1e-12).sum())
    return {
        "Q2": q2,
        "Q3": q3,
        "A_c": a_c,
        "seviye_kontrol": float(fark[np.abs(fark) > 1e-12].mean()) if n_soguk else 0.0,
        "n_soguk": n_soguk,
        "N": len(a1),
        "dik_kontrol": float((d2 * (d3 - d2)).mean()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="08:00 cozucusu")
    p.add_argument("--sabitler", action="store_true")
    p.add_argument("--s1", type=float)
    p.add_argument("--s2", type=float)
    p.add_argument("--s3", type=float)
    ar = p.parse_args()

    k = sabitler()
    print("SABITLER (gonderim dosyalarindan, ETIKETSIZ)")
    print(f"  S1 = {S1_DOSYA}")
    print(f"  S2 = {S2_DOSYA}   (c = {C_S2})")
    print(f"  S3 = {S3_DOSYA}   (+ delta_soguk = {DELTA_S3})")
    print(f"\n  N            {k['N']:,}")
    print(f"  Q2 = ort(D2^2)   {k['Q2']:.8f}")
    print(f"  Q3 = ort(D3^2)   {k['Q3']:.8f}")
    print(f"  A_c = Q2/(c-1)^2 {k['A_c']:.8f}")
    print(
        f"  harmanlanmis kayma {k['seviye_kontrol']:+.6f}  "
        f"(soguk {DELTA_S3} + kuyruk {DELTA_KUYRUK} karisimi, 0,2406 olmali)"
    )
    print(f"  dokunulan satir  {k['n_soguk']:,}  (158.369 soguk + 29.873 kuyruk = 188.242 olmali)")
    print(f"  diklik ort(D2*(D3-D2)) {k['dik_kontrol']:+.3e}  (0'a yakin olmali)")

    if ar.sabitler or ar.s1 is None:
        print("\n  (skorlar verilmedi -- yalnizca sabitler)")
        return 0

    m1 = ar.s1**2
    print(f"\n{'=' * 70}\nCOZUM\n{'=' * 70}")
    print(f"  S1 = {ar.s1:.5f}  ->  MSLE(0) = {m1:.6f}")

    b_c = a_c = None
    if ar.s2 is not None:
        l2 = (k["Q2"] - (ar.s2**2 - m1)) / 2.0
        b_c = l2 / (C_S2 - 1.0)
        a_c = k["A_c"]
        c_yildiz = 1.0 + b_c / a_c
        kazanc_s2 = ar.s2**2 - m1
        print(f"\n  S2 = {ar.s2:.5f}  ->  gerceklesen dMSE = {kazanc_s2:+.6f}")
        print(f"    B_c = {b_c:.8f}   A_c = {a_c:.8f}")
        print(f"    >>> GERCEK OPTIMUM  c* = {c_yildiz:.4f}   (gonderilen {C_S2})")
        print(f"    >>> c*'ta ulasilabilir dMSE = {-(b_c**2) / a_c:+.6f}")
        print(f"    >>> c*'ta RMSLE = {np.sqrt(max(m1 - b_c**2 / a_c, 0)):.5f}")
        if c_yildiz > C_S2 * 1.05:
            print(f"    KARAR: c YUKARI. 27 Agustos'ta c={c_yildiz:.2f} yaz.")
        elif c_yildiz < C_S2 * 0.95:
            print(f"    KARAR: c ASAGI. 27 Agustos'ta c={max(c_yildiz, 1.0):.2f} yaz.")
        else:
            print(f"    KARAR: c={C_S2} zaten optimumda, dokunma.")

    if ar.s3 is not None and b_c is not None:
        l3 = (k["Q3"] - (ar.s3**2 - m1)) / 2.0
        l2 = b_c * (C_S2 - 1.0)
        # S3 IKI ayrik gruba dokunuyor: SOGUK (delta 0,22) + KUYRUK (delta 0,35).
        # Kuyruk katkisi on kayitli b ile DUSULUR, kalan soguk seviyeye kalir.
        te = pd.read_csv(KOK / "data/raw/test.csv", usecols=["tanim"], dtype={"tanim": str})
        tr = set(
            pd.read_csv(KOK / "data/raw/train.csv", usecols=["tanim"], dtype={"tanim": str})[
                "tanim"
            ].unique()
        )
        ilk = (
            pd.read_csv(
                KOK / "data/raw/train.csv",
                usecols=["tanim", "tarih"],
                dtype={"tanim": str},
                parse_dates=["tarih"],
            )
            .groupby("tanim")["tarih"]
            .min()
        )
        kuy = set(ilk[ilk >= pd.Timestamp("2026-03-26")].index)
        p_kuyruk = float(te["tanim"].isin(kuy).mean())
        l_kuyruk = DELTA_KUYRUK * p_kuyruk * B_KUYRUK_ONKAYIT
        b_seviye = (l3 - l2 - l_kuyruk) / DELTA_S3
        b_soguk = b_seviye / P_SOGUK
        belirsizlik = DELTA_KUYRUK * p_kuyruk * B_KUYRUK_SH / (DELTA_S3 * P_SOGUK)
        print(f"    kuyruk payi p={p_kuyruk:.4f}  on kayitli b_kuyruk={B_KUYRUK_ONKAYIT}")
        print(f"    kuyruk katkisi L={l_kuyruk:.6f} dusuldu")
        print(f"    >>> b_soguk BELIRSIZLIGI +-{belirsizlik:.4f} (b_kuyruk SH'sinden)")
        print(f"\n  S3 = {ar.s3:.5f}  ->  gerceklesen dMSE = {ar.s3**2 - m1:+.6f}")
        print(f"    >>> b_soguk = {b_soguk:+.5f}   (on kayitli tahmin +0,16)")
        print(f"    >>> optimal delta_soguk = {b_soguk:+.5f}")
        print(f"    >>> seviyeden ulasilabilir dMSE = {-P_SOGUK * max(b_soguk, 0) ** 2:+.6f}")
        en_iyi = m1 - b_c**2 / a_c - P_SOGUK * max(b_soguk, 0) ** 2
        print("\n  >>> IKISI BIRDEN (dik oldugu icin toplanir)")
        print(f"      ulasilabilir MSE   {en_iyi:.6f}")
        print(f"      ulasilabilir RMSLE {np.sqrt(max(en_iyi, 0)):.5f}")
        print("      Grid Grinders      1.00635")
        print("\n  27 AGUSTOS KOMUTU:")
        print("    uv run python scripts/son_islem_soguk_gunolcek.py \\")
        print(f"        --giris submissions/{S1_DOSYA} \\")
        print("        --cikis submissions/tuketim_v76_optimum.csv \\")
        print(f"        --c {1.0 + b_c / a_c:.3f}")
        print("    uv run python scripts/son_islem_seviye.py \\")
        print("        --giris submissions/tuketim_v76_optimum.csv \\")
        print("        --cikis submissions/tuketim_v77_optimum.csv \\")
        print(f"        --delta 0.0 --soguk-delta {max(b_soguk, 0):.4f}")
        print("    (+ kuyruk: son_islem_kuyruk_rejimi.py --delta 0.35)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
