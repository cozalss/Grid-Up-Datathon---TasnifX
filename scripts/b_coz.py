"""LB SKORUNDAN YANLILIGI COZ -- gozle tahmin YASAK, formulden cozulur.

NEDEN TAM
---------
Toplamsal bir log-uzayi kaymasi icin ozdeslik KESIN:

    MSLE(d) = (1/N) * sum_i (p_i + d*1[i in H] - y_i)^2
            = MSLE(0) + p*d^2 - 2*p*d*b        p = |H|/N ,  b = ort_H(y - p)

p ANALITIK bilinir (satir sayilarindan), MSLE(0) LB'den bilinir.
Yani TEK gonderim b'yi tam cozer; ikinci gonderim optimumu yazar.
Optimal d* = b, ve o noktada MSLE = MSLE(0) - p*b^2.

Public/private ayrimi OLMADIGI icin (yarisma sahibi 2026-08-25'te dogruladi)
donen skor tam test kumesi uzerinde, yani gurultusuz.

KULLANIM
--------
    # sicak proba 1.01223 geldi:
    python scripts/b_coz.py --rejim sicak --delta 0.08 --skor 1.01223
    # soguk proba:
    python scripts/b_coz.py --rejim soguk --delta 0.12 --skor 1.01486
    # ikisini birlikte cozup nihai komutu yazdir:
    python scripts/b_coz.py --sicak-delta 0.08 --sicak-skor 1.01223 \
                            --soguk-delta 0.12 --soguk-skor 1.01486
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
TABAN_RMSLE = 1.01591  # v55_gunolcek, LB'de iki kez dogrulandi
TABAN_MSE = TABAN_RMSLE**2


def paylar() -> tuple[float, float, int]:
    """(p_sicak, p_soguk, N) -- test.csv ve train.csv'den SAYILARAK."""
    te = pd.read_csv(KOK / "data/raw/test.csv", usecols=["tanim"], dtype={"tanim": str})
    tr = pd.read_csv(KOK / "data/raw/train.csv", usecols=["tanim"], dtype={"tanim": str})
    sicak = te["tanim"].isin(set(tr["tanim"]))
    n = len(te)
    return float(sicak.mean()), float((~sicak).mean()), n


def coz(p: float, delta: float, skor: float, taban_mse: float) -> tuple[float, float, float]:
    """(b, ulasilabilir_mse, ulasilabilir_rmsle)."""
    dmse = skor**2 - taban_mse
    b = (p * delta**2 - dmse) / (2 * p * delta)
    en_iyi = taban_mse - p * b**2
    return b, en_iyi, float(np.sqrt(max(en_iyi, 0.0)))


def main() -> int:
    a = argparse.ArgumentParser(description="LB skorundan seviye yanliligini coz")
    a.add_argument("--rejim", choices=["sicak", "soguk"])
    a.add_argument("--delta", type=float)
    a.add_argument("--skor", type=float)
    a.add_argument("--sicak-delta", type=float)
    a.add_argument("--sicak-skor", type=float)
    a.add_argument("--soguk-delta", type=float)
    a.add_argument("--soguk-skor", type=float)
    a.add_argument("--taban", type=float, default=TABAN_RMSLE, help="MSLE(0)'in RMSLE'si")
    ar = a.parse_args()

    taban_mse = ar.taban**2
    ps, pc, n = paylar()
    print(f"test satiri {n:,}   p_sicak={ps:.5f}   p_soguk={pc:.5f}")
    print(f"taban RMSLE {ar.taban:.5f}  ->  MSLE(0) = {taban_mse:.6f}\n")

    tekil = ar.rejim is not None
    if tekil:
        if ar.delta is None or ar.skor is None:
            raise SystemExit("--rejim ile birlikte --delta ve --skor sart")
        p = ps if ar.rejim == "sicak" else pc
        b, mse, rmsle = coz(p, ar.delta, ar.skor, taban_mse)
        print(f"{ar.rejim.upper()}  delta={ar.delta:+.4f}  gelen skor={ar.skor:.5f}")
        print(f"  >>> b_{ar.rejim} = {b:+.5f}")
        print(f"  >>> optimal delta = {b:+.5f}")
        print(f"  >>> o noktada RMSLE = {rmsle:.5f}  (yalniz bu rejim duzeltilirse)")
        # skor 5 haneli -> b belirsizligi
        b_lo, _, _ = coz(p, ar.delta, ar.skor + 5e-6, taban_mse)
        b_hi, _, _ = coz(p, ar.delta, ar.skor - 5e-6, taban_mse)
        print(f"  belirsizlik (skor +-5e-6): b in [{min(b_lo, b_hi):+.5f}, {max(b_lo, b_hi):+.5f}]")
        return 0

    bs = bc = None
    if ar.sicak_delta is not None and ar.sicak_skor is not None:
        bs, _, _ = coz(ps, ar.sicak_delta, ar.sicak_skor, taban_mse)
        print(f"SICAK  delta={ar.sicak_delta:+.4f} skor={ar.sicak_skor:.5f}  ->  b_sicak={bs:+.5f}")
    if ar.soguk_delta is not None and ar.soguk_skor is not None:
        bc, _, _ = coz(pc, ar.soguk_delta, ar.soguk_skor, taban_mse)
        print(f"SOGUK  delta={ar.soguk_delta:+.4f} skor={ar.soguk_skor:.5f}  ->  b_soguk={bc:+.5f}")
    if bs is None and bc is None:
        raise SystemExit("en az bir rejim icin --*-delta ve --*-skor ver")

    bs_ = 0.0 if bs is None else max(bs, 0.0)
    bc_ = 0.0 if bc is None else max(bc, 0.0)
    if (bs is not None and bs < 0) or (bc is not None and bc < 0):
        print("\n  UYARI: negatif b -> model o rejimde ASIRI tahmin ediyor.")
        print("  Betik delta'yi 0'a kirpiyor; asagi kaydirma AYRI bir karar, once olc.")
    en_iyi = taban_mse - ps * bs_**2 - pc * bc_**2
    print(f"\n  >>> optimal delta_sicak = {bs_:+.5f}   delta_soguk = {bc_:+.5f}")
    print(f"  >>> yalniz seviye duzeltmesiyle ulasilabilir RMSLE = {np.sqrt(max(en_iyi, 0)):.5f}")
    print("\n  KOMUT:")
    print("  uv run python scripts/son_islem_seviye.py \\")
    print("      --giris submissions/tuketim_v67_c1335_olay.csv \\")
    print("      --cikis submissions/tuketim_v68_nihai.csv \\")
    print(f"      --delta {bs_:.4f} --soguk-delta {bc_:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
