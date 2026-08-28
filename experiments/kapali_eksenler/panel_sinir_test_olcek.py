"""PANEL SINIR -- TEST NUFUSUNA OLCEKLEME.

Blok kazanci dogrudan teste tasinamaz: sinir satirlarinin PAYI farkli.

  * SOGUK bloklarda giris payi %2,3-4,1; testte soguk giris cok daha seyrek
    (2.024 soguk trafonun cogu 2026-05-11'de TEK gun giriyor).
  * CIKIS testte buyuk olcude GOZLENEMEZ: bir trafonun son test satiri
    2026-07-31 ise bu pencere kenaridir, cikis degil. Yalnizca pencere
    ICINDE kaybolanlar bilinir.

Bu betik satir basina kazanci cikarir ve TEST sinir sayimiyla olcekler.

  test dMSE = (blok kazanci * blok_n / sinir_n) * (test_sinir_n / 714688)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
CIK = Path(__file__).resolve().parent
GUN = pd.Timedelta(days=1)
TEST_BAS = pd.Timestamp("2026-04-01")
TEST_SON = pd.Timestamp("2026-07-31")
N_TEST = 714_688


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
        encoding="utf-8",
    )
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
        encoding="utf-8",
    )
    sicak_set = set(tr["tanim"].unique())

    ort = pd.concat([tr[["tanim", "tarih"]], te[["tanim", "tarih"]]], ignore_index=True)
    ort = ort.sort_values(["tanim", "tarih"], kind="mergesort")
    onc = ort.groupby("tanim", observed=True)["tarih"].shift(1)
    son = ort.groupby("tanim", observed=True)["tarih"].shift(-1)
    ort["giris"] = onc.isna() | ((ort["tarih"] - onc) > GUN)
    ort["cikis"] = son.isna() | ((son - ort["tarih"]) > GUN)
    t = ort[ort["tarih"] >= TEST_BAS].copy()
    t["cikis"] &= t["tarih"] != TEST_SON  # pencere kenari: bilinemez
    t["soguk"] = ~t["tanim"].isin(sicak_set)

    print("=" * 92)
    print("TEST'TE UYGULANABILIR SINIR NUFUSU (train'e koprulu, pencere kenari haric)")
    print("=" * 92)
    say = {}
    for rej, m in (("SICAK", ~t["soguk"]), ("SOGUK", t["soguk"])):
        g = int((t["giris"] & m).sum())
        c = int((t["cikis"] & m).sum())
        say[rej] = {"giris": g, "cikis": c, "n": int(m.sum())}
        print(
            f"  {rej}: toplam {int(m.sum()):>7,}   giris {g:>6,} ({g / N_TEST:.6f})"
            f"   cikis {c:>6,} ({c / N_TEST:.6f})"
        )
    print(
        f"\n  TOPLAM giris {int(t['giris'].sum()):,}  cikis {int(t['cikis'].sum()):,}"
        f"  -> birlikte {int((t['giris'] | t['cikis']).sum()):,} satir "
        f"({int((t['giris'] | t['cikis']).sum()) / N_TEST:.5f})"
    )
    print("\n  SOGUK giris gunleri (en kalabalik 6):")
    print(t.loc[t["giris"] & t["soguk"], "tarih"].value_counts().head(6).to_string())
    print("\n  SICAK giris gunleri (en kalabalik 6):")
    print(t.loc[t["giris"] & ~t["soguk"], "tarih"].value_counts().head(6).to_string())

    # --- blok olcumlerinden SATIR BASI kazanc (panel_sinir_soguk.py ciktisi)
    print("\n" + "=" * 92)
    print("SATIR BASI KAZANC -> TEST OLCEGI")
    print("=" * 92)
    # (blok, rejim) -> (blok_n, giris_n, cikis_n, ortak_kazanc_blokMSE)
    olcum = {
        # SOGUK: panel_sinir_soguk.py, blok basina optimum
        ("soguk", "yaz25"): (20_633, 852, 257, -0.109177),
        ("soguk", "guz25"): (39_405, 1_237, 166, -0.011073),
        ("soguk", "kis26"): (61_918, 1_392, 584, -0.011689),
    }
    tg, tc = say["SOGUK"]["giris"], say["SOGUK"]["cikis"]
    print(f"\n  SOGUK (test giris {tg:,}, test cikis {tc:,})")
    tah = []
    for (rej, blok), (bn, gn, cn, kz) in olcum.items():
        toplam_e2_dusus = -kz * bn  # blok MSE dususu * satir sayisi
        sinir_n = gn + cn
        per = toplam_e2_dusus / sinir_n  # sinir satiri basina e^2 dususu
        test_kz = -per * (tg + tc) / N_TEST
        tah.append(test_kz)
        print(
            f"    {blok:6} blok kazanc {kz:+.6f} x {bn:,} = {toplam_e2_dusus:8.1f}"
            f"   sinir {sinir_n:5,}   satir basi {per:7.3f}"
            f"   -> TEST dMSE {test_kz:+.6f}"
        )
    print(
        f"    ORTALAMA (uc blok)  -> TEST dMSE {np.mean(tah):+.6f}"
        f"   MEDYAN {np.median(tah):+.6f}   EN KOTU {max(tah):+.6f}"
    )

    (CIK / "panel_sinir_test_olcek.json").write_text(
        json.dumps(
            {
                "test_sayim": say,
                "soguk_tahmin": tah,
                "soguk_ortalama": float(np.mean(tah)),
                "soguk_medyan": float(np.median(tah)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nNOT: SICAK tarafin satir basi kazanci panel_sinir2.py ciktisindan")
    print("     ayni yontemle olceklenir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
