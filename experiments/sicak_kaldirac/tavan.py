"""FIZIKSEL TAVAN -- gunluk tuketim kVA x 24 saati asamaz.

Bir trafonun bir gunde cekebilecegi enerji, anma gucu (kVA) x 24 saat ile
sinirlidir (guc katsayisi 1, %100 yuklenme). Model log uzayinda calisip
kapasite ofsetini ogrendigi icin bu siniri BILMIYOR ve bazi satirlarda
fiziksel olarak imkansiz degerler uretiyor.

Bu aday TASIMA VARSAYIMI GEREKTIRMEZ: sinir fizikten, kalibrasyonu ise
ETIKETLI train'in tamamindan gelir. Yine de kapi ayni -- uc blokta ayni
isaret ve test dMSE <= -0,002.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ortak import BLOKLAR, SICAK_PAY, bloklari_kur, mse, taban_r  # noqa: E402

KOK = Path(__file__).resolve().parents[2]


def main() -> int:
    print("=" * 96)
    print("1) TRAIN'DE GERCEK y / guc ORANI -- fiziksel zarf")
    print("=" * 96)
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "guc", "tuketim"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    o = tr["tuketim"].to_numpy(dtype="float64") / np.maximum(
        tr["guc"].to_numpy(dtype="float64"), 1e-9
    )
    for q in (0.5, 0.9, 0.99, 0.999, 0.9999, 1.0):
        print(f"  y/guc  q{q:<8} {np.quantile(o, q):10.3f}")
    for e in (12, 18, 24, 30, 40, 60):
        print(f"  y/guc > {e:>3}: {int((o > e).sum()):>8,} satir  (%{100 * (o > e).mean():.4f})")

    bl = bloklari_kur()
    taban = {k: taban_r(b) for k, b in bl.items()}
    tm = {k: mse(bl[k], taban[k]) for k in BLOKLAR}

    print()
    print("=" * 96)
    print("2) TAVAN IZGARASI -- dMSE ve etkilenen satir sayisi")
    print("=" * 96)
    print(
        f"{'tavan':>8}"
        + "".join(f"{k + ' dMSE':>14}{'n':>8}" for k in BLOKLAR)
        + f"{'GENEL':>11}{'testdMSE':>11}"
    )
    en_iyi = None
    for kat in (60, 48, 40, 36, 30, 26, 24, 22, 20, 18, 15, 12):
        sat = f"{kat:>8}"
        tn = td = 0.0
        isaret = []
        for k in BLOKLAR:
            b = bl[k]
            tah = np.expm1(taban[k] + b.lgc)
            sinir = kat * b.cerceve["guc"].to_numpy(dtype="float64")
            kes = tah > sinir
            e = b.lgy - np.log1p(np.clip(np.minimum(tah, sinir), 0.0, None))
            d = float((e * e).mean()) - tm[k]
            isaret.append(d)
            sat += f"{d:>+14.5f}{int(kes.sum()):>8,}"
            tn += b.n
            td += d * b.n
        g = td / tn
        sat += f"{g:>+11.5f}{g * SICAK_PAY:>+11.5f}"
        uygun = all(x <= 0 for x in isaret)
        print(sat + ("  <- uc blokta da zararsiz" if uygun else ""))
        if uygun and (en_iyi is None or g < en_iyi[1]):
            en_iyi = (kat, g)

    print()
    print("=" * 96)
    print("3) TAVANI ASAN SATIRLARIN ANATOMISI (tavan 24x)")
    print("=" * 96)
    for k in BLOKLAR:
        b = bl[k]
        tah = np.expm1(taban[k] + b.lgc)
        guc = b.cerceve["guc"].to_numpy(dtype="float64")
        kes = tah > 24 * guc
        if not kes.any():
            print(f"  {k}: asan satir YOK")
            continue
        y = b.y[kes]
        print(
            f"  {k}: n={int(kes.sum()):,}  trafo={b.cerceve.loc[kes, 'tanim'].nunique()}  "
            f"tahmin/guc medyan {np.median(tah[kes] / guc[kes]):.1f} max {(tah[kes] / guc[kes]).max():.1f}  "
            f"| gercek/guc medyan {np.median(y / guc[kes]):.2f} max {(y / guc[kes]).max():.1f}"
        )
        e0 = b.lgy[kes] - np.log1p(tah[kes])
        e1 = b.lgy[kes] - np.log1p(24 * guc[kes])
        print(
            f"      bu satirlarda MSE {float((e0 * e0).mean()):.3f} -> {float((e1 * e1).mean()):.3f}"
            f"  | blok MSE'sine katki {float((e0 * e0).sum() - (e1 * e1).sum()) / b.n:+.5f}"
        )

    print()
    print("=" * 96)
    print("4) v83 TEST DOSYASINDA TAVANI ASAN SATIRLAR")
    print("=" * 96)
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "guc"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    sub = pd.read_csv(KOK / "submissions/tuketim_v83_sicak_optimum.csv", encoding="utf-8")
    m = sub.merge(te, on="id", how="left", validate="one_to_one")
    soguk = ~m["tanim"].isin(set(tr["tanim"])).to_numpy()
    tah = m["tuketim"].to_numpy(dtype="float64")
    guc = m["guc"].to_numpy(dtype="float64")
    for kat in (60, 40, 30, 24, 20, 18):
        s = tah > kat * guc
        print(
            f"  tavan {kat:>3}x : toplam {int(s.sum()):>6,}  "
            f"sicak {int((s & ~soguk).sum()):>6,}  soguk {int((s & soguk).sum()):>6,}"
        )
    s = tah > 24 * guc
    if s.any():
        print(
            f"  24x asanlarin tahmin/guc: medyan {np.median(tah[s] / guc[s]):.1f} max {(tah[s] / guc[s]).max():.1f}"
        )
        print(f"  etkilenen trafo: {m.loc[s, 'tanim'].nunique()}")
    if en_iyi:
        print(
            f"\n  UC BLOKTA DA ZARARSIZ EN IYI TAVAN: {en_iyi[0]}x  (sicak dMSE {en_iyi[1]:+.5f}, test {en_iyi[1] * SICAK_PAY:+.5f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
