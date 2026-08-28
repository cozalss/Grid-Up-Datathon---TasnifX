"""TESHIS: duzenek dogru mu, ve sicak artik yapisi bloklar arasi TASINIYOR mu?

1) HILE testi -- ayni blogun etiketinden ogrenilen ofset kazandiriyor mu?
   Kazandirmiyorsa duzenekte hata var.
2) IKILI TASIMA -- kaynak blok basina ayri ayri; hangi cift tasiyor?
3) KORELASYON -- grup ofsetlerinin bloklar arasi korelasyonu.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ortak import BLOKLAR, bloklari_kur, mse, taban_r  # noqa: E402


def artik(b, r0):
    return b.lgy - (r0 + b.lgc)


def grup_haritasi(b, r0, anah, n0=200.0):
    e = artik(b, r0)
    e = e - e.mean()
    g = pd.Series(b.cerceve[anah].to_numpy())
    agg = pd.DataFrame({"e": e}).groupby(g)["e"].agg(["mean", "size"])
    return (agg["mean"] * agg["size"] / (agg["size"] + n0)).to_dict()


def uygula(b, r0, harita, anah, kat=1.0):
    d = pd.Series(b.cerceve[anah].to_numpy()).map(harita).fillna(0.0).to_numpy()
    return r0 + kat * (d - d.mean())


def main() -> int:
    bl = bloklari_kur()
    taban = {k: taban_r(b) for k, b in bl.items()}
    for k in BLOKLAR:
        c = bl[k].cerceve
        c["seviye_d"] = pd.qcut(c["t_log_ort"].to_numpy(), 20, labels=False, duplicates="drop")
        c["gecmis_k"] = np.digitize(c["gecmis_gun"].to_numpy(), [7, 31, 91, 181, 366])

    anahlar = ["hg", "kova", "ilce", "seviye_d", "gecmis_k", "tanim"]

    print("=" * 96)
    print("1) HILE TESTI -- ofset AYNI bloktan ogrenilirse dMSE (negatif olmali)")
    print("=" * 96)
    print(f"{'anahtar':12}" + "".join(f"{k:>12}" for k in BLOKLAR))
    for a in anahlar:
        sat = f"{a:12}"
        for k in BLOKLAR:
            b = bl[k]
            h = grup_haritasi(b, taban[k], a)
            sat += f"{mse(b, uygula(b, taban[k], h, a)) - mse(b, taban[k]):>+12.5f}"
        print(sat)

    print()
    print("=" * 96)
    print("2) IKILI TASIMA -- kaynak blokta ogren, hedef bloga uygula (dMSE)")
    print("=" * 96)
    for a in anahlar:
        print(f"\n-- {a} --")
        print("kaynak->hedef " + "".join(f"{k:>12}" for k in BLOKLAR))
        for j in BLOKLAR:
            sat = f"{j:14}"
            h = grup_haritasi(bl[j], taban[j], a)
            for k in BLOKLAR:
                if k == j:
                    sat += f"{'-':>12}"
                    continue
                b = bl[k]
                sat += f"{mse(b, uygula(b, taban[k], h, a)) - mse(b, taban[k]):>+12.5f}"
            print(sat)

    print()
    print("=" * 96)
    print("3) GRUP OFSETLERININ BLOKLAR ARASI KORELASYONU")
    print("=" * 96)
    for a in anahlar:
        h = {k: pd.Series(grup_haritasi(bl[k], taban[k], a)) for k in BLOKLAR}
        satir = f"{a:12}"
        for i in range(3):
            for j in range(i + 1, 3):
                x = pd.concat([h[BLOKLAR[i]], h[BLOKLAR[j]]], axis=1, join="inner").dropna()
                if len(x) < 3:
                    satir += f"  {BLOKLAR[i][:3]}/{BLOKLAR[j][:3]} n<3"
                    continue
                satir += (
                    f"  {BLOKLAR[i][:3]}/{BLOKLAR[j][:3]} "
                    f"kor{x.iloc[:, 0].corr(x.iloc[:, 1]):+.3f}(n={len(x)})"
                )
        print(satir)

    print()
    print("=" * 96)
    print("4) URETIM KNOB'LARI -- taban zincirindeki sabitlerin blok basi optimumu")
    print("=" * 96)
    ham = {k: np.mean(bl[k].tohum_harman, axis=0) - bl[k].lgc for k in BLOKLAR}
    print("\n-- gun ekseni olcegi c (uretim 1,3301) --")
    print(f"{'c':>7}" + "".join(f"{k:>12}" for k in BLOKLAR))
    from ortak import gun_etkisi

    etkiler = {}
    for k in BLOKLAR:
        b = bl[k]
        be = gun_etkisi(b.cerceve["tanim"].to_numpy(), b.cerceve["tarih"].to_numpy(), ham[k])
        e = pd.Series(b.cerceve["tarih"].to_numpy()).map(be).to_numpy(dtype="float64")
        etkiler[k] = e - e.mean()
    for c in (1.0, 1.15, 1.3301, 1.5, 1.75, 2.0):
        sat = f"{c:>7.4f}"
        for k in BLOKLAR:
            b = bl[k]
            r = ham[k] + (c - 1.0) * etkiler[k]
            r = r + float((b.lgy - b.lgc - r).mean())
            r0 = ham[k] + (1.3301 - 1.0) * etkiler[k]
            r0 = r0 + float((b.lgy - b.lgc - r0).mean())
            sat += f"{mse(b, r) - mse(b, r0):>+12.5f}"
        print(sat)

    print("\n-- kuyruk deltasi (uretim +0,16640) --")
    print(f"{'delta':>7}" + "".join(f"{k:>12}" for k in BLOKLAR) + f"{'n_kuyruk':>10}")
    for d in (0.0, 0.08, 0.1664, 0.25, 0.35):
        sat = f"{d:>7.4f}"
        nk = 0
        for k in BLOKLAR:
            b = bl[k]
            ku = b.cerceve["kuyruk"].to_numpy(dtype="float64")
            r = ham[k] + (1.3301 - 1.0) * etkiler[k] + d * ku
            r = r + float((b.lgy - b.lgc - r).mean())
            r0 = ham[k] + (1.3301 - 1.0) * etkiler[k] + 0.1664 * ku
            r0 = r0 + float((b.lgy - b.lgc - r0).mean())
            sat += f"{mse(b, r) - mse(b, r0):>+12.5f}"
            nk += int(ku.sum())
        print(sat + f"{nk:>10,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
