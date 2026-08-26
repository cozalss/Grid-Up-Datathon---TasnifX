"""CURUTUCU F -- recetenin tek istisnasi: 'TAM SIFIR tahmin' satirlari.

Recete diyor ki olu_hedge KOVALAR tablosu YALNIZ tahmin tam sifir olan
satirlara uygulandigi surece zararsiz. Bunu olcelim: bloklarda log_t <= 0
olan satirlarda mevcut hata ne, olu_hedge tabani ne yapardi, LOO sabiti ne
yapardi -- ve KURAL 1 kirpmasi.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

from curut_olu_a import BLOKLAR, KLER, rmse, veri_yukle  # noqa: E402

TABAN_MSE = 1.03207
KOVALAR = ((1, 15, 1.030), (15, 30, 0.855), (30, 60, 1.071), (60, 90, 0.161), (90, 10**6, 0.230))


def hedge_taban(k: float) -> float:
    for alt, ust, h in KOVALAR:
        if alt <= k < ust:
            return h
    return 0.0


def main() -> int:
    t0 = time.time()
    veri = veri_yukle()

    print("=" * 100)
    print("F1) log_t <= 0 (tahmin TAM SIFIR) satirlari -- blok tablosu")
    print("=" * 100)
    print(
        f"  {'blok':7}{'satir':>9}{'pay %':>8}{'trafo':>7}{'dirilme %':>11}"
        f"{'gercek ort log1p':>18}{'mevcut MSE':>12}"
    )
    for b in BLOKLAR:
        v = veri[b]
        m = v["log_t"] <= 0.0
        if m.sum() == 0:
            print(f"  {b:7}{'0':>9}")
            continue
        ly = v["ly"][m]
        print(
            f"  {b:7}{int(m.sum()):9,}{100 * m.mean():8.3f}{len(np.unique(v['tanim'][m])):7,}"
            f"{100 * (ly > 0).mean():11.2f}{ly.mean():18.4f}"
            f"{((v['log_t'][m] - ly) ** 2).mean():12.5f}"
        )

    print()
    print("=" * 100)
    print("F2) TAM SIFIR satirlarina taban yazmak -- tam blok dRMSLE (sicak payda)")
    print("=" * 100)
    adaylar = {
        "olu_hedge KOVALAR": lambda v, m: np.array([hedge_taban(k) for k in v["kuyruk"][m]]),
        "sabit 0,230": lambda v, m: np.full(int(m.sum()), 0.230),
        "sabit 0,500": lambda v, m: np.full(int(m.sum()), 0.500),
        "sabit 1,000": lambda v, m: np.full(int(m.sum()), 1.000),
    }
    print(f"  {'aday':22}" + "".join(f"{b:>12}" for b in BLOKLAR) + f"{'3/3':>6}{'ort':>11}")
    for ad, fn in adaylar.items():
        farklar = []
        for b in BLOKLAR:
            v = veri[b]
            m = v["log_t"] <= 0.0
            e0 = (v["log_t"] - v["ly"]) ** 2
            yeni = v["log_t"].copy()
            if m.sum():
                yeni[m] = fn(v, m)
            e1 = (yeni - v["ly"]) ** 2
            farklar.append(rmse(e1) - rmse(e0))
        kaz = sum(1 for f in farklar if f < 0)
        print(
            f"  {ad:22}"
            + "".join(f"{f:+12.5f}" for f in farklar)
            + f"{kaz:>4}/3{np.mean(farklar):+11.5f}"
        )

    print()
    print("=" * 100)
    print("F3) KIRPMA TABLOSU -- olu_hedge KOVALAR")
    print("=" * 100)
    print(f"  {'blok':7}{'trafo':>8}" + "".join(f"{'K=' + str(k):>10}" for k in KLER))
    for b in BLOKLAR:
        v = veri[b]
        m = v["log_t"] <= 0.0
        if m.sum() == 0:
            continue
        e0 = (v["log_t"] - v["ly"]) ** 2
        yeni = v["log_t"].copy()
        yeni[m] = np.array([hedge_taban(k) for k in v["kuyruk"][m]])
        e1 = (yeni - v["ly"]) ** 2
        tn = v["tanim"]
        katki = pd.DataFrame({"tanim": tn[m], "d": (e1 - e0)[m]}).groupby("tanim")["d"].sum()
        katki = katki.sort_values()
        satir = f"  {b:7}{katki.size:8,}"
        for K in KLER:
            at = set(katki.index[:K])
            tut = ~pd.Series(tn).isin(at).to_numpy()
            satir += f"{rmse(e1[tut]) - rmse(e0[tut]):+10.5f}"
        print(satir)

    print()
    print("=" * 100)
    print("F4) dMSE -- v55'teki 7.572 tam sifir satirina uygulansaydi")
    print("=" * 100)
    p = 7572 / 714688
    for ad, fn in adaylar.items():
        dm = []
        for b in BLOKLAR:
            v = veri[b]
            m = v["log_t"] <= 0.0
            if m.sum() == 0:
                continue
            e0 = ((v["log_t"][m] - v["ly"][m]) ** 2).mean()
            e1 = ((fn(v, m) - v["ly"][m]) ** 2).mean()
            dm.append(float(e1 - e0))
        ort = float(np.mean(dm))
        print(
            f"  {ad:22} blok dm ort {ort:+8.4f}  p {p:.5f}  dMSE {p * ort:+.5f}"
            f"  yeni RMSLE {np.sqrt(TABAN_MSE + p * ort):.5f}"
        )

    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
