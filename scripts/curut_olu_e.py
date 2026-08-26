"""CURUTUCU E -- dMSE muhasebesi ve 'model > sabit' iddiasinin ULASILABILIR hali.

1) Bulgunun gerekcesi "model olu satirlarda tek sabitten iyi". Bunu ULASILABILIR
   sabitle (blok-disi LOO) ve KURAL 1 kirpmasiyla sina.
2) Butun aday kurallarin dMSE'sini (test paydasina tasinmis) hesapla.
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
P_TEST = 0.04277  # D4: kuyruk>=1 trafolarin test satir payi


def main() -> int:
    t0 = time.time()
    veri = veri_yukle()

    print("=" * 100)
    print("E1) MODEL vs ULASILABILIR (blok-disi LOO) TEK SABIT -- olu satirlarda")
    print("    KURAL 1 kirpmasi: modelin ustunlugune en cok katan K trafo atiliyor")
    print("=" * 100)
    print(f"  {'blok':7}{'LOO c':>8}{'trafo':>8}" + "".join(f"{'K=' + str(k):>11}" for k in KLER))
    for b in BLOKLAR:
        v = veri[b]
        m = v["olu"]
        dig = [x for x in BLOKLAR if x != b]
        c = float(np.concatenate([veri[x]["ly"][veri[x]["olu"]] for x in dig]).mean())
        ly, lt, tn = v["ly"][m], v["log_t"][m], v["tanim"][m]
        e_mod = (lt - ly) ** 2
        e_sab = (c - ly) ** 2
        d = e_mod - e_sab
        katki = pd.DataFrame({"tanim": tn, "d": d}).groupby("tanim")["d"].sum().sort_values()
        satir = f"  {b:7}{c:8.3f}{katki.size:8,}"
        for K in KLER:
            at = set(katki.index[:K])
            tut = ~pd.Series(tn).isin(at).to_numpy()
            satir += f"{rmse(e_sab[tut]) - rmse(e_mod[tut]):+11.5f}"
        print(satir)
    print("  (POZITIF = model o alt kumede ULASILABILIR sabitten IYI)")

    print()
    print("=" * 100)
    print("E2) dMSE MUHASEBESI -- olu satirlarda MSE degisimi x test satir payi")
    print("=" * 100)
    print(
        f"  taban MSE {TABAN_MSE}  (RMSLE {np.sqrt(TABAN_MSE):.5f})   test p (kuyruk>=1) {P_TEST}"
    )
    print(
        f"\n  {'kural':38}{'yaz25 dm':>11}{'guz25 dm':>11}{'kis26 dm':>11}{'ort dm':>10}"
        f"{'dMSE(test)':>12}{'yeni RMSLE':>12}"
    )

    def satir(ad: str, uret) -> None:  # noqa: ANN001
        dm = []
        for b in BLOKLAR:
            v = veri[b]
            m = v["olu"]
            yeni = uret(b, v, m)
            e0 = (v["log_t"][m] - v["ly"][m]) ** 2
            e1 = (yeni - v["ly"][m]) ** 2
            dm.append(float(e1.mean() - e0.mean()))
        ort = float(np.mean(dm))
        dmse = P_TEST * ort
        print(
            f"  {ad:38}{dm[0]:+11.4f}{dm[1]:+11.4f}{dm[2]:+11.4f}{ort:+10.4f}"
            f"{dmse:+12.5f}{np.sqrt(TABAN_MSE + dmse):12.5f}"
        )

    def loo_c(b: str) -> float:
        dig = [x for x in BLOKLAR if x != b]
        return float(np.concatenate([veri[x]["ly"][veri[x]["olu"]] for x in dig]).mean())

    for w in (0.1, 0.2, 0.5, 1.0):
        satir(
            f"LOO tek sabit, w={w:.1f}", lambda b, v, m, w=w: (1 - w) * v["log_t"][m] + w * loo_c(b)
        )
    satir("ORAKUL tek sabit (ULASILAMAZ)", lambda b, v, m: np.full(int(m.sum()), v["ly"][m].mean()))
    satir(
        "ORAKUL trafo sabiti (ULASILAMAZ)",
        lambda b, v, m: (
            pd.Series(v["ly"][m]).groupby(pd.Series(v["tanim"][m])).transform("mean").to_numpy()
        ),
    )
    satir("YAPMA (recete)", lambda b, v, m: v["log_t"][m])

    print()
    print("=" * 100)
    print("E3) 'tek sabit w=0,10' kuralinin K=1 kirpilmis hali -- dMSE")
    print("=" * 100)
    for K in (0, 1, 5):
        dm = []
        for b in BLOKLAR:
            v = veri[b]
            m = v["olu"]
            c = loo_c(b)
            lt, ly, tn = v["log_t"][m], v["ly"][m], v["tanim"][m]
            e0 = (lt - ly) ** 2
            e1 = (0.9 * lt + 0.1 * c - ly) ** 2
            katki = pd.DataFrame({"tanim": tn, "d": e1 - e0}).groupby("tanim")["d"].sum()
            at = set(katki.sort_values().index[:K])
            tut = ~pd.Series(tn).isin(at).to_numpy()
            dm.append(float(e1[tut].mean() - e0[tut].mean()))
        ort = float(np.mean(dm))
        print(
            f"  K={K:2d}  blok dm {dm[0]:+.4f} {dm[1]:+.4f} {dm[2]:+.4f}  ort {ort:+.4f}"
            f"  dMSE {P_TEST * ort:+.5f}  yeni RMSLE {np.sqrt(TABAN_MSE + P_TEST * ort):.5f}"
        )

    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
