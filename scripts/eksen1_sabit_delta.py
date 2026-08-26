"""EKSEN 1 -- KAPANIS: teste hangi SABIT delta yazilmali?

b_i kestiricisi kesmeler arasi TASINMADIGINA gore geriye sabit delta kaliyor.
Bu betik, sabit delta d'nin BES temiz kesme ve URETIM kis26 uzerindeki
gerceklesen satir basi kazancini tarar ve teste cevirisini verir.

d uygulaninca satir basi kazanc = 2*d*delta - d^2  (delta = o kesmenin gercek
agirlikli yanliligi). Yani d'nin isareti delta'nin isaretini tutturmazsa ZARAR.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "scripts"))
CIKTI = KOK / "data" / "interim" / "eksen1_kesme"

from eksen1_coklu_kesme import b_tablosu, kazanc, panel_kur, veri_yukle  # noqa: E402
from eksen1_uretim_transfer import uretim_kis26  # noqa: E402

KESMELER = ["2025-03-31", "2025-05-31", "2025-07-31", "2025-09-30", "2025-11-30"]
SICAK_PAY = 0.7784
TABAN_MSE = 1.03207


def main():
    d_ = veri_yukle()
    P = panel_kur(d_)
    n_t = P["n_t"]

    K = {}
    for ks in KESMELER:
        z = np.load(CIKTI / f"kesme_{ks}.npz", allow_pickle=True)
        r = dict(ti=z["ti"], tahmin=z["tahmin"], gercek=z["gercek"])
        b, say = b_tablosu(r, n_t)
        w = say > 0
        r["delta"] = float(np.sum(say[w] * b[w]) / say[w].sum())
        K[ks] = r
    U = uretim_kis26(P)
    bU, sayU = b_tablosu(U, n_t)
    wU = sayU > 0
    U["delta"] = float(np.sum(sayU[wU] * bU[wU]) / sayU[wU].sum())

    dl = np.array([K[k]["delta"] for k in KESMELER])
    print("=== temiz kesme deltalari ===")
    for k in KESMELER:
        print(f"  {k}  delta={K[k]['delta']:+.4f}")
    print(
        f"  ortalama {dl.mean():+.4f}   std {dl.std(ddof=1):.4f}   "
        f"|ort|/std = {abs(dl.mean()) / dl.std(ddof=1):.3f}"
    )
    print(f"  URETIM kis26 (gercek model) delta={U['delta']:+.4f}")

    ADAYLAR = [0.0, 0.02, 0.05, 0.0703, 0.10, 0.15, 0.1899, 0.25, 0.3266]
    print("\n=== SABIT DELTA TARAMASI -- satir basi kazanc (+ = iyilesme) ===")
    print(
        f"{'d':>8}"
        + "".join(f"{k[5:]:>10}" for k in KESMELER)
        + f"{'ORT':>10}{'EN KOTU':>10}{'URETIM':>10}{'dMSE':>10}{'RMSLE':>9}"
    )
    for d in ADAYLAR:
        g = [
            kazanc(K[k]["gercek"], K[k]["tahmin"], np.full(len(K[k]["ti"]), d), 1.0)
            for k in KESMELER
        ]
        gu = kazanc(U["gercek"], U["tahmin"], np.full(len(U["ti"]), d), 1.0)
        ort = float(np.mean(g))
        dmse = -SICAK_PAY * ort
        print(
            f"{d:>8.4f}"
            + "".join(f"{v:>10.5f}" for v in g)
            + f"{ort:>10.5f}{min(g):>10.5f}{gu:>10.5f}{dmse:>10.5f}"
            f"{np.sqrt(TABAN_MSE + dmse):>9.5f}"
        )

    print("\n=== LEAVE-ONE-OUT DURUSTLUK: d'yi diger 4 kesmeden kur, 5.'de sina ===")
    top = []
    for i, k in enumerate(KESMELER):
        d = float(np.delete(dl, i).mean())
        g = kazanc(K[k]["gercek"], K[k]["tahmin"], np.full(len(K[k]["ti"]), d), 1.0)
        top.append(g)
        print(f"  {k}: d={d:+.4f} -> kazanc {g:+.5f}")
    print(f"  LOO ortalama kazanc {np.mean(top):+.5f}   pozitif {sum(v > 0 for v in top)}/5")
    dmse = -SICAK_PAY * float(np.mean(top))
    print(
        f"  -> dMSE {dmse:+.5f}   RMSLE {np.sqrt(TABAN_MSE + dmse):.5f} "
        f"(hedef 1,00635 icin gereken dMSE -0,01933)"
    )

    print("\n=== d = URETIM kis26 deltasi (+0,1899) hangi kesmede ne yapar ===")
    for k in KESMELER:
        g = kazanc(K[k]["gercek"], K[k]["tahmin"], np.full(len(K[k]["ti"]), U["delta"]), 1.0)
        print(f"  {k}: {g:+.5f}")


if __name__ == "__main__":
    main()
