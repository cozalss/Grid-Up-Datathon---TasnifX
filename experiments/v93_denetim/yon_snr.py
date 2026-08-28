"""v93 denetimi -- adim 9: her ozyonun SINYAL/GURULTU orani ve kazanca katkisi."""

from __future__ import annotations

import json
from pathlib import Path

import coz as C
import numpy as np

BURA = Path(__file__).resolve().parent
DS = 5e-6  # skor yuvarlama yari-genligi


def main() -> None:
    p0 = C.yukle(C.TABAN)
    n = p0.size
    dig = [e for e in C.OLCULENLER if e != C.TABAN]
    D = np.empty((len(dig), n))
    for i, e in enumerate(dig):
        D[i] = C.yukle(e) - p0
    G = (D @ D.T) / n
    m0 = C.ENV[C.TABAN]["skor"] ** 2
    m = np.array([C.ENV[e]["skor"] ** 2 for e in dig])
    b = (m0 + np.diag(G) - m) / 2.0
    U, s, Vt = np.linalg.svd(G, hermitian=True)
    sk = np.array([C.ENV[e]["skor"] for e in dig])
    s0 = C.ENV[C.TABAN]["skor"]

    print("Her ozyon j icin: kazanc_j = (b.v_j)^2 / s_j  (proj^2'ye katki)")
    print("Gurultu: eps_i = (dm0 - dm_i)/2, dm=2*s*ds, ds~U(-5e-6,5e-6)")
    print(
        f"\n{'j':>3} {'s_j':>11} {'b.v_j':>12} {'sd_gurultu':>11} {'SNR':>8} "
        f"{'kazanc_j':>11} {'kum.RMSLE':>10}"
    )
    kum = 0.0
    kayit = []
    for j in range(18):
        v = Vt[j]
        bv = float(b @ v)
        var = (s0**2 * DS**2 / 3) * v.sum() ** 2 + np.sum((sk**2 * DS**2 / 3) * v**2)
        sdn = float(np.sqrt(var))
        kaz = bv**2 / s[j] if s[j] > 1e-10 else float("nan")
        if j < 16:
            kum += kaz
        r = np.sqrt(max(m0 - kum, 0))
        kayit.append(
            {
                "j": j,
                "s": float(s[j]),
                "bv": bv,
                "sd": sdn,
                "snr": abs(bv) / sdn,
                "kazanc": float(kaz) if kaz == kaz else None,
                "kum_rmsle": float(r),
            }
        )
        print(
            f"{j:>3} {s[j]:>11.3e} {bv:>+12.4e} {sdn:>11.3e} {abs(bv) / sdn:>8.1f} "
            f"{(f'{kaz:.6f}' if kaz == kaz else 'n/a'):>11} {r:>10.6f}"
        )

    print("\nEn kucuk uc tutulan yonun (13,14,15) toplam katkisi:")
    ek = sum(k["kazanc"] for k in kayit[13:16])
    print(f"  {ek:.6f} MSE  ->  RMSLE'de {np.sqrt(m0 - kum + ek) - np.sqrt(m0 - kum):.6f} kazanc")
    print(f"  Bunlarsiz (rank 13) RMSLE = {np.sqrt(m0 - kum + ek):.6f}")

    (BURA / "yon_snr.json").write_text(json.dumps(kayit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
