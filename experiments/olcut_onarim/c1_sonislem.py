"""C6 son bullet: seviye/genlik son islem sabitleri ONARILMIS olcutle."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BURA = Path(__file__).resolve().parent
sys.path.insert(0, str(BURA))
import tezgah as tz, kapi as kp  # noqa

B0, C0, D0 = 0.60, 1.3301, 0.1046
setup = sys.argv[1] if len(sys.argv) > 1 else "onarilmis"

veri = {}
for b in tz.BLOKLAR:
    m = tz.meta(b)
    veri[b] = (
        m,
        m["y"].to_numpy(float),
        np.log1p(m["guc"].to_numpy(float)),
        np.load(tz.ONBELLEK / f"{setup}_{b}_cat_d7.npy").mean(axis=0),
    )

_prof = {}


def e(b, beta, c, delta):
    m, y, lgc, lg = veri[b]
    r = lg - lgc
    ort = float(r.mean())
    lg2 = ort + beta * (r - ort) + lgc
    if (b, beta) not in _prof:
        _prof[(b, beta)] = kp.gun_profili(lg2, m)
    return kp.kare_hatalar(y, lg2 + (c - 1.0) * _prof[(b, beta)] + delta)


taban = {b: e(b, B0, C0, D0) for b in tz.BLOKLAR}
print("=" * 116)
print(f"SON ISLEM SABITLERI -- setup={setup}, cat_d7, taban beta={B0} c={C0} delta={D0}")
print("=" * 116)
print(
    f"{'aday':22}"
    + "".join(f"{b + ' dMSE':>13}" for b in tz.BLOKLAR)
    + f"{'isaret':>8}{'kis26 CI':>22}{'kaz.tr':>8}{'KAPI':>6}"
)
adaylar = []
for beta in (0.30, 0.40, 0.50, 0.70, 0.80, 1.00):
    adaylar.append((f"beta={beta:.2f}", beta, C0, D0))
for c in (1.00, 1.60, 2.00, 2.20, 2.50):
    adaylar.append((f"c={c:.2f}", B0, c, D0))
for dd in (-0.10, -0.05, 0.00, 0.05, 0.15, 0.20, 0.25, 0.30):
    adaylar.append((f"delta={dd:+.2f}", B0, C0, dd))
for ad, be, c, dd in adaylar:
    es = {b: e(b, be, c, dd) for b in tz.BLOKLAR}
    dl = [es[b].mean() - taban[b].mean() for b in tz.BLOKLAR]
    ayni = "AYNI" if (all(v < 0 for v in dl) or all(v > 0 for v in dl)) else "-"
    r = kp.bootstrap(taban["kis26"], es["kis26"], veri["kis26"][0])
    ci = f"[{r['ci_lo']:+.5f},{r['ci_hi']:+.5f}]"
    print(
        f"{ad:22}"
        + "".join(f"{v:>+13.5f}" for v in dl)
        + f"{ayni:>8}{ci:>22}{100 * r['kazanan_trafo']:>7.1f}%"
        + f"{('GECTI' if r['gecti'] else '-'):>6}"
    )

# blok bazinda serbest optimum
print("\nBLOK BAZINDA SERBEST OPTIMUM (ic-orneklem, yalniz teshis)")
for b in tz.BLOKLAR:
    en = (9e9, None)
    for be in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for c in (1.0, 1.3301, 1.6, 2.0, 2.2, 2.5):
            for dd in np.round(np.arange(-0.2, 0.401, 0.02), 3):
                v = e(b, be, c, dd).mean()
                if v < en[0]:
                    en = (v, (be, c, dd))
    print(
        f"  {b:6} MSE {en[0]:.5f}  taban {taban[b].mean():.5f}  "
        f"@ beta={en[1][0]} c={en[1][1]} delta={en[1][2]:+.2f}"
    )
