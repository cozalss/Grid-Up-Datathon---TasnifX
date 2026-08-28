"""J6 - uc blokta: ust kirpma (tavan), alt taban, kuresel seviye, ufuk egimi."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(
    0, r"C:\Users\Cem\Desktop\Datahon_Laptop\Grid-Up-Datathon---TasnifX\experiments\joker"
)
from j05_tezgah import bloklari_kur  # noqa: E402

BLOKLAR = ("yaz25", "guz25", "kis26")


def p(*a):
    print(*a, flush=True)


def skorla(lp, ly):
    e = np.log1p(np.clip(np.expm1(lp), 0.0, None)) - ly
    return float((e * e).mean())


bl = bloklari_kur()
taban = {b: skorla(bl[b]["lp"].to_numpy(), bl[b]["ly"].to_numpy()) for b in BLOKLAR}
p("taban MSE:", {k: round(v, 6) for k, v in taban.items()})

p()
p("=" * 92)
p("1) UST KIRPMA (tavan C, kWh)")
p("=" * 92)
p(f"{'C':>12}" + "".join(f"{b:>12}" for b in BLOKLAR) + f"{'etkilenen':>12}")
for C in [1e9, 5e5, 3e5, 2e5, 1.5e5, 1e5, 5e4, 2e4]:
    lc = np.log1p(C)
    sat = []
    dd = []
    for b in BLOKLAR:
        lp = bl[b]["lp"].to_numpy()
        ly = bl[b]["ly"].to_numpy()
        dd.append(skorla(np.minimum(lp, lc), ly) - taban[b])
        sat.append(int((lp > lc).sum()))
    p(f"{C:12.0f}" + "".join(f"{x:>12.6f}" for x in dd) + f"{sum(sat):12,}")

p()
p("=" * 92)
p("2) ALT TABAN (taban F, kWh)")
p("=" * 92)
p(f"{'F':>12}" + "".join(f"{b:>12}" for b in BLOKLAR) + f"{'etkilenen':>12}")
for F in [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]:
    lf = np.log1p(F)
    dd, sat = [], []
    for b in BLOKLAR:
        lp = bl[b]["lp"].to_numpy()
        ly = bl[b]["ly"].to_numpy()
        dd.append(skorla(np.maximum(lp, lf), ly) - taban[b])
        sat.append(int((lp < lf).sum()))
    p(f"{F:12.2f}" + "".join(f"{x:>12.6f}" for x in dd) + f"{sum(sat):12,}")

p()
p("=" * 92)
p("3) KURESEL SEVIYE (log1p uzayinda sabit delta)")
p("=" * 92)
p(f"{'delta':>12}" + "".join(f"{b:>12}" for b in BLOKLAR))
for dlt in [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15]:
    dd = [skorla(bl[b]["lp"].to_numpy() + dlt, bl[b]["ly"].to_numpy()) - taban[b] for b in BLOKLAR]
    p(f"{dlt:12.3f}" + "".join(f"{x:>12.6f}" for x in dd))
p("blok bazinda optimum delta (= -ort artik):")
for b in BLOKLAR:
    e = np.log1p(np.clip(np.expm1(bl[b]["lp"].to_numpy()), 0, None)) - bl[b]["ly"].to_numpy()
    p("  %-7s ort artik %+0.5f  -> optimum dMSE %.6f" % (b, e.mean(), -(e.mean() ** 2)))

p()
p("=" * 92)
p("4) UFUK EGIMI (blok icinde ileriye gittikce artik kayiyor mu)")
p("=" * 92)
for b in BLOKLAR:
    d = bl[b]
    lp = d["lp"].to_numpy()
    ly = d["ly"].to_numpy()
    e = np.log1p(np.clip(np.expm1(lp), 0, None)) - ly
    u = d["ufuk_gun"].to_numpy(dtype="float64")
    p(f"--- {b}  ufuk_gun min={u.min():.0f} max={u.max():.0f} ---")
    kova = np.digitize(u, [16, 31, 46, 61, 76, 91, 106])
    t = pd.DataFrame({"k": kova, "e": e, "u": u})
    g = t.groupby("k").agg(
        n=("e", "size"), ufuk=("u", "mean"), artik=("e", "mean"), mse=("e", lambda s: (s**2).mean())
    )
    p(g.to_string())
    # dogrusal ufuk yonu:  v = (u - ort(u)) olcekli
    v = (u - u.mean()) / u.std()
    Q = float((v * v).mean())
    L = -float((e * v).mean())
    p(
        "  dogrusal ufuk yonu: Q=%.5f L=%+.6f k*=%+.5f  optimum dMSE=%.6f"
        % (Q, L, L / Q, -L * L / Q)
    )
