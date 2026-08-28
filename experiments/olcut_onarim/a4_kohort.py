"""Soguk kohort anatomisi: bloklar vs TEST (onarim kohortun kendisini degistirmez)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import tuketim_model as tm  # noqa

ONB = KOK / "data" / "interim" / "deney"
egitim = pd.read_parquet(ONB / "egitim.parquet")
test = pd.read_parquet(ONB / "test.parquet")
G = tm.GRUP
print(
    f"{'kohort':8}{'satir':>10}{'trafo':>8}{'satir/trafo':>13}{'guc med':>10}"
    f"{'guc ort':>10}{'y=0 %':>9}{'y med':>10}{'y ort':>11}"
)
for ad, df in [(b.ad, egitim[egitim["_blok"] == b.ad]) for b in tm.BLOKLAR] + [("TEST", test)]:
    s = df[df["soguk_mu"] == 1]
    if ad == "TEST":
        y = None
    else:
        y = s[tm.HEDEF].to_numpy(dtype="float64")
    n, nt = len(s), s[G].nunique()
    gm = s.groupby(G, observed=True)["guc"].first()
    print(
        f"{ad:8}{n:>10,}{nt:>8,}{n / nt:>13.1f}{gm.median():>10.0f}{gm.mean():>10.0f}"
        + (
            f"{100 * np.mean(y == 0):>9.1f}{np.median(y):>10.1f}{y.mean():>11.1f}"
            if y is not None
            else f"{'-':>9}{'-':>10}{'-':>11}"
        )
    )
# SSE payi
print()
for b in tm.BLOKLAR:
    df = egitim[egitim["_blok"] == b.ad]
    s = (df["soguk_mu"] == 1).to_numpy()
    print(f"  {b.ad}: soguk satir payi %{100 * s.mean():.1f}")
print(f"  TEST : soguk satir payi %{100 * (test['soguk_mu'] == 1).mean():.1f}")
