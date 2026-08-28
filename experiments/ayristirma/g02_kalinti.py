"""v101'in gercek bilesimini bul: kalinti r nedir?"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
G = KOK / "submissions"


def lg(a):
    d = pd.read_csv(G / a)
    return np.log1p(d["tuketim"].to_numpy("float64"))


F = {
    k: lg(v)
    for k, v in {
        "v83": "tuketim_v83_sicak_optimum.csv",
        "v93": "tuketim_v93_gram_optimum.csv",
        "P1": "tuketim_p1_sicak_ilce.csv",
        "P3": "tuketim_p3_soguk_seviye.csv",
        "B96": "tuketim_v96_grupb_optimum.csv",
        "B95": "tuketim_v95_gram_grupb.csv",
        "B91": "tuketim_v91_grupb_kaldirma.csv",
        "bos": "tuketim_v94_bosluk_oncesi.csv",
        "v101": "tuketim_v101_hepsi.csv",
        "P2": "tuketim_p2_sicak_seviye.csv",
        "P4": "tuketim_p4_sicak_ay.csv",
        "P5": "tuketim_p5_soguk_kva.csv",
    }.items()
}
n = F["v83"].size
d = F["v101"] - F["v83"]


def q(x):
    return float(x @ x) / n


print(f"Q(v101-v83)={q(d):.7f}")
cands = {
    "v93": F["v93"] - F["v83"],
    "P1v93": F["P1"] - F["v93"],
    "P3v93": F["P3"] - F["v93"],
    "B96v93": F["B96"] - F["v93"],
    "B95v93": F["B95"] - F["v93"],
    "B91v93": F["B91"] - F["v93"],
    "bosv93": F["bos"] - F["v93"],
    "P1v83": F["P1"] - F["v83"],
    "P3v83": F["P3"] - F["v83"],
    "B96v83": F["B96"] - F["v83"],
    "bosv83": F["bos"] - F["v83"],
    "P2v93": F["P2"] - F["v93"],
    "P4v93": F["P4"] - F["v93"],
    "P5v93": F["P5"] - F["v93"],
}
for k, v in cands.items():
    print(f"  {k:9s} Q={q(v):.7f}  cos(d)={float(v @ d) / n / np.sqrt(q(v) * q(d)):+.4f}")
# en iyi uyum: en kucuk kalinti veren altkume
import itertools

best = []
for r_ in range(3, 7):
    for combo in itertools.combinations(
        [
            "v93",
            "P1v93",
            "P3v93",
            "B96v93",
            "B95v93",
            "B91v93",
            "bosv93",
            "P2v93",
            "P4v93",
            "P5v93",
        ],
        r_,
    ):
        s = sum(cands[c] for c in combo)
        best.append((q(d - s), combo))
best.sort()
print("\nEN IYI TOPLAM UYUMLARI (kalinti Q):")
for e, c in best[:8]:
    print(f"  {e:.3e}  {c}")
