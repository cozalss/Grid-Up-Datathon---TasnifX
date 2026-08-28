"""D: envanterin SOGUK payi ve onarilmis olcutun bu paya soyledigi."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
GON = KOK / "submissions"
test = pd.read_parquet(KOK / "data/interim/deney/test.parquet")
soguk = (test["soguk_mu"] == 1).to_numpy()
ids = test["id"].to_numpy()


def lg(f):
    d = pd.read_csv(GON / f)
    k = [c for c in d.columns if c != "id"][0]
    s = pd.Series(np.log1p(d[k].to_numpy("float64")), index=d["id"].to_numpy())
    return s.reindex(ids).to_numpy()


N = {
    "v83": "tuketim_v83_sicak_optimum.csv",
    "v93": "tuketim_v93_gram_optimum.csv",
    "v90": "tuketim_v90_temiz_sota.csv",
    "v102": "tuketim_v102_kappa_optimum.csv",
    "P1": "tuketim_p1_sicak_ilce.csv",
    "P2": "tuketim_p2_sicak_seviye.csv",
    "P3": "tuketim_p3_soguk_seviye.csv",
    "P4": "tuketim_p4_sicak_ay.csv",
    "P5": "tuketim_p5_soguk_kva.csv",
    "yas": "tuketim_prob_yas790.csv",
    "v82": "tuketim_v82_ayirici.csv",
    "v99": "tuketim_v99_mimari_sekil.csv",
    "B": "tuketim_v96_grupb_optimum.csv",
    "bos": "tuketim_v94_bosluk_oncesi.csv",
}
F = {k: lg(v) for k, v in N.items()}
n = len(ids)
YON = [
    ("P1c", F["P1"] - F["v93"]),
    ("P3c", F["P3"] - F["v93"]),
    ("Bc", F["B"] - F["v93"]),
    ("bosc", F["bos"] - F["v93"]),
    ("v90", F["v90"] - F["v83"]),
    ("P2", F["P2"] - F["v93"]),
    ("yas", F["yas"] - F["v93"]),
    ("v99", F["v99"] - F["v90"]),
    ("P4", F["P4"] - F["v93"]),
    ("v82", F["v82"] - F["v83"]),
    ("P5", F["P5"] - F["v93"]),
]
print(f"{'yon':7}{'Q_dosya':>12}{'Q_soguk_payi':>14}{'Q_sicak_payi':>14}")
for ad, u in YON:
    Q = float(np.mean(u**2))
    qs = float(np.sum(u[soguk] ** 2) / n)
    print(
        f"{ad:7}{Q:>12.7f}{100 * qs / Q if Q else 0:>13.1f}%{100 * (1 - qs / Q) if Q else 0:>13.1f}%"
    )
