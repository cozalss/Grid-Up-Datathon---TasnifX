"""Yon winsorizasyonu -- yuksek kaldiracli az sayida satirin acisi bozmasini onler.

Klasik ve amnezik adaylarda satirlarin %0,4-0,6'si (|f| > 3) yonun enerjisinin
%23-34'unu tasiyor. m4 (LB'de kaliteli cikmis yon) icin ayni bant %3.
Modelin acikca yetkisi disinda kaldigi bu satirlarda farki e^2 ile sinirlamak
gurultu temizligi degil, KALDIRAC sinirlamasidir: L'ye katkisi supheli, Q'ya
katkisi devasa olan satirlar budanir. Ardindan rejim capasi yenilenir.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))
S = os.path.join(KOK, "submissions")

te = pd.read_csv(
    os.path.join(KOK, "data/raw/test.csv"),
    parse_dates=["tarih"],
    dtype={"tanim": str},
    usecols=["id", "tanim", "tarih"],
)
tr = pd.read_csv(
    os.path.join(KOK, "data/raw/train.csv"),
    parse_dates=["tarih"],
    dtype={"tanim": str},
    usecols=["tanim", "tarih"],
)
_ilk = tr.groupby("tanim").tarih.min()
SOGUK = (~te.tanim.isin(set(tr.tanim))).to_numpy()
KUYRUK = (~SOGUK) & (te.tanim.map(_ilk) >= pd.Timestamp("2026-03-26")).to_numpy()
CEK = (~SOGUK) & (~KUYRUK)
A6 = np.log1p(pd.read_csv(os.path.join(S, "tuketim_m6_ikiyon.csv")).tuketim.values)
SS = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))


def kirp(giris, cikis, c=2.0):
    f = np.log1p(pd.read_csv(os.path.join(S, giris)).tuketim.values) - A6
    notr = f == 0.0
    tot = float((f**2).sum())
    f = np.clip(f, -c, c)
    for m in (SOGUK, KUYRUK, CEK):
        mm = m & ~notr
        if mm.sum():
            f[mm] -= float(f[mm].mean())
    f[notr] = 0.0
    y = np.clip(np.expm1(A6 + f), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    out.to_csv(os.path.join(S, cikis), index=False)
    kapi = dict(
        satir=len(out),
        id_birebir=bool((out.id.values == SS.iloc[:, 0].values).all()),
        nan=int(out.tuketim.isna().sum()),
        negatif=int((out.tuketim < 0).sum()),
    )
    assert (
        kapi["satir"] == 714688 and kapi["id_birebir"] and not kapi["nan"] and not kapi["negatif"]
    )
    print(
        f"  {giris} -> {cikis}  kalan Q payi {float((f**2).sum()) / tot:.3f}  Q={float((f**2).mean()):.5f}"
    )
    return dict(kapi=kapi, kirpma=c, kalan_pay=float((f**2).sum()) / tot)


if __name__ == "__main__":
    ISLER = [
        ("tuketim_y43_mevsimsel_temiz.csv", "tuketim_y45_mevsimsel_kirpik.csv"),
        ("tuketim_y41_amnezik_temiz.csv", "tuketim_y46_amnezik_kirpik.csv"),
    ]
    if len(sys.argv) > 2:
        ISLER = [(sys.argv[1], sys.argv[2])]
    rap = {c: kirp(g, c) for g, c in ISLER}
    json.dump(rap, open(os.path.join(BURA, "y2_kirp.json"), "w"), indent=1)
