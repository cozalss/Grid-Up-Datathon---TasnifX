"""Q3 -- kaydedilmis L vektorlerinden daha SIKI kirpilmis surumler (kurtoz <=10 hedefi)."""

import json
import os
import sys

import numpy as np

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
import z1_ortak as Z  # noqa: E402

if __name__ == "__main__":
    import pandas as pd

    tr = pd.read_csv(
        os.path.join(Z.KOK, "data/raw/train.csv"), parse_dates=["tarih"], dtype={"tanim": str}
    )
    te = pd.read_csv(
        os.path.join(Z.KOK, "data/raw/test.csv"), parse_dates=["tarih"], dtype={"tanim": str}
    )
    msk = Z.maskeler(tr, te)
    A6 = Z.taban()
    rap = {}
    for etiket, npy, kirp, cikis in [
        ("q1c", "q1_A.npy", 0.75, "tuketim_q1c_kapasite_siki.csv"),
        ("q1d", "q1_B.npy", 0.75, "tuketim_q1d_kuantil38_siki.csv"),
    ]:
        yol = os.path.join(BURA, npy)
        if not os.path.exists(yol):
            print(f"YOK {npy}")
            continue
        rap[etiket] = Z.bitir(np.load(yol), te, msk, A6, cikis, kirp=kirp)
    json.dump(rap, open(os.path.join(BURA, "q3_kirp.json"), "w"), indent=1)
