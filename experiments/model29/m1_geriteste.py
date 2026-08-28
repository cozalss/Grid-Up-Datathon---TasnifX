"""Test'in yapisini birebir taklit eden geri-test (backtest) iskeleti."""

import os

import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(KOK, "data", "raw", "train.csv")


def yukle():
    tr = pd.read_csv(TR, parse_dates=["tarih"])
    tr["il"] = tr.lokasyon.str.split(">").str[0]
    tr["bolge"] = tr.lokasyon.str.split(">").str[1]
    tr["ilce"] = tr.lokasyon.str.split(">").str[2]
    return tr


def rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(y)) ** 2)))


def kes(tr, kesim, ufuk_ay=4):
    kesim = pd.Timestamp(kesim)
    son = kesim + pd.DateOffset(months=ufuk_ay)
    gec = tr[tr.tarih <= kesim]
    hed = tr[(tr.tarih > kesim) & (tr.tarih <= son)].copy()
    gorulen = set(gec.tanim)
    hed["soguk"] = ~hed.tanim.isin(gorulen)
    return gec, hed


if __name__ == "__main__":
    tr = yukle()
    for kesim in ["2025-09-30", "2025-10-31", "2025-11-30"]:
        gec, hed = kes(tr, kesim)
        n = len(hed)
        print(
            f"kesim {kesim}: gecmis {len(gec):,} satir / {gec.tanim.nunique():,} trafo | "
            f"hedef {n:,} satir / {hed.tanim.nunique():,} trafo | soguk %{100 * hed.soguk.mean():.1f} "
            f"({hed.soguk.sum():,} satir, {hed[hed.soguk].tanim.nunique():,} trafo)"
        )
