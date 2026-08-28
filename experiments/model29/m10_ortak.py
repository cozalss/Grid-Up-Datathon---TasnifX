"""SICAK seviye kestirimi deneyleri icin ortak iskelet."""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m1_geriteste import kes

KESIMLER = ["2025-08-31", "2025-09-30", "2025-10-31", "2025-11-30"]
BURASI = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(BURASI, "m10_sicak_seviye.json")


def hazirla(tr, kesim, ufuk_ay=4):
    """gec (tum gecmis, ly kolonlu) ve hed (SADECE SICAK satirlar) dondurur."""
    k = pd.Timestamp(kesim)
    gec, hed = kes(tr, kesim, ufuk_ay)
    gec = gec.copy()
    gec["ly"] = np.log1p(gec.tuketim)
    hed = hed[~hed.soguk].copy()
    hed["ly"] = np.log1p(hed.tuketim)
    hed["ay_ofset"] = (hed.tarih.dt.year - k.year) * 12 + (hed.tarih.dt.month - k.month)
    return gec, hed


def puan(hed, pred):
    """pred: log1p uzayinda tahmin (numpy dizi). Toplam ve ay-ofsetine gore RMSLE."""
    e2 = (np.asarray(pred, float) - hed.ly.values) ** 2
    out = {"rmsle": float(np.sqrt(e2.mean())), "n": int(len(e2))}
    for a in sorted(hed.ay_ofset.unique()):
        m = hed.ay_ofset.values == a
        out[f"ay{a}"] = float(np.sqrt(e2[m].mean()))
    return out


def pencere_seviye(gec, kesim, gun, stat="mean", trim=0.1):
    """Son `gun` gunun ly istatistigi (gun=None -> tum gecmis)."""
    k = pd.Timestamp(kesim)
    g = gec if gun is None else gec[gec.tarih > k - pd.Timedelta(days=gun)]
    if stat == "mean":
        return g.groupby("tanim").ly.mean()
    if stat == "median":
        return g.groupby("tanim").ly.median()
    if stat == "trim":

        def t(v):
            v = np.sort(v.values)
            n = len(v)
            c = int(np.floor(n * trim))
            return v[c : n - c].mean() if n - 2 * c > 0 else v.mean()

        return g.groupby("tanim").ly.apply(t)
    raise ValueError(stat)


def geri_dolgu(hed, *seviyeler, kok=None):
    """Sirayla seviye serilerinden esle; hicbiri yoksa kok (skaler)."""
    p = pd.Series(np.nan, index=hed.index)
    for s in seviyeler:
        if s is None:
            continue
        p = p.fillna(hed.tanim.map(s))
    if kok is not None:
        p = p.fillna(kok)
    return p.values


def json_yaz(anahtar, veri):
    d = {}
    if os.path.exists(JSON):
        with open(JSON, encoding="utf-8") as f:
            d = json.load(f)
    d[anahtar] = veri
    with open(JSON, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
