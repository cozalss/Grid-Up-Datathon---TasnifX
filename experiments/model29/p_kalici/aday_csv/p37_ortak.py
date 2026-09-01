# -*- coding: utf-8 -*-
"""YON 4 -- SOGUK KOHORT ortak arac takimi."""
import os
import sys
import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
sys.path.insert(0, os.path.join(KOK, "experiments/model29/p_kalici"))
import p27_ortak as P  # noqa: E402

DN = os.path.join(KOK, "data/interim/deney")
BLOKLAR = ("yaz25", "guz25", "kis26")

_BLOK = {}
_EZBER = {}


def blok(b):
    if b not in _BLOK:
        d = P.blok(b)
        d = d.reset_index(drop=True)
        _BLOK[b] = d
    return _BLOK[b]


def ezber_maskesi(b):
    """p30 usulu: bu blogun soguk satirlarinin trafosu DIGER bloklarda gorunuyor mu."""
    if b not in _EZBER:
        E = P.egitim()
        egit_tanim = set(E.loc[E._blok != b, "tanim"].unique())
        d = blok(b)
        _EZBER[b] = d.tanim.isin(egit_tanim).to_numpy()
    return _EZBER[b]


def rho_olc(d, delta, w=None, kume=None):
    """Brifing tanimi: u = delta/sqrt(E_w[delta^2]); rho = E_w[r*u]. Kumeli SE."""
    if w is None:
        w = P.agirlik(d)
    r = d.r.values.astype(np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    sw = w.sum()
    norm = np.sqrt(np.sum(w * delta * delta) / sw)
    if not np.isfinite(norm) or norm <= 0:
        return dict(rho=0.0, se=0.0, t=0.0, norm=0.0)
    u = delta / norm
    rho = float(np.sum(w * r * u) / sw)
    # kumeli SE (trafo)
    if kume is None:
        kume = d.tanim.values
    g = w * r * u
    df = pd.DataFrame({"k": kume, "g": g, "w": w})
    ag = df.groupby("k", sort=False).sum()
    resid = ag.g.values - rho * ag.w.values
    se = float(np.sqrt(np.sum(resid * resid)) / sw)
    return dict(rho=rho, se=se, t=(rho / se if se > 0 else 0.0), norm=float(norm))


def skor(rho, mse_taban=1.0013719):
    return float(np.sqrt(max(mse_taban - rho * rho, 0.0)))
