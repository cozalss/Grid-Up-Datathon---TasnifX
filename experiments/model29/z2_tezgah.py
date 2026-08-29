"""Ortak ozellik matrisi onbellegi -- z2_kantil ve z3_sinir ayni matrisi kullanir.

m30_ozellik/m33_durust tezgahinin URETIM kurulumu (m40_uret ile birebir ayni
kesimler ve ayni kur() cagrisi), tek fark: matris bir kez kurulup diske yazilir.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from m30_ozellik import kur, yukle_ham  # noqa: E402
from m33_durust import hizala  # noqa: E402

KESIM = "2026-03-31"
KESIMLER = [
    "2025-03-31",
    "2025-04-30",
    "2025-05-31",
    "2025-06-30",
    "2025-07-31",
    "2025-08-31",
    "2025-09-30",
    "2025-10-31",
    "2025-11-30",
    "2025-12-31",
]
ONBELLEK = os.path.join(BURA, "z2_onbellek")


def matrisler():
    """(Xtr, ytr, Xte, tr, te) -- varsa onbellekten okur."""
    ftr = os.path.join(ONBELLEK, "Xtr.parquet")
    fte = os.path.join(ONBELLEK, "Xte.parquet")
    fy = os.path.join(ONBELLEK, "ytr.npy")
    t0 = time.time()
    tr, te = yukle_ham()
    if os.path.exists(ftr) and os.path.exists(fte) and os.path.exists(fy):
        Xtr = pd.read_parquet(ftr)
        Xte = pd.read_parquet(fte)
        ytr = np.load(fy)
        print(f"onbellekten okundu {Xtr.shape} / {Xte.shape} ({time.time() - t0:.0f}s)", flush=True)
        return Xtr, ytr, Xte, tr, te
    os.makedirs(ONBELLEK, exist_ok=True)
    Xs, ys = [], []
    for k in KESIMLER:
        kk = pd.Timestamp(k)
        son = min(kk + pd.DateOffset(months=4), pd.Timestamp(KESIM))
        gec = tr[tr.tarih <= kk]
        hed = tr[(tr.tarih > kk) & (tr.tarih <= son)]
        if not len(hed):
            continue
        Xs.append(kur(gec, hed, k, set(gec.tanim)))
        ys.append(hed.tuketim.values)
        print(f"  kesim {k}: {len(hed):,} satir ({time.time() - t0:.0f}s)", flush=True)
    Xtr = pd.concat(Xs, ignore_index=True)
    ytr = np.concatenate(ys)
    del Xs, ys
    Xte = kur(tr, te, KESIM, set(tr.tanim))
    Xtr, Xte = hizala(Xtr, Xte)
    Xte = Xte[Xtr.columns]
    Xtr.to_parquet(ftr)
    Xte.to_parquet(fte)
    np.save(fy, ytr)
    print(f"kuruldu ve yazildi {Xtr.shape} / {Xte.shape} ({time.time() - t0:.0f}s)", flush=True)
    return Xtr, ytr, Xte, tr, te
