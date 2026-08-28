"""NIHAI URETIM: tum train ile egit, GERCEK test'i tahmin et, kapi denetiminden gecir."""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from m30_ozellik import KOK, kur, yukle_ham
from m33_durust import VARSAYILAN, hizala, parca

KESIM_TEST = "2026-03-31"
EGIT_KESIMLERI = [
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


def test_matrisi(tr, te):
    gec = tr  # tum train = gecmis
    sicak = set(gec.tanim)
    X = kur(gec, te, KESIM_TEST, sicak)
    return X


def uret(cikti, tur, tohumlar=(7, 17, 27, 37, 47), **pk):
    t0 = time.time()
    tr, te = yukle_ham()
    Xs, ys = [], []
    for k in EGIT_KESIMLERI:
        r = parca(tr, k, tavan=KESIM_TEST)
        if r is None:
            continue
        Xs.append(r[0])
        ys.append(r[1])
        print(f"  kesim {k}: {len(r[0]):,} satir ({time.time() - t0:.0f}s)", flush=True)
    Xtr = pd.concat(Xs, ignore_index=True)
    ytr = np.concatenate(ys)
    del Xs, ys
    Xte = test_matrisi(tr, te)
    print(
        f"  TEST matrisi: {len(Xte):,} satir, soguk %{100 * Xte.soguk.mean():.1f} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    Xtr, Xte = hizala(Xtr, Xte)
    Xte = Xte[Xtr.columns]
    p = dict(VARSAYILAN)
    p.update(pk)
    tahminler = []
    for s in tohumlar:
        p["seed"] = s
        p["bagging_seed"] = s + 1
        p["feature_fraction_seed"] = s + 2
        m = lgb.train(p, lgb.Dataset(Xtr, ytr), tur)
        tahminler.append(m.predict(Xte))
        print(f"  tohum {s} bitti ({time.time() - t0:.0f}s)", flush=True)
    lg = np.mean(tahminler, axis=0)
    y = np.expm1(lg)
    y = np.clip(y, 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    yol = os.path.join(KOK, "submissions", cikti)
    out.to_csv(yol, index=False)
    np.save(cikti.replace(".csv", "_log.npy"), lg)
    # KAPI DENETIMI
    ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
    kapi = dict(
        satir=len(out),
        beklenen=len(ss),
        id_birebir=bool((out.id.values == ss.iloc[:, 0].values).all()),
        nan=int(out.tuketim.isna().sum()),
        negatif=int((out.tuketim < 0).sum()),
        min=float(out.tuketim.min()),
        maks=float(out.tuketim.max()),
        log_ort=float(np.log1p(out.tuketim).mean()),
    )
    print("KAPI:", json.dumps(kapi))
    assert (
        kapi["satir"] == kapi["beklenen"]
        and kapi["id_birebir"]
        and kapi["nan"] == 0
        and kapi["negatif"] == 0
    )
    print(f"YAZILDI {yol}  ({time.time() - t0:.0f}s)")
    return kapi


if __name__ == "__main__":
    uret(
        sys.argv[1] if len(sys.argv) > 1 else "tuketim_m1_ileri.csv",
        tur=int(sys.argv[2]) if len(sys.argv) > 2 else 170,
        objective="huber",
        alpha=2.0,
        lambda_l2=20.0,
    )
