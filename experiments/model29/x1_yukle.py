"""x1 -- BAGIMSIZ yukleyici. Kendi onbellegim, kendi denetimlerim."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
KOK = Path(__file__).resolve().parents[2]
GON = KOK / "submissions"
BURA = Path(__file__).resolve().parent
CACHE = Path(os.environ["XCACHE"])
CACHE.mkdir(parents=True, exist_ok=True)
SKOR = json.loads((BURA / "olculmus_skorlar.json").read_text(encoding="utf-8"))


def oku(yol):
    yol = Path(yol)
    ad = yol.name
    c = CACHE / (ad + ".npy")
    if c.exists():
        return np.load(c)
    t = pd.read_csv(yol)
    v = t.iloc[:, 1].to_numpy(dtype="float64")
    x = np.log1p(v)
    np.save(c, x)
    return x


def okuham(yol):
    t = pd.read_csv(yol)
    return t.iloc[:, 0].to_numpy(), t.iloc[:, 1].to_numpy(dtype="float64")


def matris():
    dosyalar = sorted(d for d in SKOR if d != "gun1_baseline.csv")
    c = CACHE / "X25.npy"
    if c.exists():
        return dosyalar, np.load(c), np.array([SKOR[d] for d in dosyalar])
    N = len(oku(GON / dosyalar[0]))
    X = np.empty((len(dosyalar), N))
    for i, d in enumerate(dosyalar):
        X[i] = oku(GON / d)
        print("yuklendi", d, flush=True)
    np.save(c, X)
    return dosyalar, X, np.array([SKOR[d] for d in dosyalar])


if __name__ == "__main__":
    dos, X, s = matris()
    print(X.shape, s.shape)
