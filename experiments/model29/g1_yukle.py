"""g1 -- olculmus 25 gonderimi log1p uzayinda yukle, onbellege al.

KAGGLE'A HICBIR SEY GONDERILMEZ. Yalnizca yerel dosya okuma.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

KOK = Path(__file__).resolve().parents[2]
GON = KOK / "submissions"
BURA = Path(__file__).resolve().parent
SKOR = json.loads((BURA / "olculmus_skorlar.json").read_text(encoding="utf-8"))

# gun1_baseline.csv farkli format (id=R00320, kolon=hedef) -> DISLANDI
DISLA = {"gun1_baseline.csv"}

# bugun olculen YENI yonler
YENI = {"tuketim_m4_hava_capali.csv", "tuketim_p51_sicak05.csv", "tuketim_m6_ikiyon.csv"}


def kisa(dosya: str) -> str:
    s = dosya.replace("tuketim_", "").replace(".csv", "")
    return s


def yukle():
    npy = BURA / "g1_X.npy"
    meta = BURA / "g1_meta.json"
    dosyalar = sorted(d for d in SKOR if d not in DISLA)
    adlar = [kisa(d) for d in dosyalar]
    skorlar = np.array([SKOR[d] for d in dosyalar], dtype="float64")
    if npy.exists() and meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        if m["dosyalar"] == dosyalar:
            return adlar, dosyalar, np.load(npy), skorlar

    ids = pd.read_csv(GON / dosyalar[0], usecols=["id"])["id"].to_numpy()
    X = np.empty((len(dosyalar), ids.size), dtype="float64")
    for k, d in enumerate(dosyalar):
        t = pd.read_csv(GON / d)
        assert list(t.columns) == ["id", "tuketim"], (d, list(t.columns))
        assert (t["id"].to_numpy() == ids).all(), f"{d}: id sirasi farkli"
        v = t["tuketim"].to_numpy(dtype="float64")
        assert np.isfinite(v).all() and (v >= 0).all(), d
        X[k] = np.log1p(v)
    np.save(npy, X)
    meta.write_text(json.dumps({"dosyalar": dosyalar, "n": int(ids.size)}), encoding="utf-8")
    return adlar, dosyalar, X, skorlar


if __name__ == "__main__":
    adlar, dosyalar, X, skorlar = yukle()
    print(f"K={X.shape[0]} dosya, N={X.shape[1]} satir")
    print(f"log1p araligi [{X.min():.6f}, {X.max():.6f}]  NaN={int(np.isnan(X).sum())}")
    for a, d, s in zip(adlar, dosyalar, skorlar):
        print(f"  {a:22s} {s:.5f}  {'YENI' if d in YENI else ''}")
