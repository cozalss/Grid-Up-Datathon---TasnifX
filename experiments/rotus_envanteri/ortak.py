"""Ortak yukleyiciler -- rotus envanteri / grup B olcumleri."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
SUB = KOK / "submissions"
A_SINIRI = pd.Timestamp("2026-03-27")


def ilce_ayikla(lok: pd.Series) -> pd.Series:
    """lokasyon alani sabit bicimli DEGIL (docs/51 §7). Son parca = ilce."""
    return lok.str.split(">").str[-1].str.strip()


@lru_cache(maxsize=1)
def train() -> pd.DataFrame:
    tr = pd.read_csv(KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str})
    tr["tarih"] = pd.to_datetime(tr["tarih"])
    tr["ilce"] = ilce_ayikla(tr["lokasyon"])
    return tr


@lru_cache(maxsize=1)
def test() -> pd.DataFrame:
    te = pd.read_csv(KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str})
    te["tarih"] = pd.to_datetime(te["tarih"])
    te["ilce"] = ilce_ayikla(te["lokasyon"])
    return te


@lru_cache(maxsize=32)
def sub(ad: str) -> pd.DataFrame:
    d = pd.read_csv(SUB / ad, encoding="utf-8")
    return d


def hizala(ad: str, te: pd.DataFrame) -> np.ndarray:
    """Gonderim dosyasini test.csv satir sirasina hizala; tuketim dizisi dondur."""
    s = sub(ad)
    m = te[["id"]].merge(s, on="id", how="left", validate="one_to_one")
    if m["tuketim"].isna().any():
        raise RuntimeError(f"{ad}: eksik id")
    return m["tuketim"].to_numpy(dtype=float)


def bagil_maske(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """docs/51 Duzeltme 1: bagil toleransli fark maskesi."""
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-9) >= tol


def yon_enerjisi(a: np.ndarray, b: np.ndarray) -> dict:
    """log1p uzayinda a-b yon enerjisi Q = ||d||^2 / n."""
    d = np.log1p(np.clip(a, 0, None)) - np.log1p(np.clip(b, 0, None))
    n = len(d)
    return {
        "degisen_satir": int(bagil_maske(a, b).sum()),
        "norm2": float(d @ d),
        "Q": float(d @ d / n),
        "ort_fark": float(d.mean()),
        "maxabs": float(np.abs(d).max()),
    }
