"""ONARILMIS SICAK OLCUT -- TEST geometrisiyle ayni katlarda olcum.

01_geometri.py'nin bulgusu:

    blok    ileri kat payi   ozet-kapsama
    yaz25        %92,8           %92,8
    guz25        %64,9           %64,9
    kis26         %0,0            %0,0
    TEST          %0,0            %0,0

yaz25/guz25 modelleri hedef donemin GELECEGINDEN egitiliyor; TEST modeli
egitilmiyor. Bu yuzden o iki blogun ARTIK YAPISI TEST'inkiyle ayni turden
degil ve uc-blok isaret kapisi yanlis konfigurasyonu seciyor.

ONARIM: hukum yalniz kis26'dan verilir (kural 13'un sicak karsiligi), ve
kis26 icindeki karar de ILERI (past->future) kurulur:

    OGREN = 2025-12-01 .. 2026-01-31   (62 gun)
    SINA  = 2026-02-01 .. 2026-03-31   (59 gun)

Bu, TEST'in geometrisinin kucuk olcekli ikizidir: temiz taban, gecmisten
ogrenilen ayar, gelecege uygulama. Guven trafo-kumeli bootstrap ile.

Kirpma: uretim olcutu ``np.clip(np.expm1(.),0,None)`` -> log uzayinda
``max(r + lgc, 0)``. ortak.mse ile ayni.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sicak_kaldirac"))
from ortak import AGIRLIK, C_GUN, KUYRUK_DELTA, gun_etkisi  # noqa: E402

BOLME = pd.Timestamp("2026-02-01")


def hazirla(b) -> None:
    """Bloga tanilayici kolonlari ekler (yerinde)."""
    c = b.cerceve
    c["seviye_d"] = pd.qcut(c["t_log_ort"].to_numpy(), 20, labels=False, duplicates="drop")
    c["seviye_d10"] = pd.qcut(c["t_log_ort"].to_numpy(), 10, labels=False, duplicates="drop")
    c["gecmis_k"] = np.digitize(c["gecmis_gun"].to_numpy(), [7, 31, 91, 181, 366])
    c["sifir_k"] = np.digitize(c["t_sifir_orani"].fillna(0).to_numpy(), [0.01, 0.05, 0.2, 0.5, 0.9])
    c["ilce_kova"] = c["ilce"].astype(str) + "|" + c["kova"].astype(str)


def mse_alt(b, r: np.ndarray, m: np.ndarray) -> float:
    e = b.lgy[m] - np.maximum(r[m] + b.lgc[m], 0.0)
    return float((e * e).mean())


def hata2(b, r: np.ndarray) -> np.ndarray:
    e = b.lgy - np.maximum(r + b.lgc, 0.0)
    return e * e


def delta_coz(b, r: np.ndarray, m: np.ndarray) -> float:
    """Kirpma altinda ``m`` maskesi uzerinde optimum kuresel seviye kaymasi."""
    d0 = float((b.lgy[m] - b.lgc[m] - r[m]).mean())
    en, en_m = d0, mse_alt(b, r + d0, m)
    adim = 0.08
    for _ in range(6):
        for d in np.arange(en - 4 * adim, en + 4.001 * adim, adim):
            v = mse_alt(b, r + float(d), m)
            if v < en_m:
                en, en_m = float(d), v
        adim /= 4.0
    return en


def ham_r(b, agirlik: dict[str, float] | None = None) -> np.ndarray:
    w = AGIRLIK if agirlik is None else agirlik
    pay = sum(w.values())
    s = np.zeros(b.n, dtype="float64")
    for a, k in w.items():
        s += k * b.ham[a]
    return s / pay - b.lgc


def zincir(b, *, agirlik=None, gun_olcek=C_GUN, kuyruk=KUYRUK_DELTA) -> np.ndarray:
    """URETIM ZINCIRI, kuresel seviye kalibrasyonu YAPILMADAN."""
    r = ham_r(b, agirlik)
    be = gun_etkisi(b.cerceve["tanim"].to_numpy(), b.cerceve["tarih"].to_numpy(), r)
    e = pd.Series(b.cerceve["tarih"].to_numpy()).map(be).to_numpy(dtype="float64")
    e = e - e.mean()
    r = r + (gun_olcek - 1.0) * e
    return r + kuyruk * b.cerceve["kuyruk"].to_numpy(dtype="float64")


def grup_ofseti(b, r: np.ndarray, m: np.ndarray, anah: str, n0: float = 200.0) -> dict:
    """``m`` maskesindeki satirlardan buzulmus grup ofseti ogrenir."""
    e = b.lgy[m] - (r[m] + b.lgc[m])
    e = e - e.mean()
    g = pd.Series(b.cerceve[anah].to_numpy()[m])
    agg = pd.DataFrame({"e": e}).groupby(g, observed=True)["e"].agg(["mean", "size"])
    return (agg["mean"] * agg["size"] / (agg["size"] + n0)).to_dict()


def ofset_uygula(b, r: np.ndarray, harita: dict, anah: str, m: np.ndarray, kat=1.0) -> np.ndarray:
    d = pd.Series(b.cerceve[anah].to_numpy()).map(harita).fillna(0.0).to_numpy(dtype="float64")
    d = d - d[m].mean()  # SINA bilesimine gore merkezle -> kuresel seviye degismez
    out = r.copy()
    out[m] = r[m] + kat * d[m]
    return out


def bootstrap(b, r0: np.ndarray, r1: np.ndarray, m: np.ndarray, B: int = 1000, tohum: int = 7):
    """Trafo-kumeli bootstrap. Doner: (dMSE, lo, hi, kazanan_pay, n_trafo)."""
    d = hata2(b, r1)[m] - hata2(b, r0)[m]
    t = pd.Series(b.cerceve["tanim"].to_numpy()[m])
    g = pd.DataFrame({"d": d}).groupby(t, observed=True)["d"].agg(["sum", "size"])
    S = g["sum"].to_numpy()
    N = g["size"].to_numpy(dtype="float64")
    k = len(S)
    rng = np.random.default_rng(tohum)
    idx = rng.integers(0, k, size=(B, k))
    dm = S[idx].sum(axis=1) / N[idx].sum(axis=1)
    lo, hi = np.percentile(dm, [2.5, 97.5])
    kazanan = float((S < 0).mean())
    return float(d.mean()), float(lo), float(hi), kazanan, k
