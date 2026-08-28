"""B: TRAFO-KUMELI BOOTSTRAP KAPISI + son islem zinciri.

Mevcut kirpma tablosu (K=0/1/5/10/25/50) yerine standart kapi:
    (1) dMSE'nin trafo-kumeli bootstrap %95 CI'si SIFIRI ICERMEYECEK
    (2) KAZANAN TRAFO ORANI >= %60
Referans: kabul edilmis soguk gun ekseni mudahalesi %65,8 veriyordu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

B_VARSAYILAN = 1000
KAZANAN_ESIK = 0.60


def kare_hatalar(y: np.ndarray, lg: np.ndarray) -> np.ndarray:
    """URETIM OLCUTU: np.clip(np.expm1(.),0,None) kirpmasi DAHIL."""
    t = np.clip(np.expm1(lg), 0.0, None)
    return (np.log1p(t) - np.log1p(y)) ** 2


def trafo_indeksi(meta: pd.DataFrame):
    kod, tekil = pd.factorize(meta["tanim"])
    n = len(tekil)
    sira = np.argsort(kod, kind="stable")
    sinir = np.searchsorted(kod[sira], np.arange(n + 1))
    return kod, n, sira, sinir


def bootstrap(
    e_taban: np.ndarray,
    e_aday: np.ndarray,
    meta: pd.DataFrame,
    B: int = B_VARSAYILAN,
    tohum: int = 7,
) -> dict:
    """dMSE = MSE(aday) - MSE(taban). Negatif = IYILESME."""
    kod, n, sira, sinir = trafo_indeksi(meta)
    d = e_aday - e_taban
    # trafo basina toplam ve sayim
    top = np.bincount(kod, d, minlength=n)
    say = np.bincount(kod, minlength=n).astype("float64")
    t_taban = np.bincount(kod, e_taban, minlength=n)
    t_aday = np.bincount(kod, e_aday, minlength=n)
    kazanan = float(np.mean((t_aday / say) < (t_taban / say)))
    # satir agirlikli kazanan orani
    kazanan_satir = float(np.sum(say[(t_aday / say) < (t_taban / say)]) / say.sum())
    rng = np.random.default_rng(tohum)
    idx = rng.integers(0, n, size=(B, n))
    num = top[idx].sum(axis=1)
    den = say[idx].sum(axis=1)
    dagilim = num / den
    lo, hi = np.percentile(dagilim, [2.5, 97.5])
    return {
        "dmse": float(d.sum() / len(d)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_iyilesme": float(np.mean(dagilim < 0)),
        "kazanan_trafo": kazanan,
        "kazanan_satir": kazanan_satir,
        "n_trafo": int(n),
        "n_satir": int(len(d)),
        "gecti": bool((lo < 0 and hi < 0) and kazanan >= KAZANAN_ESIK),
    }


def kirpma_tablosu(e_taban, e_aday, meta, Klar=(0, 1, 5, 10, 25, 50)) -> dict[int, float]:
    """Eski kapi -- karsilastirma icin. En cok kazandiran K trafo atilir."""
    kod, n, _, _ = trafo_indeksi(meta)
    d = e_aday - e_taban
    tk = np.bincount(kod, d, minlength=n)
    sira = np.argsort(tk)  # en negatif (en cok kazandiran) once
    out = {}
    for K in Klar:
        at = set(sira[:K].tolist())
        m = ~np.isin(kod, list(at)) if K else np.ones(len(d), bool)
        out[K] = float(d[m].sum() / m.sum())
    return out


# ------------------------------------------------ URETIM SOGUK SON ISLEM
def iki_yonlu(v, bi, gi, tur=400):
    nb, ng = int(bi.max()) + 1, int(gi.max()) + 1
    mu = float(v.mean())
    a = np.zeros(nb)
    b = np.zeros(ng)
    cb = np.maximum(np.bincount(bi, minlength=nb), 1)
    cg = np.maximum(np.bincount(gi, minlength=ng), 1)
    for _ in range(tur):
        a = np.bincount(bi, v - mu - b[gi], minlength=nb) / cb
        b = np.bincount(gi, v - mu - a[bi], minlength=ng) / cg
        b -= b.mean()
    return b


def gun_profili(lg, meta, min_yas=7, min_gun=60):
    sc = meta.copy()
    sc["lg"] = lg
    ilk = sc.groupby("tanim")["tarih"].transform("min")
    yas = (sc["tarih"] - ilk).dt.days.to_numpy()
    say = sc.groupby("tanim")["tanim"].transform("size").to_numpy()
    temiz = (yas >= min_yas) & (say >= min_gun)
    if temiz.sum() < 1000:
        return np.zeros(len(sc))
    t = sc.loc[temiz]
    bi, _ = pd.factorize(t["tanim"])
    gi, gun = pd.factorize(t["tarih"])
    b = iki_yonlu(t["lg"].to_numpy(dtype="float64"), bi, gi)
    profil = pd.Series(b, index=pd.Index(gun, name="tarih")).sort_index()
    ek = np.nan_to_num(sc["tarih"].map(profil).to_numpy(dtype="float64"))
    return ek - float(ek.mean())


def zincir(lg, log_guc, meta, beta, c, delta):
    r = lg - log_guc
    ort = float(r.mean())
    lg2 = ort + beta * (r - ort) + log_guc
    if c != 1.0:
        lg2 = lg2 + (c - 1.0) * gun_profili(lg2, meta)
    return lg2 + delta
