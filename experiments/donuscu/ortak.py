"""DONUSCU ekseni -- ortak yukleyiciler ve trafo bazli oznitelik tablosu.

Fikir: uzun raporlama boslugundan panele DONEN trafo, donduğunde TUKETEREK
donuyor; uretim modeli ise onu son (sifir/dusuk) gecmisine bakarak DUSUK
yaziyor. Grup B (93 trafo) bu olgunun en dar hali. Burada nufusu buyutup
`Q * delta*^2` kazanc egrisinin tepesini ariyoruz.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

KOK = Path(__file__).resolve().parents[2]
SUB = KOK / "submissions"
CIK = KOK / "experiments" / "donuscu"
N_TEST = 714688
UFUK = 122


@lru_cache(maxsize=1)
def train() -> pd.DataFrame:
    tr = pd.read_csv(KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str})
    tr["tarih"] = pd.to_datetime(tr["tarih"])
    tr["lp"] = np.log1p(tr["tuketim"].clip(lower=0))
    tr["lg"] = np.log1p(tr["guc"].astype(float))
    tr["ofs"] = tr["lp"] - tr["lg"]
    return tr


@lru_cache(maxsize=1)
def test() -> pd.DataFrame:
    te = pd.read_csv(KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str})
    te["tarih"] = pd.to_datetime(te["tarih"])
    te["lg"] = np.log1p(te["guc"].astype(float))
    return te


@lru_cache(maxsize=8)
def hizala(ad: str) -> np.ndarray:
    """Gonderim dosyasini test.csv satir sirasina hizala."""
    te = test()
    s = pd.read_csv(SUB / ad, encoding="utf-8")
    m = te[["id"]].merge(s, on="id", how="left", validate="one_to_one")
    if m["tuketim"].isna().any():
        raise RuntimeError(f"{ad}: eksik id")
    return m["tuketim"].to_numpy(dtype=float)


def lp(v: np.ndarray) -> np.ndarray:
    """Olcutun kirpmasini ICEREN log1p."""
    return np.log1p(np.clip(v, 0.0, None))


def trafo_ozet(d: pd.DataFrame, sinir: pd.Timestamp | None = None) -> pd.DataFrame:
    """Trafo bazli oznitelikler; `sinir` verilirse yalniz tarih < sinir kullanilir.

    Kolonlar:
      son_tarih, ilk_tarih, n_kayit, n_poz, maks_tuketim,
      son60_maks (son 60 kaydin maksimumu), en_buyuk_bosluk (gun),
      capa_ofs (son 60 POZITIF kaydin ortalama ofseti; yoksa NaN)
    """
    if sinir is not None:
        d = d[d["tarih"] < sinir]
    d = d.sort_values(["tanim", "tarih"])
    g = d.groupby("tanim", sort=True)
    t = pd.DataFrame(
        {
            "son_tarih": g["tarih"].max(),
            "ilk_tarih": g["tarih"].min(),
            "n_kayit": g.size(),
            "n_poz": g["tuketim"].apply(lambda s: int((s > 0).sum())),
            "maks_tuketim": g["tuketim"].max(),
        }
    )
    son60 = d.groupby("tanim").tail(60)
    t["son60_maks"] = son60.groupby("tanim")["tuketim"].max()
    t["son60_n"] = son60.groupby("tanim").size()
    # ic boslugun en buyugu (ardisik kayit tarihleri arasi fark)
    df = d[["tanim", "tarih"]].copy()
    df["dfark"] = df.groupby("tanim")["tarih"].diff().dt.days
    t["en_buyuk_ic_bosluk"] = df.groupby("tanim")["dfark"].max().fillna(0.0)
    # capa: son 60 pozitif kaydin ortalama ofseti
    poz = d[d["tuketim"] > 0]
    sp = poz.groupby("tanim").tail(60)
    ga = sp.groupby("tanim")["ofs"]
    capa, capa_n = ga.mean(), ga.size()
    t["capa_ofs"] = capa.where(capa_n >= 10)
    t["capa_n"] = capa_n.reindex(t.index).fillna(0).astype(int)
    return t
