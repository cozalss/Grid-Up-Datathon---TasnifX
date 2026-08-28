"""SOGUK KOHORT KALDIRACI -- ortak veri kurulumu ve olcut.

Uc CV blogunun (yaz25 / guz25 / kis26) onbelleklenmis SOGUK UZMAN
tahminlerini yukler, uretimdeki son islemi (cat-harmani + James-Stein
buzmesi beta=0,60 + seviye ofseti 0,1046) birebir yeniden kurar ve aday
son-islemleri bu TABAN uzerinde olcer.

Uretim zinciri (docs/47, scripts/son_islem.py, sota_tuketim_pipeline.py:806):

    r  = log1p(tahmin) - log1p(guc)
    r' = ort(r) + 0.60 * (r - ort(r)) + 0.1046
    yeni = expm1(r' + log1p(guc))

Olcut: SOGUK satirlarda log1p uzayinda MSE. Genel (test) MSE'ye cevrim
carpani SOGUK_PAY = 0,2216 (158.369 / 714.688).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
BLOKLAR = ("yaz25", "guz25", "kis26")
ORTAK_TOHUM = (1000, 1001, 1002)  # uc blokta da mevcut olan tohumlar
BETA_URETIM = 0.60
DELTA_URETIM = 0.1046
SOGUK_PAY = 158_369 / 714_688

BLOK_BASI = {"yaz25": "2025-04-01", "guz25": "2025-08-01", "kis26": "2025-12-01"}


@dataclass
class Blok:
    ad: str
    tanim: np.ndarray
    tarih: np.ndarray  # datetime64[ns]
    guc: np.ndarray
    y: np.ndarray
    lgy: np.ndarray  # log1p(y)
    lgc: np.ndarray  # log1p(guc)
    ham: dict[str, np.ndarray]  # aile -> tohum-ortalamasi log tahmin
    lokasyon: np.ndarray
    il: np.ndarray
    ilce: np.ndarray
    giris: np.ndarray  # trafonun blok icindeki ILK gunu
    yas: np.ndarray  # giristen bu yana gun

    @property
    def n(self) -> int:
        return len(self.y)


def _lokasyon_parcala(lok: pd.Series) -> tuple[pd.Series, pd.Series]:
    """``IL>...>ILCE`` -- parca sayisi DEGISKEN (Izmir 3, Manisa 2).

    Il her zaman ilk parca, ilce her zaman SON parca. Ortadaki bolge
    parcasi Izmir'de var, Manisa'da yok; bu yuzden sabit indeks kullanmak
    59 Manisa trafosunu dusururdu.
    """
    p = lok.fillna("").str.split(">")
    il = p.str[0].str.strip()
    ilce = p.str[-1].str.strip()
    return il, ilce


def train_meta() -> pd.DataFrame:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih", "lokasyon"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    tr["tarih"] = pd.to_datetime(tr["tarih"])
    lok = tr.groupby("tanim", observed=True)["lokasyon"].first()
    ilk = tr.groupby("tanim", observed=True)["tarih"].min()
    return pd.DataFrame({"lokasyon": lok, "ilk_gun": ilk})


def blok_yukle(ad: str, tmeta: pd.DataFrame, *, tohum=ORTAK_TOHUM) -> Blok:
    z = np.load(KOK / f"data/interim/deney/soguk_tahmin_{ad}.npz")
    m = pd.read_parquet(KOK / f"data/interim/{ad}_soguk_meta.parquet")
    m["tanim"] = m["tanim"].astype(str)
    ham: dict[str, np.ndarray] = {}
    for aile in ("cat", "xgb", "lgbm"):
        anah = [f"{t}_{aile}" for t in tohum if f"{t}_{aile}" in z.files]
        if anah:
            ham[aile] = np.mean([z[k] for k in anah], axis=0)

    j = m.merge(tmeta, left_on="tanim", right_index=True, how="left")
    il, ilce = _lokasyon_parcala(j["lokasyon"])
    # blok icindeki ilk gun: uretimde test penceresinde de gozlenebilir
    ilk_blok = m.groupby("tanim", observed=True)["tarih"].transform("min")
    yas = (m["tarih"] - ilk_blok).dt.days.to_numpy()

    guc = m["guc"].to_numpy(dtype="float64")
    y = np.clip(m["y"].to_numpy(dtype="float64"), 0.0, None)
    return Blok(
        ad=ad,
        tanim=m["tanim"].to_numpy(),
        tarih=m["tarih"].to_numpy(),
        guc=guc,
        y=y,
        lgy=np.log1p(y),
        lgc=np.log1p(guc),
        ham=ham,
        lokasyon=j["lokasyon"].to_numpy(),
        il=il.to_numpy(),
        ilce=ilce.to_numpy(),
        giris=ilk_blok.to_numpy(),
        yas=yas,
    )


def tum_bloklar(tohum=ORTAK_TOHUM) -> dict[str, Blok]:
    tm = train_meta()
    return {b: blok_yukle(b, tm, tohum=tohum) for b in BLOKLAR}


def taban_r(
    b: Blok, *, aile: str = "cat", beta: float = BETA_URETIM, delta: float = DELTA_URETIM
) -> np.ndarray:
    """URETIM TABANI: ofset uzayinda buzulmus ve seviyelenmis tahmin."""
    r = b.ham[aile] - b.lgc
    return r.mean() + beta * (r - r.mean()) + delta


def mse(b: Blok, r: np.ndarray) -> float:
    """SOGUK satirlarda log1p uzayinda MSE."""
    e = b.lgy - (r + b.lgc)
    return float((e * e).mean())
