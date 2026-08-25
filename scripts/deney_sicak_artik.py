# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""SICAK ARTIGIN ANATOMISI -- trafo ve gun etkileri cikarildiktan sonra ne kaliyor?

Onbellekten okur (data/interim/aile_onbellek), FIT YOK.

    python scripts/deney_sicak_artik.py --adim taban
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

ONB = KOK / "data" / "interim" / "aile_onbellek"
TOHUMLAR = (1000, 1001, 1002)
AGIRLIK = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 1.4}


def blok_verisi(egitim: pd.DataFrame, blok: str) -> dict:
    _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, blok)
    sicak = ~soguk
    dg = dogrulama[sicak].reset_index(drop=True)
    y = gercek[sicak]
    pay = sum(AGIRLIK.values())
    loglar = []
    for t in TOHUMLAR:
        s = np.zeros(len(dg), dtype="float64")
        for a, w in AGIRLIK.items():
            s += w * np.load(ONB / f"{blok}_{t}_{a}_uretim.npy").astype("float64")
        loglar.append(s / pay)
    log_t = np.mean(loglar, axis=0)
    lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    return {
        "cerceve": dg,
        "r": log_t - lg,
        "g": np.log1p(np.clip(y, 0, None)) - lg,
        "tohum_loglari": loglar,
        "lg": lg,
        "y": y,
    }


def iki_yonlu_arindir(
    v: np.ndarray, trafo: np.ndarray, gun: np.ndarray, tur: int = 8
) -> np.ndarray:
    """Trafo ve gun sabit etkilerini iteratif olarak cikarir (dengesiz panel)."""
    x = v - v.mean()
    s = pd.Series(x)
    for _ in range(tur):
        s = s - s.groupby(trafo).transform("mean")
        s = s - s.groupby(gun).transform("mean")
    return s.to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adim", default="taban")
    ar = ap.parse_args()

    egitim, _test = d.cerceveleri_kur()
    print(f"  egitim {egitim.shape}")

    for b in tm.BLOKLAR:
        v = blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        trafo = dg["tanim"].to_numpy()
        gun = pd.to_datetime(dg["tarih"]).values.astype("datetime64[D]")
        ikili = pd.MultiIndex.from_arrays([trafo, gun])
        print(
            f"\n=== {b.ad}  n={len(dg):,}  trafo={pd.unique(trafo).size:,}  "
            f"gun={pd.unique(gun).size}  tekil(trafo,gun)={ikili.nunique():,}"
        )
        ra = iki_yonlu_arindir(v["r"], trafo, gun)
        ga = iki_yonlu_arindir(v["g"], trafo, gun)
        c = float(np.corrcoef(ra, ga)[0, 1])
        egim = float(np.polyfit(ra, ga, 1)[0])
        print(
            f"  ARTIK  model std {ra.std():.4f}  gercek std {ga.std():.4f}  "
            f"kor {c:+.3f}  OLS egimi {egim:+.4f}"
        )
        # MSE ayrisimi
        e = v["g"] - v["r"]
        mse = float((e**2).mean())
        ea = ga - ra
        print(
            f"  sicak MSE {mse:.5f}  (RMSLE {np.sqrt(mse):.5f})   "
            f"artik payi {float((ea**2).mean()) / mse * 100:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
