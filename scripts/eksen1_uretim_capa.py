"""EKSEN 1 -- CAPA: uretim modelinin kis26 b_i profilini yeniden uret.

Amac: vekil modelin kis26 yanliligi, URETIM modelinin kis26 yanliligina
benziyor mu? Benziyorsa vekil uzerinden verilen "tasiniyor mu" hukmu
uretime tasinabilir. Benzemiyorsa vekil gecerli bir capa degildir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
CIKTI = KOK / "data" / "interim" / "eksen1_kesme"
CIKTI.mkdir(parents=True, exist_ok=True)

BLOK = {
    "yaz25": ("2025-01-01", "2025-03-31", "2025-04-01", "2025-07-31"),
    "guz25": ("2025-01-01", "2025-07-31", "2025-08-01", "2025-11-30"),
    "kis26": ("2025-01-01", "2025-11-30", "2025-12-01", "2026-03-31"),
}


def main():
    d = pd.read_csv(KOK / "data" / "raw" / "train.csv", parse_dates=["tarih"])
    d["ofs"] = np.log1p(d["tuketim"])  # onbellek log1p(tuketim) uzayinda
    z = np.load(KOK / "data" / "interim" / "deney" / "sicak_tahmin.npz")

    for blok, (ob, os_, eb, es) in BLOK.items():
        sicak_tanim = set(d.loc[(d.tarih >= ob) & (d.tarih <= os_), "tanim"].unique())
        m = (d.tarih >= eb) & (d.tarih <= es) & d.tanim.isin(sicak_tanim)
        alt = d.loc[m]
        p = np.mean(
            [
                (3 * z[f"{blok}_{s}_cat"] + z[f"{blok}_{s}_xgb"] + z[f"{blok}_{s}_lgbm"]) / 5
                for s in (1000, 1001, 1002)
            ],
            axis=0,
        )
        if len(alt) != len(p):
            print(f"[{blok}] SATIR UYUSMUYOR: veri {len(alt)} vs onbellek {len(p)} -- atlaniyor")
            continue
        art = alt["ofs"].to_numpy() - p
        g = pd.DataFrame({"tanim": alt["tanim"].to_numpy(), "art": art})
        b = g.groupby("tanim")["art"].mean()
        n = g.groupby("tanim")["art"].size()
        mse = float(np.mean(art**2))
        tavan = mse - float(np.mean((art - b.reindex(alt["tanim"]).to_numpy()) ** 2))
        sab = float(np.sum(n * b) / n.sum())
        sabit = mse - float(np.mean((art - sab) ** 2))
        print(f"[{blok}] n={len(alt):,} trafo={len(b)}  MSE={mse:.5f}")
        print(
            f"    agirlikli ort b={sab:+.4f}  duz ort={b.mean():+.4f} std={b.std():.4f}"
            f"  medyan={b.median():+.4f}  poz%={100 * (b > 0).mean():.1f}"
        )
        print(f"    TAVAN kazanc={tavan:.5f}   SABIT delta kazanc={sabit:.5f}")
        b.to_frame("b").assign(n=n).to_csv(CIKTI / f"uretim_b_{blok}.csv")


if __name__ == "__main__":
    main()
