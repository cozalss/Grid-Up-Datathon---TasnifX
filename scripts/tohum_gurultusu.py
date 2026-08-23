"""TOHUM GURULTUSU: uretim tahmincisinin tohum varyansini ETIKETSIZ olcer.

NEDEN
-----
Tohum ortalamasinin kazanci once onbellekten kestirilmisti (cat/xgb/lgbm,
uc tohum, MSE(k)=A+B/k uydurmasi) ve 3->6 icin -0,0009 cikmisti. Ama o
onbellekte SINIR AGI YOK -- ve ag harmanin en degisken uyesi (tekil RMSLE
0,838 / 0,867 / 1,106). Yani kestirim sistematik olarak DUSUK.

Bu betik gercek uretim tahmincisini olcer ve ETIKETE IHTIYAC DUYMAZ.

MANTIK
------
Her tohum tahmini log uzayinda  yhat_i = mu + eps_i,  eps bagimsiz, ortalamasi
sifir, varyansi sigma^2. k tohumun ortalamasi icin:

    MSLE(k) = yanlilik^2 + sigma^2 / k

Elimizde her biri m tohumun ortalamasi olan P parti var. Partiler arasi
orneklem varyansi sigma^2/m'i kestirir:

    sigma^2 = m * Var_partiler(yhat)

Etiket gerekmez cunku olculen sey tahmincinin KENDI dagilimidir. Buradan
her k icin beklenen MSLE degisimi kapali formulle cikar.

    python scripts/tohum_gurultusu.py --parti-tohum 3 \
        submissions/tuketim_v32_ham.csv submissions/tuketim_v34_ek3tohum.csv \
        submissions/tuketim_v38_ek3tohum.csv --lb 1.02639
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]


def main() -> int:
    a = argparse.ArgumentParser(description="tohum varyansi (etiketsiz)")
    a.add_argument("parti", nargs="+", help="her biri ayni sayida tohumun ortalamasi olan csv")
    a.add_argument("--parti-tohum", type=int, default=3, help="parti basina tohum sayisi")
    a.add_argument("--lb", type=float, default=None, help="bilinen bir LB skoru (olcek icin)")
    a.add_argument("--k", type=int, nargs="+", default=[3, 6, 9, 12, 15, 18])
    ar = a.parse_args()

    if len(ar.parti) < 2:
        raise SystemExit("en az iki parti gerekli")

    ornek = pd.read_csv(KOK / "data/raw/sample_submission.csv", encoding="utf-8")
    te = pd.read_csv(KOK / "data/raw/test.csv", usecols=["id", "tanim"],
                     encoding="utf-8", dtype={"tanim": str})
    tr = pd.read_csv(KOK / "data/raw/train.csv", usecols=["tanim"],
                     encoding="utf-8", dtype={"tanim": str})
    eslesen = ornek[["id"]].merge(te, on="id", how="left")["tanim"]
    soguk = ~eslesen.isin(set(tr["tanim"])).to_numpy()

    satirlar = []
    for yol in ar.parti:
        d = pd.read_csv(KOK / yol, encoding="utf-8")
        m = ornek[["id"]].merge(d, on="id", how="left")
        if m["tuketim"].isna().any():
            raise RuntimeError(f"{yol}: id kumesi ortusmuyor")
        satirlar.append(np.log1p(m["tuketim"].to_numpy(dtype="float64")))
        print(f"  {yol}")
    A = np.vstack(satirlar)
    print(f"  {A.shape[0]} parti x {ar.parti_tohum} tohum, {A.shape[1]:,} satir")

    print(f"\n  {'kesim':7}{'partiler arasi std':>20}{'tek tohum sigma':>18}")
    sigma2 = {}
    kesimler = (("TUM", np.ones(A.shape[1], dtype=bool)),
                ("SICAK", ~soguk), ("SOGUK", soguk))
    for ad, maske in kesimler:
        v = float(A[:, maske].var(axis=0, ddof=1).mean())
        sigma2[ad] = ar.parti_tohum * v
        print(f"  {ad:7}{np.sqrt(v):20.5f}{np.sqrt(sigma2[ad]):18.5f}")

    print("\n  MSE katkisi (sigma^2 / k)")
    print("  " + f"{'kesim':7}" + "".join(f"{f'k={k}':>11}" for k in ar.k))
    for ad in ("TUM", "SICAK", "SOGUK"):
        print(f"  {ad:7}" + "".join(f"{sigma2[ad] / k:11.6f}" for k in ar.k))

    if ar.lb is not None:
        msle = ar.lb**2
        taban = msle - sigma2["TUM"] / ar.parti_tohum
        print(f"\n  Bilinen skor {ar.lb:.5f} (k={ar.parti_tohum}) -> MSLE {msle:.5f}")
        print(f"  tohumdan BAGIMSIZ taban {taban:.5f}")
        print(f"  {'k':>4}{'MSLE':>11}{'RMSLE':>10}{'k=' + str(ar.parti_tohum) + 'e gore':>14}")
        for k in ar.k:
            r = float(np.sqrt(taban + sigma2["TUM"] / k))
            print(f"  {k:4d}{taban + sigma2['TUM'] / k:11.6f}{r:10.5f}{r - ar.lb:+14.5f}")
        print("\n  NOT: taban, olcegi vermek icin BILINEN bir skordan turetildi.")
        print("  Yapilandirma degistiyse taban da degisir; tasinan sey sigma^2/k FARKIDIR.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
