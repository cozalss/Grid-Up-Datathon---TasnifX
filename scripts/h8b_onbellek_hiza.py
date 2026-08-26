"""H8b -- yaz25 gun ekseni onbelleginin HIZASI ve icerigi.

data/interim/gun_ekseni/yaz25_meta.parquet (20.633 satir, 678 trafo,
2025-04-06..07-31) tam olarak SOGUK IKIZ paneli: o pencerede DOGMUS trafolar.
yaz25_{tohum}_taban.npy ayni uzunlukta.

Bu betik taban dizisinin NE oldugunu kesinlestirir (log1p tahmin mi, ham tahmin
mi, hangi model), cunku H8'in dogrulamasi buna dayanacak.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]


def main() -> int:
    m = pd.read_parquet(KOK / "data/interim/gun_ekseni/yaz25_meta.parquet")
    print("meta:", m.shape, list(m.columns))
    print(m.head(3).to_string(), "\n")

    tohumlar = sorted(Path(KOK / "data/interim/gun_ekseni").glob("yaz25_*_taban.npy"))
    print("tohum dosyalari:", [p.name for p in tohumlar], "\n")

    y = m["y"].to_numpy(dtype="float64")
    lgy = np.log1p(np.clip(y, 0, None))
    print(
        f"y        ort {y.mean():12.3f}  std {y.std():12.3f}  min {y.min():.3f} max {y.max():.3f}"
    )
    print(f"log1p(y) ort {lgy.mean():12.4f}  std {lgy.std():12.4f}\n")

    for p in tohumlar:
        a = np.load(p).astype("float64")
        # iki yorum: (i) a zaten log1p tahmin, (ii) a ham tahmin
        r_log = lgy - a
        r_ham = lgy - np.log1p(np.clip(a, 0, None))
        print(
            f"{p.name}  n={len(a)}  ort {a.mean():.4f} std {a.std():.4f} "
            f"min {a.min():.3f} max {a.max():.3f}"
        )
        print(
            f"   [a = log1p tahmin]  artik ort {r_log.mean():+.4f} std {r_log.std():.4f}"
            f"  RMSE {np.sqrt((r_log**2).mean()):.4f}  kor(a,lgy) {np.corrcoef(a, lgy)[0, 1]:+.4f}"
        )
        print(
            f"   [a = ham tahmin  ]  artik ort {r_ham.mean():+.4f} std {r_ham.std():.4f}"
            f"  RMSE {np.sqrt((r_ham**2).mean()):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
