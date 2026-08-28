"""v93 denetimi -- adim 10: v93 dosyasinin bicim/gecerlilik kapisi."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
SS = pd.read_csv(KOK / "data/raw/sample_submission.csv")
gecti = True


def kontrol(ad: str, kosul: bool, ek: str = "") -> None:
    global gecti
    gecti &= bool(kosul)
    print(f"  [{'GECTI' if kosul else 'KALDI'}] {ad}{(' -- ' + ek) if ek else ''}")


for dosya in ("tuketim_v93_gram_optimum.csv", "tuketim_v83_sicak_optimum.csv"):
    print(f"\n=== {dosya} ===")
    df = pd.read_csv(KOK / "submissions" / dosya)
    kontrol(
        "kolonlar sample_submission ile ayni",
        list(df.columns) == list(SS.columns),
        f"{list(df.columns)} vs {list(SS.columns)}",
    )
    kontrol("satir sayisi 714688", len(df) == 714688, str(len(df)))
    kontrol("id kumesi birebir", set(df["id"]) == set(SS["id"]))
    kontrol("id sirasi birebir", df["id"].equals(SS["id"]))
    kontrol("id tekil", df["id"].is_unique)
    v = df[df.columns[1]].to_numpy(dtype=float)
    kontrol("NaN yok", not np.isnan(v).any(), f"{int(np.isnan(v).sum())} NaN")
    kontrol("sonsuz yok", np.isfinite(v).all())
    kontrol("negatif yok", (v >= 0).all(), f"min={v.min():.6g}")
    print(
        f"  bilgi: min={v.min():.6g} medyan={np.median(v):.4f} "
        f"maks={v.max():.2f} sifir={int((v == 0).sum())} ort={v.mean():.4f}"
    )

print(f"\nSONUC: {'TUM KAPILAR GECTI' if gecti else 'KAPIDA KALDI'}")
