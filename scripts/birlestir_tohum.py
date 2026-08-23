"""Ayri kosulmus tohum partilerini LOG UZAYINDA birlestirir.

NEDEN
-----
Son egitim tohumlari log uzayinda ortalanir: ``expm1(mean(log1p(...)))``.
Bu ortalama ILISKILI: k tohumun ortalamasi, ayni k tohumun iki partide
kosulup log uzayinda agirlikli birlestirilmesine BIREBIR esittir.

    mean_6 = (3 * mean_3a + 3 * mean_3b) / 6

Yani 3 tohumluk bir gonderim varsa, 3 tohum daha kosup birlestirmek
6 tohumluk tek kosuyla ayni sonucu verir -- ilk kosuyu tekrarlamadan.

Neden 6 tohum: tohum ortalamasi YANLILIGI degistirmez, yalnizca tahminci
gurultusunun varyansini ~1/k ile duser (Krogh & Vedelsby, NeurIPS 1994 --
ayrisma log uzayinda aritmetik ortalama icin bir OZDESLIKtir ve RMSLE log
uzayinda kareli hata oldugu icin burada birebir gecerlidir). Risksiz: yeni
bir varsayim getirmez, yalnizca ayni tahmincinin daha kararli bir
kestirimini verir.

    python scripts/birlestir_tohum.py --cikis submissions/tuketim_v35_ham.csv \
        submissions/tuketim_v32_ham.csv:3 submissions/tuketim_v34_ek3tohum.csv:3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]


def main() -> int:
    a = argparse.ArgumentParser(description="tohum partilerini log uzayinda birlestir")
    a.add_argument("parti", nargs="+", help="dosya.csv:tohum_sayisi")
    a.add_argument("--cikis", required=True)
    ar = a.parse_args()

    ornek = pd.read_csv(KOK / "data/raw/sample_submission.csv", encoding="utf-8")
    birikim = np.zeros(len(ornek), dtype="float64")
    toplam = 0.0
    for p in ar.parti:
        yol, _, adet = p.rpartition(":")
        if not yol or not adet.isdigit():
            raise SystemExit(f"bicim 'dosya.csv:tohum_sayisi' olmali: {p}")
        n = int(adet)
        d = pd.read_csv(KOK / yol, encoding="utf-8")
        m = ornek[["id"]].merge(d, on="id", how="left")
        if m["tuketim"].isna().any():
            raise RuntimeError(f"{yol}: ornek gonderim id kumesiyle ortusmuyor")
        v = m["tuketim"].to_numpy(dtype="float64")
        if not np.isfinite(v).all() or (v < 0).any():
            raise RuntimeError(f"{yol}: NaN/sonsuz/negatif iceriyor")
        birikim += n * np.log1p(v)
        toplam += n
        print(f"  {yol}  {n} tohum  log1p ort {np.log1p(v).mean():.5f}")

    tahmin = np.clip(np.expm1(birikim / toplam), 0.0, None)
    print(f"  TOPLAM {toplam:.0f} tohum  ->  log1p ort {np.log1p(tahmin).mean():.5f}")
    print(f"  min {tahmin.min():.4g}  medyan {np.median(tahmin):.1f}  maks {tahmin.max():.1f}")

    yol = KOK / ar.cikis
    yol.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ornek["id"], "tuketim": tahmin}).to_csv(yol, index=False, encoding="utf-8")
    print(f"  yazildi: {ar.cikis}  ({len(tahmin):,} satir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
