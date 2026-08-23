"""SON ISLEM: soguk tahminlere ofset uzayinda BUZME. Yeniden egitim YOK.

NEDEN
-----
kis26 (ezber orani %0 olan TEK durust kat) soguk uzmani ONEMSIZ bir tabandan
geride:  uretim 1,86931  vs  duz kVA kovasi ortalamasi 1,8162.

Bir tahminci onemsiz bir tabandan kotuyse en olasi sebep ASIRI YAYILMA.
RMSLE log uzayinda kareli hata oldugu icin tahmini ortalamaya dogru buzmek
(James-Stein) beklenen hatayi dusurur: yanlilik ekler, varyanstan daha cok
kazandirir.

OLCULDU (2026-08-23, scripts/deney_soguk_buzme.py, kis26 soguk, 3 tohum):

    beta    RMSLE      tabana gore
    1,00   1,86931     +0,00000
    0,80   1,85156     -0,01775
    0,66   1,84268     -0,02662
    0,60   1,83979     -0,02952
    0,50   1,83619     -0,03312
    0,40   1,83412     -0,03519   (egrinin dibi)

Egrinin dibine GIDILMIYOR: beta'yi kis26'da secmek o bloga asiri uydurmak
olur. beta=0,60 secildi -- bagimsiz bir ajan analizi de 0,50-0,66 bandini
onerdi, ve egri optimum civarinda duz oldugu icin kazancin ~%84'unu alip
kirilma noktasini uzaga tasiyor.

DONUSUM (yalnizca soguk satirlarda):
    r  = log1p(tahmin) - log1p(guc)        # kapasite ofsetli uzay
    r' = ort(r) + beta * (r - ort(r))
    yeni = expm1(r' + log1p(guc))

SOGUK TANIMI: test trafosunun tanim kodu train.csv'de HIC gecmiyor.
Test satirlarinin %22,16'si (158.369 satir, 2.024 trafo).

    python scripts/son_islem.py --giris submissions/X.csv --cikis submissions/Y.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
BETA = 0.60


def main() -> int:
    a = argparse.ArgumentParser(description="soguk ofset buzmesi")
    a.add_argument("--giris", required=True)
    a.add_argument("--cikis", required=True)
    a.add_argument("--beta", type=float, default=BETA)
    ar = a.parse_args()

    ornek = pd.read_csv(KOK / "data/raw/sample_submission.csv", encoding="utf-8")
    te = pd.read_csv(KOK / "data/raw/test.csv", usecols=["id", "tanim", "guc"],
                     encoding="utf-8", dtype={"tanim": str})
    tr = pd.read_csv(KOK / "data/raw/train.csv", usecols=["tanim"],
                     encoding="utf-8", dtype={"tanim": str})
    sub = pd.read_csv(ar.giris, encoding="utf-8")
    if not sub["id"].equals(ornek["id"]):
        raise RuntimeError("id sirasi sample_submission ile ayni degil")

    m = sub.merge(te, on="id", how="left", validate="one_to_one")
    if len(m) != len(sub):
        raise RuntimeError("birlestirme satir sayisini bozdu")
    soguk = ~m["tanim"].isin(set(tr["tanim"])).to_numpy()
    print(f"  soguk satir {int(soguk.sum()):,} / {len(m):,}  (%{100 * soguk.mean():.2f})")
    print(f"  farkli soguk trafo {m.loc[soguk, 'tanim'].nunique():,}")

    log_guc = np.log1p(m["guc"].to_numpy(dtype="float64"))
    r = np.log1p(m["tuketim"].to_numpy(dtype="float64")) - log_guc
    ort = float(r[soguk].mean())
    std_once = float(r[soguk].std())
    r_yeni = r.copy()
    r_yeni[soguk] = ort + ar.beta * (r[soguk] - ort)
    yeni = np.expm1(r_yeni + log_guc)
    yeni = np.clip(yeni, 0.0, None)

    print(f"  ofset ortalamasi {ort:+.5f} | std {std_once:.5f} -> {float(r_yeni[soguk].std()):.5f}")
    print(f"  beta {ar.beta:.2f}")

    cikti = pd.DataFrame({"id": sub["id"], "tuketim": yeni})
    if cikti["tuketim"].isna().any() or (cikti["tuketim"] < 0).any():
        raise RuntimeError("cikti NaN ya da negatif iceriyor")
    if not np.allclose(yeni[~soguk], m["tuketim"].to_numpy()[~soguk], rtol=1e-12, atol=1e-12):
        raise RuntimeError("SICAK satirlar degismis olmamaliydi")
    cikti.to_csv(ar.cikis, index=False)

    f = np.log1p(yeni) - np.log1p(m["tuketim"].to_numpy())
    print(f"  degisen satir {int((np.abs(f) > 1e-9).sum()):,}")
    print(f"  soguk kayma: min {f[soguk].min():+.4f}  medyan {np.median(f[soguk]):+.4f}"
          f"  max {f[soguk].max():+.4f}")
    print(f"  yazildi: {ar.cikis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
