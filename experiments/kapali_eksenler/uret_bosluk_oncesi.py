"""BOSLUK ONCESI CIKIS GUNU -- son_islem_olay.py'nin KACIRDIGI tek parca.

KANIT (experiments/kapali_eksenler/panel_sinir*.py):
  * Panel sinir gunu (giris VE cikis) yanliligi uc blokta da negatif, 6/6
    hucre, t = -5,1 .. -29,8. MODELSIZ eslenik kontrol: sinir gunu komsu
    gununden GIRIS -0,510 / CIKIS -0,579 dusuk; ic gun farki +0,002.
    Mekanizma fiziksel: sinir gunu KISMI gun.
  * AMA bu mekanizma URETIMDE ZATEN VAR: scripts/son_islem_olay.py, v67
    zincirinde. Benim sinir maskem (4.764 satir) onun maskesinin (4.070)
    KATI USTKUMESI; kesisim 4.070, olayda olup bende olmayan 0.

GERIYE KALAN: 694 satir. Ayrisimi:
    677  ic sinir  -> BOSLUK ONCESI CIKIS GUNU
     11  son gun, OLU trafo (olay'in ``canli`` filtresi disliyor)
      6  ilk gun

son_islem_olay ``ic_bosluk``i (bosluktan DONUS gunu) duzeltiyor ama
BOSLUGA GIRIS gununu (bosluktan onceki son gun) duzeltmiyor. Ucu blokta
da olculdu (uretim tabani, kirpmali):

    bosluk-giris yanliligi:  yaz25 -0,774 (n=349)  guz25 -0,842 (n=414)
                             kis26 -0,878 (n=646)     3/3 AYNI ISARET

TAVAN: p = 694/714.688 = 0,000971;  b ~ -0,83
    optimum dMSE = -p*b^2 = -0,00067   (docs/43 §6 madde 2'nin sayisi)
    d = -0,50 ile  dMSE = p*(d^2 - 2*d*b) = -0,00056

Yani GERCEK ama KUCUK. Kapinin (-0,002) ALTINDA.

DIKKAT: bu betik KAGGLE'A HICBIR SEY GONDERMEZ.

Kullanim:
  uv run python experiments/kapali_eksenler/uret_bosluk_oncesi.py \
      --giris submissions/tuketim_v93_gram_optimum.csv \
      --cikis submissions/tuketim_v94_bosluk_oncesi.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
GUN = pd.Timedelta(days=1)
TEST_SON = pd.Timestamp("2026-07-31")
BEKLENEN_SATIR = 714_688
DELTA = -0.50


def yol(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else KOK / q


def main() -> int:
    a = argparse.ArgumentParser(description="bosluk oncesi cikis gunu duzeltmesi")
    a.add_argument("--giris", required=True)
    a.add_argument("--cikis", required=True)
    a.add_argument("--delta", type=float, default=DELTA)
    ar = a.parse_args()
    if not -1.2 <= ar.delta <= 0.0:
        raise SystemExit(f"delta mantik disi: {ar.delta}")

    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
        encoding="utf-8",
    )
    d = pd.read_csv(yol(ar.giris))
    if len(d) != BEKLENEN_SATIR:
        raise SystemExit(f"giris {len(d)} satir")
    if not (d["id"].values == te["id"].values).all():
        raise SystemExit("id sirasi test.csv ile ayni degil")

    m = te.copy()
    m["_i"] = np.arange(len(m))
    m = m.sort_values(["tanim", "tarih"], kind="mergesort")
    g = m.groupby("tanim", observed=True)
    son_t = g["tarih"].transform("max")
    sonraki = g["tarih"].shift(-1)
    # BOSLUK ONCESI: sonraki kayit var ama ertesi gun DEGIL
    hedef_s = sonraki.notna() & ((sonraki - m["tarih"]) > GUN)
    m["hedef"] = hedef_s
    m = m.sort_values("_i")
    hedef = m["hedef"].to_numpy()

    lg = np.log1p(np.clip(d["tuketim"].to_numpy(dtype="float64"), 0, None))
    yeni = lg + ar.delta * hedef
    cikti = np.clip(np.expm1(yeni), 0.0, None)
    fark = yeni - lg

    p = float(hedef.mean())
    print(f"giris  {yol(ar.giris).name}")
    print(f"  bosluk oncesi cikis gunu: {int(hedef.sum()):,} satir ({p:.6f})")
    print(f"  farkli trafo            : {te.loc[hedef, 'tanim'].nunique():,}")
    print(f"  delta {ar.delta:+.3f}")
    print(
        f"  degisen {int((np.abs(fark) > 1e-12).sum()):,}   "
        f"dokunulmayan {int((np.abs(fark) <= 1e-12).sum()):,}"
    )
    print(
        f"  NaN {int(np.isnan(cikti).sum())}  negatif {int((cikti < 0).sum())}  "
        f"sifir {int((cikti == 0).sum())}"
    )
    if float(np.abs(fark[~hedef]).max()) > 1e-12:
        raise SystemExit("hedef disi satirlar degismis")
    if float(np.abs(fark[hedef] - ar.delta).max()) > 1e-12:
        raise SystemExit("uygulanan kayma istenen degil (kirpma yemis olabilir)")
    if int(np.isnan(cikti).sum()) or int((cikti < 0).sum()):
        raise SystemExit("NaN veya negatif")

    o = yol(ar.cikis)
    o.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": d["id"], "tuketim": cikti}).to_csv(o, index=False)
    print(f"\nyazildi: {o}")
    print("\nBEKLENTI (b = gercek yanlilik):")
    for b in (-0.60, -0.83, -1.00):
        print(
            f"  b={b:+.2f} -> dMSE {p * (ar.delta**2 - 2 * ar.delta * b):+.6f}"
            f"   (o b'de optimum {-p * b * b:+.6f})"
        )
    print("\n*** KAGGLE'A GONDERIM BU BETIGIN ISI DEGIL. ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
