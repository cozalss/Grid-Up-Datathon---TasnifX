"""PANEL SINIR GUNU DUZELTMESI -- gonderim dosyasi ureticisi.

BULGU (panel_sinir2.py + panel_sinir_soguk.py, uc blok, uretim kirpmasiyla):
bir trafonun panele GIRDIGI ve panelden CIKTIGI gun, model SISTEMATIK
OLARAK YUKSEK tahmin ediyor. Alti (blok x sinir turu) hucrenin ALTISI da
negatif ve hicbiri sinirda degil:

    SICAK  giris:  yaz25 -0,662 (t=-9,7)  guz25 -0,324 (t=-5,1)  kis26 -0,777 (t=-16,7)
    SICAK  cikis:  yaz25 -0,754 (t=-15,3) guz25 -1,096 (t=-29,8) kis26 -0,880 (t=-22,4)
    SOGUK  giris:  yaz25 -1,080 (t=-12,5) guz25 -0,338 (t=-6,2)  kis26 -0,335 (t=-7,2)
    SOGUK  cikis:  yaz25 -3,239 (t=-17,6) guz25 -1,571 (t=-9,2)  kis26 -0,659 (t=-9,7)

MEKANIZMA (fiziksel, uydurma degil): sinir gunu KISMI gundur. Sayac o gun
devreye giriyor ya da o gun kesiliyor; 24 saatlik tuketim olculmuyor.
Model gunu TAM sayiyor. Bu bir mevsim/kohort etkisi degil, olcum
artefaktidir -- ve tam bu yuzden bloklar arasi TASINIR.

PENCERE KENARI ARTEFAKTI TEMIZLENDI:
  * train'de 2025-01-01 (panel basi) giris SAYILMAZ, 2026-03-31 (train sonu)
    cikis SAYILMAZ. Temizlenmeden once kis26'nin "olum" kumesi train sonuyla
    doluydu ve yanliligi -0,147'ye seyreliyordu (gercegi -0,88).
  * testte 2026-04-01 ancak trafo 2026-03-31'de train'de YOKSA giristir
    (train'e koprulu). 2026-07-31 ASLA cikis sayilmaz -- bilinemez.

TESTTE UYGULANABILIR NUFUS (train'e koprulu):
    giris 3.860 satir (%0,540)   -- 2.370'i 2026-05-11'de
    cikis   986 satir (%0,138)
    kesisim 82

GENLIK (kalici kural 12: fold YON verir, GENLIK vermez):
    blok optimumlari  d_giris -0,20..-0,70   d_cikis -0,80..-1,10
    varsayilan olarak OPTIMUMUN ALTI secildi.

DIKKAT: bu betik KAGGLE'A HICBIR SEY GONDERMEZ.

Kullanim:
  uv run python experiments/kapali_eksenler/uret_panel_sinir.py \
      --giris submissions/tuketim_v93_gram_optimum.csv \
      --cikis submissions/tuketim_v94_panel_sinir.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
GUN = pd.Timedelta(days=1)
TEST_BAS = pd.Timestamp("2026-04-01")
TEST_SON = pd.Timestamp("2026-07-31")
BEKLENEN_SATIR = 714_688
D_GIRIS = -0.35
D_CIKIS = -0.45


def yol(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else KOK / q


def main() -> int:
    a = argparse.ArgumentParser(description="panel sinir gunu duzeltmesi")
    a.add_argument("--giris", required=True)
    a.add_argument("--cikis", required=True)
    a.add_argument("--d-giris", type=float, default=D_GIRIS)
    a.add_argument("--d-cikis", type=float, default=D_CIKIS)
    ar = a.parse_args()
    for nm, v in (("d_giris", ar.d_giris), ("d_cikis", ar.d_cikis)):
        if not -1.5 <= v <= 0.0:
            raise SystemExit(f"{nm} mantik disi: {v}")

    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
        encoding="utf-8",
    )
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
        encoding="utf-8",
    )

    # train'e KOPRULU sinir tespiti
    ort = pd.concat([tr[["tanim", "tarih"]], te[["tanim", "tarih"]]], ignore_index=True)
    ort["_t"] = np.arange(len(ort))
    ort = ort.sort_values(["tanim", "tarih"], kind="mergesort")
    onc = ort.groupby("tanim", observed=True)["tarih"].shift(1)
    son = ort.groupby("tanim", observed=True)["tarih"].shift(-1)
    ort["giris"] = onc.isna() | ((ort["tarih"] - onc) > GUN)
    ort["cikis"] = son.isna() | ((son - ort["tarih"]) > GUN)
    ort = ort.sort_values("_t")
    t = ort.iloc[len(tr) :].reset_index(drop=True)
    if len(t) != len(te) or not (t["tarih"].values == te["tarih"].values).all():
        raise SystemExit("test dilimi hizalanmadi")
    g = t["giris"].to_numpy()
    c = (t["cikis"] & (t["tarih"] != TEST_SON)).to_numpy()

    d = pd.read_csv(yol(ar.giris))
    if len(d) != BEKLENEN_SATIR:
        raise SystemExit(f"giris {len(d)} satir, beklenen {BEKLENEN_SATIR}")
    if not (d["id"].values == te["id"].values).all():
        raise SystemExit("id sirasi test.csv ile ayni degil")

    lg = np.log1p(np.clip(d["tuketim"].to_numpy(dtype="float64"), 0, None))
    yeni = lg + ar.d_giris * g + ar.d_cikis * c
    cikti = np.clip(np.expm1(yeni), 0.0, None)
    fark = yeni - lg

    print(f"giris  {yol(ar.giris).name}")
    print(
        f"  GIRIS satir {int(g.sum()):,} ({g.mean():.6f})   "
        f"CIKIS satir {int(c.sum()):,} ({c.mean():.6f})   "
        f"kesisim {int((g & c).sum()):,}"
    )
    print(f"  d_giris {ar.d_giris:+.3f}   d_cikis {ar.d_cikis:+.3f}")
    print(
        f"  degisen satir {int((np.abs(fark) > 1e-12).sum()):,}   "
        f"dokunulmayan {int((np.abs(fark) <= 1e-12).sum()):,}"
    )
    print(f"  ortalama kayma (yalniz sinir) {fark[g | c].mean():+.6f}")
    print(
        f"  NaN {int(np.isnan(cikti).sum())}  negatif {int((cikti < 0).sum())}  "
        f"sifir {int((cikti == 0).sum())}   "
        f"genel ort log1p {lg.mean():.6f} -> {yeni.mean():.6f}"
    )

    # KAPILAR
    dis = ~(g | c)
    if float(np.abs(fark[dis]).max()) > 1e-12:
        raise SystemExit("sinir disi satirlar degismis")
    bek_g = ar.d_giris * g.astype("float64") + ar.d_cikis * c.astype("float64")
    if float(np.abs(fark - bek_g).max()) > 1e-12:
        raise SystemExit("uygulanan kayma beklenenden farkli (kirpma yemis olabilir)")
    if int(np.isnan(cikti).sum()) or int((cikti < 0).sum()):
        raise SystemExit("NaN veya negatif uretildi")
    if int((t["tarih"] == TEST_SON).sum()) and bool(c[(t["tarih"] == TEST_SON).to_numpy()].any()):
        raise SystemExit("2026-07-31 cikis olarak isaretlenmis -- pencere kenari sizdi")

    o = yol(ar.cikis)
    o.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": d["id"], "tuketim": cikti}).to_csv(o, index=False)
    print(f"\nyazildi: {o}")
    print("\nBEKLENTI (uc blok, seviye-notr, kirpmali olcut):")
    print("  bkz. experiments/kapali_eksenler/panel_sinir_ihtiyat.json")
    print("\n*** KAGGLE'A GONDERIM BU BETIGIN ISI DEGIL. ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
