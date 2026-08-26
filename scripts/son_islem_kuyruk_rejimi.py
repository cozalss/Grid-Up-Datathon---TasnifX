"""KUYRUK REJIMI DUZELTMESI -- ozet penceresinin ucunda dogan "sicak" trafolar.

BULGU (docs/45 tik 6, scripts/h18*.py)
--------------------------------------
Model, ILK KAYDI kesmeye <=6 gun kala olusan trafolari SICAK sayar (tanim
train'de vardir) ama gecmisleri pratikte YOKTUR. Bu, sicak/soguk ikili
ayriminin yakalamadigi UCUNCU REJIMDIR ve model orada sistematik olarak
DUSUK tahmin eder.

IKI ORTUSMEYEN KESMEDE OLCULDU (kural 9):

    blok    kesme        KUYRUK<=6g              SUREN>180g   FAZLA      t
    guz25   2025-07-31   +0,1270 (182 trafo)     -0,3484     +0,4754  +27,7
    kis26   2025-11-30   +0,5623 (202 trafo)     +0,2091     +0,3531   +7,2

Isaret IKI BLOKTA DA ayni, 3/3 tohum. NOT: yaz25 bu ekseni GOREMEZ --
kesmesi 2025-03-31 ve train 2025-01-01'de basladigi icin orada yalnizca
4 kuyruk trafosu var. Bu MEVSIMSEL bir eksen degil (gecmis UZUNLUGU ekseni),
o yuzden kural 7 devrede degil; kural 9 (iki ortusmeyen kesme) saglandi.

NEDEN TASINIR: bu bir MUTLAK seviye degil, ayni blok icinde GRUPLAR ARASI
FARK. guz25'te butun gruplar NEGATIF yanlilikta (fold gelecegi goruyor,
asiri tahmin), kis26'da POZITIF -- ama FAZLA iki blokta da POZITIF. Fold'un
kuresel bilgisi farkta sadelesiyor.

MEKANIZMA: parti degil GECMIS UZUNLUGU. Her iki blokta hem toplu hem tekil
kuyruk dogumlulari etkileniyor (guz25: toplu +0,53 / tekil +0,33;
kis26: toplu +0,27 / tekil +0,60). Doz-tepki de var (kis26):
    <=6g +0,562 | 7-30g +0,539 | 31-90g +0,385 | 91-180g +0,161 | >180g +0,209

KIRPMA (kural 1) -- K=50'de IKI BLOKTA DA AYAKTA:
    guz25  K=0 +0,475  K=10 +0,393  K=25 +0,316  K=50 +0,196 (t=+7,8)
    kis26  K=0 +0,353  K=10 +0,306  K=25 +0,252  K=50 +0,175 (t=+4,3)

TEST TARAFI
-----------
2026-03-26..31 arasinda ilk kaydi olusan 356 trafonun 353'u testte:
**29.873 satir = testin %4,18'i**. Train kayitlari medyan 2 (min 1, max 6) --
ikizdeki gruplardan bile KISA, yani etki orada daha guclu olabilir.

DELTA SECIMI
------------
Blok kestirimleri 0,3531 (kis26) ve 0,4754 (guz25); K=50 kirpilmis hali
~0,18-0,20. Kuadratik kayipta optimum delta = E[b]. Secilen **0,30**:
iki blok kestiriminin altinda, kirpilmis degerlerin ustunde.

    dMSE = 2*p*delta*b - p*delta^2      p = 0,0418
    b=0,414 -> -0,00662    b=0,20 -> -0,00126    b=0,10 -> +0,00125

UYGULAMA: yalnizca o 353 trafonun satirlarina DUZ kayma. Gun ekseni,
trafo ekseni, soguk taraf DEGISMEZ.

    python scripts/son_islem_kuyruk_rejimi.py --giris X.csv --cikis Y.csv [--delta 0.30]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
TRAIN_SON = pd.Timestamp("2026-03-31")
KUYRUK_GUN = 6
DELTA_VARSAYILAN = 0.30
BEKLENEN_SATIR = 714688


def main() -> int:
    a = argparse.ArgumentParser(description="kuyruk rejimi duz kaymasi")
    a.add_argument("--giris", required=True)
    a.add_argument("--cikis", required=True)
    a.add_argument("--delta", type=float, default=DELTA_VARSAYILAN)
    ar = a.parse_args()
    if not -0.10 <= ar.delta <= 0.60:
        raise SystemExit(f"delta mantik disi: {ar.delta}")

    giris = KOK / ar.giris if not Path(ar.giris).is_absolute() else Path(ar.giris)
    cikis = KOK / ar.cikis if not Path(ar.cikis).is_absolute() else Path(ar.cikis)

    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    d = pd.read_csv(giris)
    if len(d) != BEKLENEN_SATIR:
        raise SystemExit(f"giris {len(d)} satir")
    if not (d["id"].values == te["id"].values).all():
        raise SystemExit("id sirasi test.csv ile ayni degil")

    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    ilk = tr.groupby("tanim")["tarih"].min()
    n_kayit = tr.groupby("tanim").size()
    kuyruk = set(ilk[ilk >= TRAIN_SON - pd.Timedelta(days=KUYRUK_GUN - 1)].index)
    hedef = te["tanim"].isin(kuyruk).to_numpy()

    tanimlar = sorted({t for t in te.loc[hedef, "tanim"].unique()})
    print(f"giris {giris.name}")
    print(
        f"  kuyruk trafosu (train'de ilk kayit >= "
        f"{(TRAIN_SON - pd.Timedelta(days=KUYRUK_GUN - 1)).date()}): {len(kuyruk):,}"
    )
    print(
        f"  bunlardan TESTTE olan: {len(tanimlar):,} trafo, "
        f"{int(hedef.sum()):,} satir ({hedef.mean():.4f})"
    )
    kk = n_kayit.reindex(tanimlar)
    print(f"  train kayit sayisi: min {kk.min()} medyan {int(kk.median())} max {kk.max()}")

    # sicak mi? (train'de olduklari icin evet -- dogrula)
    sicak_set = set(tr["tanim"].unique())
    assert all(t in sicak_set for t in tanimlar), "kuyruk trafolari train'de degil?!"
    print("  hepsi SICAK rejimde (train'de tanim var) -> ucuncu rejim, ikili ayrim gormuyor")

    lg = np.log1p(np.clip(d["tuketim"].to_numpy(dtype="float64"), 0, None))
    yeni = lg.copy()
    yeni[hedef] = lg[hedef] + ar.delta
    cikti = np.clip(np.expm1(yeni), 0.0, None)

    fark = yeni - lg
    uyg = float(fark[hedef].mean())
    print(f"\ndelta = {ar.delta}")
    print(f"  degisen satir      {int((np.abs(fark) > 1e-12).sum()):,}")
    print(f"  uygulanan kayma    {uyg:+.6f}  (istenen {ar.delta})")
    print(
        f"  dokunulmayan       {int((np.abs(fark) <= 1e-12).sum()):,} satir, "
        f"maxabs {float(np.abs(fark[~hedef]).max()):.2e}"
    )
    print(f"  genel ort log1p    {lg.mean():.6f} -> {yeni.mean():.6f}")
    print(
        f"  NaN {int(np.isnan(cikti).sum())}  negatif {int((cikti < 0).sum())}  "
        f"sifir {int((cikti == 0).sum())}"
    )

    if abs(uyg - ar.delta) > 1e-12:
        raise SystemExit(f"uygulanan kayma {uyg} != istenen {ar.delta}")
    if float(np.abs(fark[~hedef]).max()) > 1e-12:
        raise SystemExit("hedef disi satirlar degismis")

    cikis.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": d["id"], "tuketim": cikti}).to_csv(cikis, index=False)
    print(f"\nyazildi: {cikis}")
    p = float(hedef.mean())
    for b in (0.414, 0.353, 0.20, 0.10):
        print(
            f"  gercek b={b:.3f} -> dMSE {2 * p * ar.delta * b - p * ar.delta**2:+.6f}".replace(
                "+", "-", 1
            )
            if 2 * p * ar.delta * b - p * ar.delta**2 > 0
            else f"  gercek b={b:.3f} -> dMSE {-(2 * p * ar.delta * b - p * ar.delta**2):+.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
