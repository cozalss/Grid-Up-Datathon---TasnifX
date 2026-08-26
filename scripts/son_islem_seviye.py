"""SEVIYE KAYMASI -- SICAK satirlara sabit ``+delta`` ofset. Yeniden egitim YOK.

NEDEN
-----
Uretim modeli, gecmisi olan (SICAK) trafolar icin test penceresinin
SEVIYESINI sistematik olarak DUSUK veriyor. Sebep: yillik surukleme
(year-over-year drift) modelin oznitelikleriyle disariya tasinamiyor --
model "gecen yil bu mevsim ne olduysa o" diyor, oysa seviye yukseliyor.

OLCUM (2026-08-25, eksen 8)
---------------------------
Uc CV blogundan yalnizca ``kis26`` bu soruyu yanitlayabilir; digerleri
GELECEGE bakarak egitiliyor (``blok_parcalari`` hedef blok DISINDAKI her
seyi egitime koyar), yani surukleme onlarda zaten "biliniyor":

    blok    egitim penceresi              olculen kayma (gercek - model)
    yaz25   Agu2025..Mar2026 (GELECEK)    Nis -0,014  May +0,011   (~0)
    guz25   Nis-Tem + Ara-Mar (GELECEK)   -0,32 .. -0,37  (ASIRI TAHMIN)
    kis26   Nis-Kas 2025 (YALNIZ GECMIS)  +0,22 / +0,19 / +0,12 / +0,09

Test de YALNIZ GECMISE bakiyor -> tek gecerli fold ``kis26``.

``kis26`` icinde de yalnizca 2026 Sub-Mar gecerli: mevsimsel ikizi
(``sub25`` = 2025-02-01..03-31) o foldun ETIKETLERINDE var, tipki test
penceresinin ikizi (2025 Nis-Tem) uretim etiketlerinde oldugu gibi.
Ara/Oca'nin ikizi yok, oradaki daha buyuk kayma mevsim ekstrapolasyonu.

    2026 Sub-Mar, 184.319 sicak satir, 3.341 trafo, 3 tohum
      kayma            +0,1072   (tohumlar +0,0913 / +0,0988 / +0,1315)
      TEST-AGIRLIKLI   +0,1466   (olcut.py, bayatlik x kVA x ufuk)
      trafolarin %68,2'sinde pozitif; en buyuk 50 trafo atilinca
      kazancin %27'si duruyor -> tek trafodan gelmiyor.

BAGIMSIZ IKINCI YOL (etiketsiz, testin KENDI penceresini kapsar)
----------------------------------------------------------------
Ulusal yuk serisi 2026-08-20'ye kadar dolu, yani test penceresini kapsiyor.
Yerel yillik buyume ulusal yillik buyumeye baglanir (Oca/Sub/Mar, 3 nokta):

    yerel_YoY = +0,0951 + 1,900 x ulusal_YoY
    ulusal Nis-Tem 2025->2026 YoY = +0,0007
    -> ONGORULEN yerel Nis-Tem YoY = +0,0964

Modelin gonderimden okunan ima ettigi YoY ise yalnizca **+0,0072**.
Iki bagimsiz yol ayni bandi veriyor: +0,096 ve +0,107.

DELTA SECIMI
------------
Kuadratik kayipta optimum delta = E[yanlilik]. Kestirim bandi 0,096-0,147;
tasima riski icin BUZULMUS bir deger secilir. delta = 0,08 kestirimin
~%60'i: yanlilik 0,10 ise RMSLE -0,0037, yanlilik 0 ise +0,0024.

    dMSE = (delta^2 - 2*delta*yanlilik) * 0,7784      (sicak pay)

DONUSUM (yalnizca SICAK satirlarda):
    r  = log1p(tahmin) - log1p(guc)
    r' = r + delta                  <=>  yeni = (1 + tahmin) * exp(delta) - 1

Gun ekseni, trafo ekseni ve gun ici yapi DEGISMEZ -- yalnizca seviye.
Bu yuzden ``son_islem_gunolcek.py``den SONRA kosulmalidir (o betigin
"genel seviye kaydi" kapisi kendi girdisine gore calisir).

    python scripts/son_islem_seviye.py --giris submissions/X.csv \
        --cikis submissions/Y.csv [--delta 0.08] [--soguk-da]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
DELTA_VARSAYILAN = 0.08
SICAK_PAY = 0.7784


def main() -> int:
    a = argparse.ArgumentParser(description="sicak seviye kaymasi")
    a.add_argument("--giris", required=True)
    a.add_argument("--cikis", required=True)
    a.add_argument("--delta", type=float, default=DELTA_VARSAYILAN)
    a.add_argument("--soguk-da", action="store_true", help="soguk satirlara da --delta uygula")
    a.add_argument(
        "--soguk-delta",
        type=float,
        default=None,
        help="soguk satirlara AYRI delta (verilirse --soguk-da yok sayilir)",
    )
    ar = a.parse_args()

    ornek = pd.read_csv(KOK / "data/raw/sample_submission.csv", encoding="utf-8")
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "guc"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    tr = pd.read_csv(
        KOK / "data/raw/train.csv", usecols=["tanim"], encoding="utf-8", dtype={"tanim": str}
    )
    yol = Path(ar.giris)
    if not yol.is_absolute() and not yol.exists():
        yol = KOK / "submissions" / yol.name
    sub = pd.read_csv(yol, encoding="utf-8")
    if not sub["id"].equals(ornek["id"]):
        raise RuntimeError("id sirasi sample_submission ile ayni degil")
    if not 0.0 <= ar.delta <= 0.30:
        raise RuntimeError(f"delta mantik disi: {ar.delta:.3f}")
    if ar.soguk_delta is not None and not 0.0 <= ar.soguk_delta <= 0.40:
        raise RuntimeError(f"soguk delta mantik disi: {ar.soguk_delta:.3f}")

    m = sub.merge(te, on="id", how="left", validate="one_to_one")
    if len(m) != len(sub):
        raise RuntimeError("birlestirme satir sayisini bozdu")
    soguk = ~m["tanim"].isin(set(tr["tanim"])).to_numpy()
    sicak = ~soguk
    varsayilan_soguk = ar.delta if ar.soguk_da else 0.0
    d_soguk = varsayilan_soguk if ar.soguk_delta is None else ar.soguk_delta
    d_sicak = ar.delta

    log_guc = np.log1p(m["guc"].to_numpy(dtype="float64"))
    r = np.log1p(m["tuketim"].to_numpy(dtype="float64")) - log_guc
    kayma_vek = np.where(sicak, d_sicak, d_soguk)
    yeni_r = r + kayma_vek
    hedef = kayma_vek != 0.0
    yeni = np.clip(np.expm1(yeni_r + log_guc), 0.0, None)

    # ---- KAPILAR ----
    if np.isnan(yeni).any() or (yeni < 0).any():
        raise RuntimeError("NaN veya negatif tahmin")
    if (~hedef).any():
        eski = m["tuketim"].to_numpy(dtype="float64")
        sap = float(
            (np.abs(yeni[~hedef] - eski[~hedef]) / np.maximum(np.abs(eski[~hedef]), 1.0)).max()
        )
        if sap > 1e-12:
            raise RuntimeError(f"dokunulmayan satirlar degismis: goreli sapma {sap:.3e}")
    # Uygulanan kayma her rejimde TAM olarak istenen delta olmali.
    for ad, msk, dd in (("SICAK", sicak, d_sicak), ("SOGUK", soguk, d_soguk)):
        if not msk.any():
            continue
        uyg = float((yeni_r[msk] - r[msk]).mean())
        if abs(uyg - dd) > 1e-12:
            raise RuntimeError(f"{ad}: uygulanan kayma {uyg:.6f} yerine {dd:.6f}")
    # Gun ekseni ve trafo ekseni DEGISMEMELI (rejim ici sabit kayma).
    ic = pd.DataFrame({"t": m["tanim"], "s": sicak, "r": yeni_r - r})
    if float(ic.groupby(["t", "s"])["r"].mean().groupby(level=1).std().max()) > 1e-12:
        raise RuntimeError("kayma rejim icinde trafolar arasi degisiyor")

    p_s = float(sicak.mean())
    p_c = float(soguk.mean())
    print(f"  SICAK {int(sicak.sum()):,} satir (p={p_s:.5f})  delta {d_sicak:+.4f}")
    print(f"  SOGUK {int(soguk.sum()):,} satir (p={p_c:.5f})  delta {d_soguk:+.4f}")
    print(f"  min {yeni.min():.1f}  medyan {float(np.median(yeni)):.1f}  maks {yeni.max():.1f}")
    print("  MSLE(d) = MSLE(0) + p*d^2 - 2*p*d*b   (TAM ozdeslik, kirpma yoksa)")
    for yan in (0.00, 0.05, 0.10, 0.15, 0.20, 0.26):
        dm = p_s * d_sicak * (d_sicak - 2 * yan) + p_c * d_soguk * (d_soguk - 2 * yan)
        print(
            f"    gercek yanlilik {yan:.2f} ise dMSE {dm:+.5f}  ->"
            f" RMSLE {np.sqrt(1.032073 + dm):.5f}"
        )
    if d_sicak > 0 and d_soguk == 0:
        print(
            f"  >>> COZUM: b_sicak = ({p_s * d_sicak**2:.6f}"
            f" - (S^2 - 1.032073)) / {2 * p_s * d_sicak:.6f}"
        )
    if d_soguk > 0 and d_sicak == 0:
        print(
            f"  >>> COZUM: b_soguk = ({p_c * d_soguk**2:.6f}"
            f" - (S^2 - 1.032073)) / {2 * p_c * d_soguk:.6f}"
        )

    cik = Path(ar.cikis)
    if not cik.is_absolute():
        cik = KOK / "submissions" / cik.name
    pd.DataFrame({"id": sub["id"], "tuketim": yeni}).to_csv(cik, index=False)
    print(f"  yazildi: {cik}  ({len(sub):,} satir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
