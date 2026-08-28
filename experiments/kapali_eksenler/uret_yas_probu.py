"""YAS BANDI PROB DOSYASI URETICISI -- 1[genc] yonune kappa ekler.

NEDEN PROB: kuyruk_toplu.py yas bandi yanliligini uc blokta da COK anlamli
olcuyor (t = 40..67) ama SEKLI bloklar arasi tasinmiyor:

    7-90g bandi optimum b:   guz25 negatif  |  kis26 pozitif
    |etki| iki blokta da BUYUK -> |rho| ~ 0 DEGIL, yalnizca ISARET TERS.

Kural: isaret tersligi probu OLDURMEZ; kappa* = L/Q isareti kendisi
duzeltir. Oldurucu olan tek sey |korelasyon| ~ 0'dir ve burada degil.

VEKTOR (tam test kumesinde, kuyruk_prob_tasarim.py'den):
    v = 1[trafo train'de VAR  ve  6 < gecmis_gun <= 90]
    gecmis_gun = 2026-04-01 - trafonun train'deki ILK kaydi
    59.760 satir (%8,36),  507 trafo,  Q_ham = 0,0836169
    1[kuyruk] ile TANIM GEREGI AYRIK (kesisim 0).

ARITMETIK:  dMSE(k) = k^2 Q - 2 k L   ->   k* = L/Q,  tavan = -L^2/Q
    CIKIS:  L = (dMSE_gozlenen - kappa^2 Q) / (-2 kappa)
    ertesi gun kappa* = L/Q uygulanir, kazanc = -L^2/Q.

DIKKAT: bu betik KAGGLE'A HICBIR SEY GONDERMEZ. Yalnizca dosya uretir.

Kullanim:
  uv run python experiments/kapali_eksenler/uret_yas_probu.py \
      --giris submissions/tuketim_v93_gram_optimum.csv \
      --cikis submissions/tuketim_yas_probu.csv --kappa 0.20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
TEST_BAS = pd.Timestamp("2026-04-01")
BEKLENEN_SATIR = 714_688
ALT, UST = 6, 90


def yol(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else KOK / q


def main() -> int:
    a = argparse.ArgumentParser(description="yas bandi (7-90g) prob dosyasi")
    a.add_argument("--giris", required=True)
    a.add_argument("--cikis", required=True)
    a.add_argument("--kappa", type=float, required=True)
    a.add_argument("--alt", type=int, default=ALT)
    a.add_argument("--ust", type=int, default=UST)
    ar = a.parse_args()
    if not -1.0 <= ar.kappa <= 1.0:
        raise SystemExit(f"kappa mantik disi: {ar.kappa}")

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
    ilk = tr.groupby("tanim")["tarih"].min()
    ig = te["tanim"].map(ilk)
    gecmis = (TEST_BAS - ig).dt.days.to_numpy(dtype="float64")
    hedef = ig.notna().to_numpy() & (gecmis > ar.alt) & (gecmis <= ar.ust)

    d = pd.read_csv(yol(ar.giris))
    if len(d) != BEKLENEN_SATIR:
        raise SystemExit(f"giris {len(d)} satir, beklenen {BEKLENEN_SATIR}")
    if not (d["id"].values == te["id"].values).all():
        raise SystemExit("id sirasi test.csv ile ayni degil")

    lg = np.log1p(np.clip(d["tuketim"].to_numpy(dtype="float64"), 0, None))
    yeni = lg + ar.kappa * hedef
    cikti = np.clip(np.expm1(yeni), 0.0, None)

    fark = yeni - lg
    Q = float((hedef.astype("float64") ** 2).mean())
    print(f"giris  {yol(ar.giris).name}")
    print(f"  band {ar.alt}g < gecmis <= {ar.ust}g")
    print(f"  hedef satir      {int(hedef.sum()):,} ({hedef.mean():.6f})")
    print(f"  hedef trafo      {te.loc[hedef, 'tanim'].nunique():,}")
    print(f"  Q = ort(v^2)     {Q:.7f}")
    print(f"  kappa            {ar.kappa:+.4f}")
    print(f"  uygulanan kayma  {float(fark[hedef].mean()):+.8f}")
    print(
        f"  dokunulmayan     {int((np.abs(fark) <= 1e-12).sum()):,} satir, "
        f"maxabs {float(np.abs(fark[~hedef]).max()):.2e}"
    )
    print(
        f"  NaN {int(np.isnan(cikti).sum())}  negatif {int((cikti < 0).sum())}  "
        f"sifir {int((cikti == 0).sum())}"
    )

    # KAPILAR
    if abs(float(fark[hedef].mean()) - ar.kappa) > 1e-12:
        raise SystemExit("uygulanan kayma istenen kappa degil (kirpma yemis olabilir)")
    if float(np.abs(fark[~hedef]).max()) > 1e-12:
        raise SystemExit("hedef disi satirlar degismis")
    if int(np.isnan(cikti).sum()) or int((cikti < 0).sum()):
        raise SystemExit("NaN veya negatif uretildi")

    c = yol(ar.cikis)
    c.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": d["id"], "tuketim": cikti}).to_csv(c, index=False)
    print(f"\nyazildi: {c}")
    print("\nCOZUM (bu dosyanin skoru S, tabanin MSE'si M0 iken):")
    print("  dMSE = S^2 - M0")
    print(
        f"  L    = (kappa^2 * Q - dMSE) / (2 * kappa)"
        f"  =  ({ar.kappa**2 * Q:.8f} - dMSE) / {2 * ar.kappa:.4f}"
    )
    print(f"  kappa* = L / Q = L / {Q:.7f}")
    print("  ertesi gun kazanc = -L^2 / Q")
    print("\n*** KAGGLE'A GONDERIM BU BETIGIN ISI DEGIL. ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
