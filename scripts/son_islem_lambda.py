"""TRAFO BAZINDA GUN DUYARLILIGI (lambda) DUZELTMESI -- etkilesim terimi.

NEDEN
-----
Sicak MSE ayrisimi (olcut.py agirliklari, uretim harmani):

    sabit  %4,5-7,1 | TRAFO %46,6-69,9 | GUN %0,3-5,3 | ETKILESIM %24,4-43,6

Gun ANA etkisi kutlenin kucuk bir parcasi (``son_islem_gunolcek.py`` onu
duzeltiyor). Asil kutle ETKILESIMDE: her trafonun ortak gun faktorune KENDI
duyarliligi var.

    r_it = a_i + lambda_i * b_t + e_it

Model lambda'yi neredeyse SABIT aliyor. Olculdu (uretim harmani, trafo bazinda
regresyon egimi):

    blok    MODEL lambda        GERCEK lambda       korelasyon   R^2
    yaz25   +0,344 (std 0,151)  +0,951 (std 1,220)    +0,098     0,010
    guz25   +0,805 (std 0,576)  +0,924 (std 1,396)    +0,264     0,070
    kis26   +0,219 (std 0,447)  +0,998 (std 1,770)    +0,198     0,039

Yani model her trafoya AYNI ve cok kucuk duyarlilik atiyor.

OLCULEN KAZANC ve NULL SINAMASI
-------------------------------
Duzeltme (yaz25 sicak, olcut.py agirliklari, uc tohum):

    r' = r + [c * (1 + m*(lambda_i - 1)) - 1] * b_t

``lambda_i`` DOGRULAMA BLOGUNUN ETIKETINE HIC DOKUNMADAN, baska bir donemin
ham verisinden kestirilir.

    lambda kaynagi                         en iyi m   ek kazanc    genele
    Eyl-Ara 2025 (komsu, kor +0,657)         0,35     +0,00418   -0,00224
    Ara-Mar      (kis,   kor -0,099)          --      HER m'de KOTU (-0,01179)

NULL SINAMASI GECTI: lambda gercekten tasindiginda kazandiriyor, tasinmadiginda
duzeltme MONOTON olarak zarar veriyor. Yani kazanc, duzeltmenin bicimsel
esnekliginden degil, TASINAN BILGIDEN geliyor.

KAYNAK SECIMI -- lambda MEVSIME OZGUDUR
---------------------------------------
Hedef 2026 Ocak-Mart lambdasi olarak sinandi:

    2025 Oca-Mar (dar, AYNI mevsim)    kor +0,400   R^2 0,160
    2025 Oca-Haz (genis)               kor +0,157   R^2 0,025
    2025 tam yil (en genis)            kor +0,029   R^2 0,001
    2025 Eki-Ara (komsu, HEMEN ONCE)   kor +0,512   R^2 0,262

Pencereyi genisletmek sinyali YOK EDIYOR. Komsu donem en iyi ama TEST icin
komsu donem Ocak-Mart 2026'dir ve o KIStir; yaz lambdasiyla korelasyonu
-0,099. O yuzden test icin tek mesru kaynak **2025 Nisan-Temmuz** -- ayni
mevsim, bir yil once, beklenen korelasyon ~0,40.

BEKLENTI DURUST TUTULUR
-----------------------
Ust sinir (kor 0,657) genele -0,00224 verdi. Optimal buzulmus kestiricide
kazanc ~rho^2 ile olcekelenir, yani kor 0,40 icin (0,40/0,657)^2 = 0,371 ->
beklenen genele **-0,0008**. ``m`` de ayni oranla kucultulur: 0,35 x 0,371
= 0,13.

    python scripts/son_islem_lambda.py --giris submissions/X.csv \
        --cikis submissions/Y.csv [--m 0.13]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
#: Ust sinir m=0,35 (kor 0,657). Test kaynagi kor ~0,40 -> 0,35 * (0,40/0,657)^2.
M_VARSAYILAN = 0.13
#: lambda kaynagi: testin mevsimsel ikizi, bir yil once.
KAYNAK_BASI, KAYNAK_SONU = "2025-04-01", "2025-07-31"
ASGARI_GUN = 40
ASGARI_DEN = 0.05


def gun_faktoru(tanim: np.ndarray, gun: np.ndarray, r: np.ndarray) -> pd.Series:
    """Trafo etkisi cikarilmis ortak gun faktoru, merkezlenmis."""
    c = r - pd.Series(r).groupby(tanim).transform("mean").to_numpy()
    b = pd.Series(c).groupby(gun).mean()
    return b - b.mean()


def lambdalar(df: pd.DataFrame) -> pd.Series:
    """Trafo bazinda gun duyarliligi: c_it = lambda_i * b_t + hata."""
    r = np.log1p(df["tuketim"].to_numpy(dtype="float64")) - np.log1p(
        df["guc"].to_numpy(dtype="float64")
    )
    tanim = df["tanim"].to_numpy()
    gun = df["tarih"].to_numpy()
    b = gun_faktoru(tanim, gun, r)
    x = pd.DataFrame(
        {
            "t": tanim,
            "c": r - pd.Series(r).groupby(tanim).transform("mean").to_numpy(),
            "b": pd.Series(gun).map(b).to_numpy(),
        }
    )
    g = x.groupby("t")
    num = g.apply(lambda q: float((q["c"] * q["b"]).sum()), include_groups=False)
    den = g.apply(lambda q: float((q["b"] ** 2).sum()), include_groups=False)
    n = g.size()
    return (num / den)[(n >= ASGARI_GUN) & (den > ASGARI_DEN)]


def main() -> int:
    a = argparse.ArgumentParser(description="trafo bazinda gun duyarliligi duzeltmesi")
    a.add_argument("--giris", required=True)
    a.add_argument("--cikis", required=True)
    a.add_argument("--m", type=float, default=M_VARSAYILAN)
    ar = a.parse_args()

    ornek = pd.read_csv(KOK / "data/raw/sample_submission.csv", encoding="utf-8")
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "guc", "tarih"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "guc", "tarih", "tuketim"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    yol = Path(ar.giris)
    if not yol.is_absolute() and not yol.exists():
        yol = KOK / "submissions" / yol.name
    sub = pd.read_csv(yol, encoding="utf-8")
    if not sub["id"].equals(ornek["id"]):
        raise RuntimeError("id sirasi sample_submission ile ayni degil")

    m = sub.merge(te, on="id", how="left", validate="one_to_one")
    if len(m) != len(sub):
        raise RuntimeError("birlestirme satir sayisini bozdu")
    soguk = ~m["tanim"].isin(set(tr["tanim"])).to_numpy()
    # SOGUK trafolarin gecmisi YOK, yani lambda kestirilemez -- dokunulmaz.
    hedef = ~soguk

    kay = tr[(tr["tarih"] >= KAYNAK_BASI) & (tr["tarih"] <= KAYNAK_SONU) & (tr["tuketim"] > 0)]
    lam = lambdalar(kay)

    log_guc = np.log1p(m["guc"].to_numpy(dtype="float64"))
    r = np.log1p(m["tuketim"].to_numpy(dtype="float64")) - log_guc
    tanim_h = m.loc[hedef, "tanim"].to_numpy()
    gun_h = m.loc[hedef, "tarih"].to_numpy()
    b = gun_faktoru(tanim_h, gun_h, r[hedef])
    b_row = pd.Series(gun_h).map(b).to_numpy(dtype="float64")
    b_row = b_row - b_row.mean()

    li = pd.Series(tanim_h).map(lam).to_numpy(dtype="float64")
    kapsam = float(np.isfinite(li).mean())
    li = np.where(np.isfinite(li), li, 1.0)
    # Kapsanmayan trafoya lambda=1 verilir, yani ek duzeltme almaz.
    # CIFT MERKEZLEME. Duzeltme SAF ETKILESIM olmali: ne trafo eksenini ne de
    # ortak gun faktorunu kaydirmali. Tek yonlu merkezleme YETMEZ -- kapi bunu
    # iki kez yakaladi: gun_faktoru once TRAFO ortalamasini cikardigi icin,
    # etkinin trafo bazindaki ortalamasi sifir degilse gun eksenine sizar.
    # Dengesiz panelde tam izdusum icin donusumlu supurge yapilir; birkac
    # supurgede makine hassasiyetine iner ve kapi bunu dogrular.
    etki = ar.m * (li - 1.0) * b_row
    es = pd.Series(etki)
    for _ in range(20):
        es = es - es.groupby(tanim_h).transform("mean")
        es = es - es.groupby(gun_h).transform("mean")
    etki = es.to_numpy(dtype="float64")

    yeni_r = r.copy()
    yeni_r[hedef] = r[hedef] + etki
    yeni = np.clip(np.expm1(yeni_r + log_guc), 0.0, None)

    # ---- KAPILAR ----
    if np.isnan(yeni).any() or (yeni < 0).any():
        raise RuntimeError("NaN veya negatif tahmin")
    eski_c = m.loc[soguk, "tuketim"].to_numpy(dtype="float64")
    sap = float((np.abs(yeni[soguk] - eski_c) / np.maximum(np.abs(eski_c), 1.0)).max())
    if sap > 1e-12:
        raise RuntimeError(f"soguk satirlar degisti: goreli sapma {sap:.3e}")
    kayma = float(abs(yeni_r[hedef].mean() - r[hedef].mean()))
    if kayma > 1e-9:
        raise RuntimeError(f"genel seviye kaydi: {kayma:.3e}")
    yeni_b = gun_faktoru(tanim_h, gun_h, yeni_r[hedef])
    if float((yeni_b - b).abs().max()) > 1e-9:
        raise RuntimeError("ortak gun faktoru degisti -- duzeltme yalnizca ETKILESIME dokunmali")

    print(f"  hedef satir {int(hedef.sum()):,} / {len(m):,}  (yalniz SICAK)")
    print(
        f"  lambda kaynagi {KAYNAK_BASI}..{KAYNAK_SONU}  {len(lam):,} trafo"
        f"  ort {lam.mean():+.3f}  std {lam.std():.3f}"
    )
    print(f"  kapsanan satir %{100 * kapsam:.1f}  (kapsanmayan lambda=1 alir)")
    print(f"  m = {ar.m:.3f}   etki std {etki.std():.5f}")
    print(f"  ORTAK GUN FAKTORU korundu; genel seviye kaymasi {kayma:.1e}")
    print(f"  SOGUK satirlar dokunulmadi, goreli sapma {sap:.1e}")
    print(f"  min {yeni.min():.1f}  medyan {float(np.median(yeni)):.1f}  maks {yeni.max():.1f}")

    cik = Path(ar.cikis)
    if not cik.is_absolute():
        cik = KOK / "submissions" / cik.name
    pd.DataFrame({"id": sub["id"], "tuketim": yeni}).to_csv(cik, index=False)
    print(f"  yazildi: {cik}  ({len(sub):,} satir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
