"""OLU TRAFO HEDGE -- tam sifir tahminlerini olculmus optimal tabanla degistirir.

NEDEN
-----
Uretim ``np.clip(np.expm1(log_tahmin), 0.0, None)`` uyguluyor. Log tahmini
sifirin altina dustugunde cikti TAM SIFIR oluyor: v23'te 8.748 satir, 121
trafo. Bu dusunulmus bir tahmin degil, KIRPMA ARTIGI -- ayni trafolarin
diger gunlerinde model 0,05-0,26 arasi degerler yaziyor (ortanca 0,13).

RMSLE'de tam sifir asimetrik bir bahis: trafo gercekten oluyse bedava,
dirilirse satir basina kare hata ~48. Ve olculdu -- olu trafolar diriliyor:

    kesme 2025-03-31, ileri Nisan-Temmuz (TESTIN MEVSIMSEL ESI)
      1-15 gun olu     %33,5 sifir-olmayan   optimal log1p 1,030
     15-30 gun olu     %21,0                 optimal        1,103
     30-60 gun olu     %20,8                 optimal        2,121
     60-90 gun olu     % 2,9                 optimal        0,161
       90+ gun olu     % 3,4                 optimal        0,230

    kesme 2025-09-30, ileri Ekim-Ocak (BAGIMSIZ IKINCI OLCUM)
      1-15  %41,3 -> 2,069 | 15-30 %23,5 -> 0,855 | 30-60 % 9,7 -> 1,071
      60-90 % 8,9 -> 0,883 |   90+ %11,0 -> 0,867

Iki pencere de ayni yonu soyluyor. Ince kovalar (30-60: analogda 13 trafo)
kirilgan oldugu icin her kovada IKI OLCUMUN KUCUGU kullaniliyor.

    python scripts/olu_hedge.py --girdi submissions/tuketim_v23.csv \
                                --cikti submissions/tuketim_v25_hedge.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]

#: (alt, ust) -> log1p tabani. Iki bagimsiz kesmenin KUCUGU -- muhafazakar.
KOVALAR: tuple[tuple[int, int, float], ...] = (
    (1, 15, 1.030),
    (15, 30, 0.855),
    (30, 60, 1.071),
    (60, 90, 0.161),
    (90, 10**6, 0.230),
)
VARSAYILAN = 0.230  # egitimde hic gorulmemis trafo -> en kalabalik kovanin tabani


def olu_gun_sayisi(egitim: pd.DataFrame) -> pd.Series:
    """Her trafo icin egitim sonundaki ardisik sifir gun sayisi."""
    d = egitim.sort_values(["tanim", "tarih"])
    sifir = (d["tuketim"] <= 0).to_numpy()
    # her trafonun son kaydindan geriye dogru kac sifir var
    d = d.assign(_s=sifir)
    kuyruk = d.groupby("tanim")["_s"].apply(
        lambda s: int(np.argmin(s.to_numpy()[::-1])) if not s.to_numpy()[::-1].all() else len(s)
    )
    canli = d.groupby("tanim")["_s"].last()
    return kuyruk.where(canli, 0).astype(int)


def taban(olu: float) -> float:
    for alt, ust, h in KOVALAR:
        if alt <= olu < ust:
            return h
    return 0.0


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--girdi", required=True)
    a.add_argument("--cikti", required=True)
    a.add_argument("--olcek", type=float, default=1.0, help="tabanlari olcekle (0=kapali)")
    ar = a.parse_args()

    tr = pd.read_csv(KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str})
    tr["tarih"] = pd.to_datetime(tr["tarih"])
    olu = olu_gun_sayisi(tr)

    te = pd.read_csv(
        KOK / "data/raw/test.csv", usecols=["id", "tanim"], encoding="utf-8", dtype={"tanim": str}
    )
    sub = pd.read_csv(ar.girdi, encoding="utf-8")
    n0 = len(sub)
    m = sub.merge(te, on="id", how="left", validate="one_to_one")
    if len(m) != n0:
        raise RuntimeError(f"birlestirme satir sayisini bozdu: {n0} -> {len(m)}")

    hedef = m["tuketim"] <= 0
    m["_olu"] = m["tanim"].map(olu)
    h = m.loc[hedef, "_olu"].map(lambda v: VARSAYILAN if pd.isna(v) else taban(float(v)))
    m.loc[hedef, "tuketim"] = np.expm1(h.to_numpy() * ar.olcek)

    out = m[["id", "tuketim"]]
    if out["tuketim"].isna().any() or (out["tuketim"] < 0).any():
        raise RuntimeError("cikti NaN ya da negatif iceriyor")
    out.to_csv(ar.cikti, index=False)

    print(f"girdi  {ar.girdi}")
    print(f"cikti  {ar.cikti}")
    print(f"  degisen satir {int(hedef.sum())} / {n0}  ({100 * hedef.mean():.2f}%)")
    print(f"  farkli trafo  {m.loc[hedef, 'tanim'].nunique()}")
    for alt, ust, hh in KOVALAR:
        k = int(((m["_olu"] >= alt) & (m["_olu"] < ust) & hedef).sum())
        if k:
            ust_etiket = ust if ust < 10**6 else 999
            print(f"    {alt:3d}-{ust_etiket:<4} {k:5d} satir -> log1p {hh * ar.olcek:.3f}")
    print(f"  yeni min tahmin {out['tuketim'].min():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
