"""v89: genisletilmis olu maske + ay bazli optimum taban.

Iki iyilestirme:

1) MASKE GENISLEMESI. SOTA maskesi 193 trafo yakaliyor, ama egitimde 455 gun
   boyunca HIC tuketmemis 298 trafo var. Kacan 122 trafonun olculmus davranisi
   maskedekilerden DAHA guvenilir:
     Nis-Tem 2025      4.514 satir  sifir orani %100.00
     Ara-Mar son 122g  3.016 satir  sifir orani %100.00
   Testte 5.355 satir, 0.06964 MSE tasiyorlar.

2) AY BAZLI TABAN. Sert sifir yerine, her ay icin MSLE-optimum sabit
   v* = (1-p)*b; p ve b o ay icin tam bu trafolardan olculdu:
     Nisan  p=%96.8 b=log1p(31)    -> 0.12 kWh
     Mayis  p=%96.3 b=log1p(66)    -> 0.17 kWh
     Haziran p=%91.2 b=log1p(938)  -> 0.82 kWh
     Temmuz p=%72.3 b=log1p(1543)  -> 6.67 kWh
   Temmuz'daki yuksek taban sulama reaktivasyonuna karsi sigorta.

Beklenen: olculen p'de 0.88447, her ay 20 puan kotuyse 0.94730.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
G = KOK / "submissions"
D = KOK / "data" / "raw"

#: mevcut maske icin ay -> (olculen sifir orani, yanlis satirin ort log1p)
AY_OLCUM: dict[int, tuple[float, float]] = {
    4: (0.9676, float(np.log1p(31))),
    5: (0.9627, float(np.log1p(66))),
    6: (0.9122, float(np.log1p(938))),
    7: (0.7226, float(np.log1p(1543))),
}
#: genisleme grubu -- iki dogrulama penceresinde de %100; ihtiyatla %97 alindi
EK_OLCUM: tuple[float, float] = (0.97, 7.0)


def main() -> int:
    taban = pd.read_csv(G / "tuketim_v83_sicak_optimum.csv")
    sota = pd.read_csv(G / "tuketim_sota_v1.csv")
    te = pd.read_csv(D / "test.csv", usecols=["id", "tanim", "tarih"], dtype={"tanim": str})
    if not taban["id"].equals(sota["id"]) or not taban["id"].equals(te["id"]):
        raise RuntimeError("id sirasi eslesmiyor")

    tr = pd.read_csv(D / "train.csv", usecols=["tanim", "tuketim"], dtype={"tanim": str})
    hic_tuketmeyen = set(tr.groupby("tanim")["tuketim"].max().pipe(lambda s: s[s == 0]).index)

    sota_maske = sota["tuketim"].to_numpy() == 0.0
    mevcut = set(te.loc[sota_maske, "tanim"].unique())
    ek = hic_tuketmeyen - mevcut
    ek_maske = te["tanim"].isin(ek).to_numpy()

    ay = pd.to_datetime(te["tarih"]).dt.month.to_numpy()
    tahmin = taban["tuketim"].to_numpy().astype("float64").copy()

    for a, (p, b) in AY_OLCUM.items():
        m = sota_maske & (ay == a)
        tahmin[m] = float(np.expm1((1.0 - p) * b))
    p, b = EK_OLCUM
    tahmin[ek_maske] = float(np.expm1((1.0 - p) * b))

    yol = G / "tuketim_v89_genis_taban.csv"
    pd.DataFrame({"id": taban["id"], "tuketim": tahmin}).to_csv(yol, index=False)
    print(f"YAZILDI {yol.name}")
    print(f"  mevcut maske  : {int(sota_maske.sum()):>7,} satir / {len(mevcut):>3} trafo")
    print(f"  genisleme     : {int(ek_maske.sum()):>7,} satir / {len(ek):>3} trafo")
    print(f"  TOPLAM        : {int((sota_maske | ek_maske).sum()):>7,} satir")
    for a, (p, b) in AY_OLCUM.items():
        print(f"    ay {a}: taban {np.expm1((1 - p) * b):.2f} kWh  (p=%{100 * p:.1f})")
    print(f"    ek grup: taban {np.expm1((1 - EK_OLCUM[0]) * EK_OLCUM[1]):.2f} kWh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
