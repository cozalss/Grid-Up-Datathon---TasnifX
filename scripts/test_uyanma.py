"""2025 ilkbaharinda sifir olan trafolar yazin calisiyor mu test et."""

from __future__ import annotations

import pandas as pd

tr = pd.read_csv("data/raw/train.csv")
tr["tarih"] = pd.to_datetime(tr["tarih"])

# 2025 Mart ayinda (son 14 gun) sifir olan trafolar:
mart_son = tr[(tr["tarih"] >= "2025-03-15") & (tr["tarih"] <= "2025-03-31")]
mart_ort = mart_son.groupby("tanim")["tuketim"].mean()
mart_sifir_trafo = set(mart_ort[mart_ort == 0].index)

print(f"2025 Mart Sonunda Tamamen Sifir Olan Trafo Sayisi: {len(mart_sifir_trafo):,}")

# Bu trafolar 2025 Yazinda (Nisan-Temmuz) ne yapti?
yaz25 = tr[
    (tr["tarih"] >= "2025-04-01")
    & (tr["tarih"] <= "2025-07-31")
    & tr["tanim"].isin(mart_sifir_trafo)
]
yaz_ort = yaz25.groupby("tanim")["tuketim"].mean()

uyanmis_trafo = (yaz_ort > 0).sum()
hala_sifir = (yaz_ort == 0).sum()

print(
    f"  Yazin (Nisan-Temmuz) Tekrar Elektrik Tuketmeye Baslayan Trafo Sayisi: {uyanmis_trafo:,} (%{uyanmis_trafo / len(mart_sifir_trafo) * 100:.1f})"  # noqa: E501
)
print(
    f"  Yazin da Sifir Kalan Trafo Sayisi: {hala_sifir:,} (%{hala_sifir / len(mart_sifir_trafo) * 100:.1f})"  # noqa: E501
)
