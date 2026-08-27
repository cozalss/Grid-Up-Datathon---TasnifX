"""Test.csv icindeki gun sayilari dagilimini incele."""

import pandas as pd

te = pd.read_csv("data/raw/test.csv")
sayilar = te.groupby("tanim")["tarih"].count()
print(f"Toplam Trafo: {len(sayilar):,}")
print("Gun sayisi dagilimi:")
print(sayilar.value_counts())
