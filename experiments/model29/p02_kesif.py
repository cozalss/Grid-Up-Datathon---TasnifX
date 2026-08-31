"""p02: Sifirdan temiz taban -- adim 1, veri kesfi."""
import numpy as np
import pandas as pd

K = "c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
tr = pd.read_csv(f"{K}/data/raw/train.csv", parse_dates=["tarih"])
te = pd.read_csv(f"{K}/data/raw/test.csv", parse_dates=["tarih"])

print("EGITIM", tr.shape, "TEST", te.shape)
print("egitim tarih", tr.tarih.min().date(), "->", tr.tarih.max().date(),
      "gun sayisi", tr.tarih.nunique())
print("test tarih", te.tarih.min().date(), "->", te.tarih.max().date(),
      "gun sayisi", te.tarih.nunique())
print("egitim trafo", tr.tanim.nunique(), "test trafo", te.tanim.nunique())
ort = set(tr.tanim) & set(te.tanim)
print("kesisim trafo", len(ort), "sadece testte", len(set(te.tanim) - set(tr.tanim)))

print("\n--- hedef ---")
print(tr.tuketim.describe())
print("negatif", (tr.tuketim < 0).sum(), "sifir", (tr.tuketim == 0).sum(),
      "nan", tr.tuketim.isna().sum())
print("log1p std", np.log1p(tr.tuketim.clip(lower=0)).std())

print("\n--- eksik ---")
print(tr.isna().sum())
print(te.isna().sum())

print("\n--- guc ---")
print(tr.guc.value_counts().head(15))
print("test guc yeni deger", set(te.guc) - set(tr.guc))
# trafo basina guc sabit mi?
g = tr.groupby("tanim").guc.nunique()
print("guc degisen trafo sayisi", (g > 1).sum())

print("\n--- lokasyon ---")
print("essiz lokasyon", tr.lokasyon.nunique())
sp = tr.lokasyon.str.split(">", expand=True)
for i in range(sp.shape[1]):
    print(f"  seviye{i}", sp[i].nunique(), sp[i].unique()[:8])

print("\n--- trafo basina gun ---")
c = tr.groupby("tanim").size()
print(c.describe())
ct = te.groupby("tanim").size()
print("test trafo basina gun", ct.describe())

print("\n--- gunluk toplam (aylik ortalama) ---")
tr["ay"] = tr.tarih.dt.to_period("M")
print(tr.groupby("ay").agg(n=("tuketim", "size"), ort=("tuketim", "mean"),
                           med=("tuketim", "median"),
                           sifir=("tuketim", lambda s: (s == 0).mean())))
