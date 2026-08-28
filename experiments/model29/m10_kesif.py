import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m1_geriteste import yukle

tr = yukle()
print("tarih araligi", tr.tarih.min(), tr.tarih.max())
print("trafo sayisi", tr.tanim.nunique(), "satir", len(tr))
gunsay = tr.groupby("tanim").tarih.nunique()
print("trafo basina gun sayisi: ", gunsay.describe())
# yogunluk: her trafo icin ilk-son arasi gun sayisi vs gercek satir
rng = tr.groupby("tanim").tarih.agg(["min", "max"])
rng["span"] = (rng["max"] - rng["min"]).dt.days + 1
rng["n"] = gunsay
print("tam dolu trafo orani", (rng.n == rng.span).mean())
print("doluluk orani dagilimi", (rng.n / rng.span).describe())
print("\nilk tarih dagilimi:")
print(rng["min"].dt.to_period("M").value_counts().sort_index().head(20))
print("\nson tarih dagilimi:")
print(rng["max"].dt.to_period("M").value_counts().sort_index().tail(20))
print("\ntuketim: sifir orani", (tr.tuketim == 0).mean(), " negatif", (tr.tuketim < 0).sum())
print(tr.tuketim.describe())
