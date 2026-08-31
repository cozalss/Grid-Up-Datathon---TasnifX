import numpy as np, pandas as pd
K = "c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
e = pd.read_parquet(f"{K}/data/interim/deney/egitim.parquet")
t = pd.read_parquet(f"{K}/data/interim/deney/test.parquet")
print("EGITIM", e.shape); print(e.dtypes)
print(e.head(3).to_string())
print("\nbloklar"); print(e.groupby("_blok").agg(n=("tuketim","size"),
    t0=("tarih","min"), t1=("tarih","max"), soguk=("soguk_mu","mean")))
print("\nTEST", t.shape); print(list(t.columns))
print("test soguk orani", t.soguk_mu.mean(), "ufuk", t.ufuk_gun.min(), t.ufuk_gun.max())
print("\nindeks tipi", e.index[:5].tolist())
b = e[e._blok=="yaz25"]
print("yaz25 n", len(b), "trafo", b.tanim.nunique() if "tanim" in b else "?")
