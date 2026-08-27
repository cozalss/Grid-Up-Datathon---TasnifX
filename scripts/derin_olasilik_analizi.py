"""193 sifirlanan trafonun gecmis davranisi ve Bayesian uyanma olasiligi."""

from __future__ import annotations

import pandas as pd

tr = pd.read_csv("data/raw/train.csv")
te = pd.read_csv("data/raw/test.csv")
v87 = pd.read_csv("submissions/tuketim_v87_olu_izole.csv")
v83 = pd.read_csv("submissions/tuketim_v83_sicak_optimum.csv")

tr["tarih"] = pd.to_datetime(tr["tarih"])
te["tarih"] = pd.to_datetime(te["tarih"])

# Sifirlanan 14,484 satirin trafolari:
sifirlanan_id = set(v87.loc[v87["tuketim"] == 0.0, "id"])
sifirlanan_te = te[te["id"].isin(sifirlanan_id)]
sifirlanan_trafolar = set(sifirlanan_te["tanim"].unique())

print(f"Toplam Sifirlanan Satir: {len(sifirlanan_id):,}")
print(f"Toplam Sifirlanan Trafo: {len(sifirlanan_trafolar):,}")

# Bu trafolarin train.csv'deki gecmisleri:
tr_olu = tr[tr["tanim"].isin(sifirlanan_trafolar)]

# Trafo bazli toplam gun ve sifir gun sayilari:
stats = (
    tr_olu.groupby("tanim")
    .agg(
        toplam_gun=("tuketim", "count"),
        sifir_gun=("tuketim", lambda s: (s == 0).sum()),
        pozitif_gun=("tuketim", lambda s: (s > 0).sum()),
        ortalama_tuketim=("tuketim", "mean"),
        max_tuketim=("tuketim", "max"),
    )
    .reset_index()
)

# Hic train verisi olmayan trafolar:
train_olmayan = sifirlanan_trafolar - set(stats["tanim"])
print(f"Train'de hic kaydi olmayan (soguk) sifirlanan trafo: {len(train_olmayan)}")

tam_455_gun_sifir = stats[stats["pozitif_gun"] == 0]
gecmiste_pozitif_olan = stats[stats["pozitif_gun"] > 0]

print("\n193 Trafonun Gecmis Dagilimi:")
print(f"  1. 455 Gun Boyunca (1.5 Yil) KESINTISIZ SIFIR Olanlar: {len(tam_455_gun_sifir):,} trafo")
print(f"  2. Gecmiste kisa sure calisip sonra susanlar: {len(gecmiste_pozitif_olan):,} trafo")

print("\nGecmiste pozitif olanlarin ozeti:")
for _, row in gecmiste_pozitif_olan.iterrows():
    print(
        f"  Trafo: {row['tanim']}, Toplam Gun: {row['toplam_gun']}, Pozitif Gun: {row['pozitif_gun']}, Ort: {row['ortalama_tuketim']:.1f}"  # noqa: E501
    )  # noqa: E501

# Simdi genel train.csv icinde 455 gun boyunca 0 olan baska trafo var mi?
tum_stats = (
    tr.groupby("tanim")
    .agg(
        toplam_gun=("tuketim", "count"),
        pozitif_gun=("tuketim", lambda s: (s > 0).sum()),
    )
    .reset_index()
)

# 2025 ilk 180 gunu 0 olup sonraki 180 gunu pozitif olan var mi?
tr25_ilk = tr[tr["tarih"] <= "2025-06-30"].groupby("tanim")["tuketim"].sum()
tr25_son = tr[tr["tarih"] >= "2025-07-01"].groupby("tanim")["tuketim"].sum()

ilk_sifir_son_pozitif = set(tr25_ilk[tr25_ilk == 0].index) & set(tr25_son[tr25_son > 0].index)
print(
    f"\n2025 ilk 6 ay sifir olup 2. 6 ay uyanan trafo orani: {len(ilk_sifir_son_pozitif):,} trafo"
)  # noqa: E501
