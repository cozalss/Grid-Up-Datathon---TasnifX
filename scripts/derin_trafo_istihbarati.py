"""315 olu trafonun derin cografi, fiziksel ve altyapi istihbarati."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
D = KOK / "data" / "raw"
EXT = KOK / "data" / "external"
G = KOK / "submissions"

tr = pd.read_csv(D / "train.csv", parse_dates=["tarih"], dtype={"tanim": str})
te = pd.read_csv(D / "test.csv", parse_dates=["tarih"], dtype={"tanim": str})
v89 = pd.read_csv(G / "tuketim_v89_genis_taban.csv")
v83 = pd.read_csv(G / "tuketim_v83_sicak_optimum.csv")

# v89'da degisen 19,839 satiri bul
fark_mask = (v89["tuketim"] != v83["tuketim"]).to_numpy()
degisen_te = te[fark_mask]
olu_trafolar = set(degisen_te["tanim"].unique())

print(f"Toplam Arastirilan Trafo: {len(olu_trafolar):,}")
print(f"Toplam Etkilenen Satir: {len(degisen_te):,}")

# Lokasyon ayristirma
degisen_te = degisen_te.copy()
p = degisen_te["lokasyon"].str.split(">")
degisen_te["il"] = p.str[0].str.strip()
degisen_te["bolge"] = np.where(p.str.len() >= 3, p.str[1].str.strip(), "YOK")
degisen_te["ilce"] = p.str[-1].str.strip()

# Trafo bazli ozet
trafo_meta = degisen_te.drop_duplicates("tanim").copy()

print("\n" + "=" * 80)
print("1. IL VE BOLGE DAGILIMI:")
print("=" * 80)
print(trafo_meta["il"].value_counts())
print("\nBolge Dagilimi:")
print(trafo_meta["bolge"].value_counts())

print("\n" + "=" * 80)
print("2. EN COK OLU TRAFO OLAN ILK 10 ILCE:")
print("=" * 80)
print(trafo_meta["ilce"].value_counts().head(10))

print("\n" + "=" * 80)
print("3. GUC (KAPASITE) DAGILIMI (kVA):")
print("=" * 80)
guc = trafo_meta["guc"]
print(f"Min Guc    : {guc.min():.1f} kVA")
print(f"Medyan Guc : {guc.median():.1f} kVA")
print(f"Ortalama   : {guc.mean():.1f} kVA")
print(f"Max Guc    : {guc.max():.1f} kVA")
print("\nGuc Aralıkları:")
print(pd.cut(guc, bins=[0, 50, 160, 400, 1000, 10000]).value_counts().sort_index())

# 4. Arazi ve Altyapi Karakteristigi
arazi_yol = EXT / "arazi_ortusu_ilce.parquet"
if arazi_yol.exists():
    from gridup.turkish import join_key

    trafo_meta["ilce_key"] = trafo_meta["ilce"].map(join_key)
    arazi = pd.read_parquet(arazi_yol)
    trafo_arazi = trafo_meta.merge(arazi, on="ilce_key", how="left")

    print("\n" + "=" * 80)
    print("4. ARAZI ORTUSU VE SEKTOR ANALIZI (Bu trafolar nerede?):")
    print("=" * 80)
    print(
        f"Ortalama Yerlesim (Sehir/Sanayi) Orani : %{trafo_arazi['yerlesim_orani'].mean() * 100:.1f}"  # noqa: E501
    )  # noqa: E501
    print(
        f"Ortalama Tarim Orani                   : %{trafo_arazi['tarim_orani'].mean() * 100:.1f}"
    )
    print(f"Ortalama Agac / Orman Orani            : %{trafo_arazi['agac_orani'].mean() * 100:.1f}")

# 5. Train Verisindeki Ayrintili Davranis
tr_olu = tr[tr["tanim"].isin(olu_trafolar)]
tr_stats = (
    tr_olu.groupby("tanim")["tuketim"]
    .agg(toplam_gun="count", max_val="max", mean_val="mean")
    .reset_index()
)

hic_olmayan = len(olu_trafolar - set(tr_stats["tanim"]))
sifir_kalan = len(tr_stats[tr_stats["max_val"] == 0])
pozitif_kalan = len(tr_stats[tr_stats["max_val"] > 0])

print("\n" + "=" * 80)
print("5. 455 GUNLUK TRAIN GECMISI SONUCU:")
print("=" * 80)
print(
    f"Egitim Verisinde 455 Gun Kesintisiz SIFIR Kalan : {sifir_kalan:,} trafo (%{sifir_kalan / len(olu_trafolar) * 100:.1f})"  # noqa: E501
)  # noqa: E501
print(
    f"Gecmiste Kisa Sureli Calisip Sonra Kapanan      : {pozitif_kalan:,} trafo (%{pozitif_kalan / len(olu_trafolar) * 100:.1f})"  # noqa: E501
)  # noqa: E501
print(f"Train'de Hic Gorunmeyen                         : {hic_olmayan:,} trafo")
print("=" * 80)
