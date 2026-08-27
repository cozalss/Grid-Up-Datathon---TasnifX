"""Testte etkili 251 olu trafonun tam odakli istihbarati (egitim tarafi 315)."""

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
sota = pd.read_csv(G / "tuketim_sota_v1.csv")

hic_tuketmeyen = set(tr.groupby("tanim")["tuketim"].max().pipe(lambda s: s[s == 0]).index)
sota_maske = sota["tuketim"].to_numpy() == 0.0
mevcut = set(te.loc[sota_maske, "tanim"].unique())
ek = hic_tuketmeyen - mevcut
# DIKKAT: ek grubun bir kismi test setinde hic yok -> maskeye giremez.
# Egitim tarafi 315, TESTTE ETKILI olan 251. Paydayi karistirma.
test_trafolari = set(te["tanim"].unique())
tum_olu_trafolar = (mevcut | ek) & test_trafolari

print(f"Egitim tarafinda aday OLU trafo : {len(mevcut | ek):,}")
print(f"  bunlardan TESTTE olan (etkili) : {len(tum_olu_trafolar):,}")
print(f"  testte olmayan (maskeye giremez): {len((mevcut | ek) - test_trafolari):,}")

te_olu = te[te["tanim"].isin(tum_olu_trafolar)].copy()
p = te_olu["lokasyon"].str.split(">")
te_olu["il"] = p.str[0].str.strip()
te_olu["bolge"] = np.where(p.str.len() >= 3, p.str[1].str.strip(), "YOK")
te_olu["ilce"] = p.str[-1].str.strip()

trafo_meta = te_olu.drop_duplicates("tanim").copy()

print("\n" + "=" * 80)
print("1. TESTTE ETKILI 251 OLU TRAFONUN IL VE BOLGE DAGILIMI:")
print("=" * 80)
print(trafo_meta["il"].value_counts())
print("\nBolge:")
print(trafo_meta["bolge"].value_counts())

print("\n" + "=" * 80)
print("2. EN COK BULUNDUKLARI ILK 10 ILCE:")
print("=" * 80)
print(trafo_meta["ilce"].value_counts().head(10))

print("\n" + "=" * 80)
print("3. GUC VE TIP PROFILI (kVA):")
print("=" * 80)
guc = trafo_meta["guc"]
print(f"Min Guc    : {guc.min():.1f} kVA")
print(f"Medyan Guc : {guc.median():.1f} kVA (Cogu orta olcekli ticari/sanayi trafosu)")
print(f"Ortalama   : {guc.mean():.1f} kVA")
print(f"Max Guc    : {guc.max():.1f} kVA")
print("\nGuc Gruplari:")
print(pd.cut(guc, bins=[0, 100, 400, 1000, 5000]).value_counts().sort_index())

print("\n" + "=" * 80)
print("4. TRAIN.CSV'DEKI TAM TARIHSEL KAYIT DURUMU:")
print("=" * 80)
tr_olu = tr[tr["tanim"].isin(tum_olu_trafolar)]
g_tr = tr_olu.groupby("tanim")["tuketim"].agg(
    toplam_gun="count", sifir_gun=lambda s: (s == 0).sum(), max_tuketim="max"
)

tam_455_gun_sifir = (g_tr["max_tuketim"] == 0).sum()
pozitif_gunu_olan = (g_tr["max_tuketim"] > 0).sum()

print(
    f"Egitim Verisinde 455 Gun Boyunca 1 WATT Bile Elektrik Cekmeyen : {tam_455_gun_sifir} trafo (%{tam_455_gun_sifir / len(tum_olu_trafolar) * 100:.1f})"  # noqa: E501
)  # noqa: E501
print(
    f"Gecmiste Kisa Sure Elektrik Cekip Sonra Tamamen Kapanan         : {pozitif_gunu_olan} trafo (%{pozitif_gunu_olan / len(tum_olu_trafolar) * 100:.1f})"  # noqa: E501
)  # noqa: E501
print("=" * 80)
