"""H2-A: Degistirme eslestirmesi -- KAC eslesme var, YANLIS POZITIF orani ne?

Hipotez: test-only (soguk) trafolarin bir kismi yeni degil, DEGISTIRILMIS
birim. Eskisi hizmetten cikti, yenisi ayni lokasyon+kVA ile devreye girdi.

Bu betik sadece SAYAR. Eslesme sayisi cok azsa (< ~100) burada duruyoruz.
Anahtarlar: lokasyon (tam string), guc (kVA), zamanlama (eski olum ~ yeni dogum).
`tanim` string yapisi KULLANILMIYOR (yasak bolge: kimlik komsulugu R2 0,019).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KOK = "C:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
TRAIN_SON = pd.Timestamp("2026-03-31")
TEST_ILK = pd.Timestamp("2026-04-01")


def yukle():
    tr = pd.read_csv(f"{KOK}/data/raw/train.csv", parse_dates=["tarih"])
    te = pd.read_csv(f"{KOK}/data/raw/test.csv", parse_dates=["tarih"])
    return tr, te


def trafo_ozet(df: pd.DataFrame, hedef_var: bool) -> pd.DataFrame:
    g = df.groupby("tanim")
    ozet = pd.DataFrame(
        {
            "ilk": g["tarih"].min(),
            "son": g["tarih"].max(),
            "n": g["tarih"].size(),
        }
    )
    ozet["guc"] = g["guc"].median()
    ozet["lokasyon"] = g["lokasyon"].agg(lambda s: s.mode().iloc[0])
    if hedef_var:
        ozet["ort_tuketim"] = g["tuketim"].mean()
        ozet["log_ort"] = g["tuketim"].apply(lambda s: np.log1p(s).mean())
    return ozet.reset_index()


def main() -> None:
    tr, te = yukle()
    o_tr = trafo_ozet(tr, True)
    o_te = trafo_ozet(te, False)

    tr_set = set(o_tr["tanim"])
    te_set = set(o_te["tanim"])
    soguk = o_te[~o_te["tanim"].isin(tr_set)].copy()
    sicak = o_te[o_te["tanim"].isin(tr_set)].copy()
    yalniz_tr = o_tr[~o_tr["tanim"].isin(te_set)].copy()

    print(f"train trafo {len(o_tr)}  test trafo {len(o_te)}")
    print(f"SOGUK (test-only) {len(soguk)}  SICAK {len(sicak)}  train-only {len(yalniz_tr)}")

    # OLU tanimi: train icinde son kaydi TRAIN_SON'dan cok once biten trafolar.
    # Iki grup: (a) test'te hic yok = 332 kesin cikmis, (b) test'te var ama
    # train'de erken olmus -- bunlar degistirme adayi DEGIL (ayni tanim donmus).
    for esik in (7, 14, 30, 60, 90):
        olu = o_tr[o_tr["son"] < TRAIN_SON - pd.Timedelta(days=esik)]
        olu_yalniz = yalniz_tr[yalniz_tr["son"] < TRAIN_SON - pd.Timedelta(days=esik)]
        print(
            f"  son kayit > {esik:3d} gun once: tum train {len(olu):5d} | train-only {len(olu_yalniz):5d}"
        )

    # Soguk trafolarin test icindeki ILK gunu -- "dogum" vekili
    print("\nSOGUK ilk gun dagilimi:")
    print(soguk["ilk"].dt.to_period("M").value_counts().sort_index())

    # --- ESLESTIRME HAVUZU ---
    # Aday olu: train-only + son kayit 2026-03-31'den >= 7 gun once
    olu = yalniz_tr[yalniz_tr["son"] < TRAIN_SON - pd.Timedelta(days=7)].copy()
    print(f"\nAday OLU havuzu (train-only, >=7g once olmus): {len(olu)}")

    # Anahtar: (lokasyon, guc)
    olu["anahtar"] = olu["lokasyon"] + "|" + olu["guc"].astype(int).astype(str)
    soguk["anahtar"] = soguk["lokasyon"] + "|" + soguk["guc"].astype(int).astype(str)

    olu_say = olu["anahtar"].value_counts()
    soguk_say = soguk["anahtar"].value_counts()
    ortak = olu_say.index.intersection(soguk_say.index)
    print(f"ortak (lokasyon,kVA) anahtar sayisi: {len(ortak)}")

    # TEKIL eslesme: anahtarda tam 1 olu ve tam 1 soguk
    tekil = [k for k in ortak if olu_say[k] == 1 and soguk_say[k] == 1]
    print(f"TEKIL 1-1 eslesme: {len(tekil)}")
    coklu_soguk = int(soguk_say.reindex(ortak).sum())
    print(f"anahtari eslesen toplam soguk trafo: {coklu_soguk}")

    # YANLIS POZITIF vekili: ayni anahtardaki soguk/olu kardinalitesi
    kard = pd.DataFrame({"olu": olu_say.reindex(ortak), "soguk": soguk_say.reindex(ortak)})
    print("\nkardinalite dagilimi (olu x soguk):")
    print(kard.groupby(["olu", "soguk"]).size().head(20))

    # AMBIYANS TESTI: bir soguk trafo, ayni lokasyon+kVA'da KAC CANLI (test'te
    # de olan) trafo ile komsu? Cok varsa "eslesme" bilgi tasimaz.
    o_te["anahtar"] = o_te["lokasyon"] + "|" + o_te["guc"].astype(int).astype(str)
    tum_say = o_te["anahtar"].value_counts()
    soguk["kardes_test"] = tum_say.reindex(soguk["anahtar"]).to_numpy()
    print("\nsoguk trafonun (lokasyon,kVA) anahtarindaki TOPLAM test trafo sayisi:")
    print(soguk["kardes_test"].value_counts().sort_index().head(15))

    # Daha gevsek anahtar: sadece lokasyon
    olu_lok = olu["lokasyon"].value_counts()
    soguk_lok = soguk["lokasyon"].value_counts()
    print(f"\nsadece-lokasyon: ortak {len(olu_lok.index.intersection(soguk_lok.index))}")
    print(
        f"lokasyon esssiz sayisi train {o_tr['lokasyon'].nunique()} test {o_te['lokasyon'].nunique()}"
    )
    print("lokasyon basina ort trafo (test):", len(o_te) / o_te["lokasyon"].nunique())

    # ZAMANLAMA: tekil eslesmelerde olum gunu ile soguk ilk gun farki
    if tekil:
        o_idx = olu.set_index("anahtar")
        s_idx = soguk.set_index("anahtar")
        fark = s_idx.loc[tekil, "ilk"].to_numpy() - o_idx.loc[tekil, "son"].to_numpy()
        fark_gun = pd.Series(fark).dt.days
        print("\nTEKIL eslesmelerde (soguk_ilk - olu_son) gun:")
        print(fark_gun.describe())

    olu.to_parquet(f"{KOK}/data/interim/h2_olu_havuz.parquet", index=False)
    soguk.to_parquet(f"{KOK}/data/interim/h2_soguk_ozet.parquet", index=False)
    o_tr.to_parquet(f"{KOK}/data/interim/h2_train_ozet.parquet", index=False)
    print("\nyazildi: data/interim/h2_*.parquet")


if __name__ == "__main__":
    main()
