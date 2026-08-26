"""H2-B: Degistirme sinyali GERCEK mi, yoksa hucre kalabaligindan mi?

A adiminda (lokasyon,kVA) anahtarinda yalnizca 7 tekil 1-1 eslesme cikti ve
o 7'sinde bile olum-dogum farki medyan 248 gun. Burada iki sey olculuyor:

1. OLUM ve DOGUM zamanlamasi: gercek degistirmede olum, yeni birimin dogumuna
   YAKIN olmali (gunler-haftalar). Dagilimlari cikar.
2. PERMUTASYON NULLU: soguk trafolari hucreler arasi karistirip ayni kurallarla
   kac "eslesme" cikiyor say. Gercek eslesme sayisi nulldan farkli degilse
   kural sifir bilgi tasiyor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KOK = "C:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
TRAIN_SON = pd.Timestamp("2026-03-31")
RNG = np.random.default_rng(20260825)


def main() -> None:
    olu = pd.read_parquet(f"{KOK}/data/interim/h2_olu_havuz.parquet")
    soguk = pd.read_parquet(f"{KOK}/data/interim/h2_soguk_ozet.parquet")
    o_tr = pd.read_parquet(f"{KOK}/data/interim/h2_train_ozet.parquet")

    print("=== 1. OLUM ZAMANLAMASI (train-only 332 trafo) ===")
    yalniz = o_tr[
        ~o_tr["tanim"].isin(
            set(pd.read_parquet(f"{KOK}/data/interim/h2_soguk_ozet.parquet")["tanim"])
        )
    ]
    tr_only = o_tr[o_tr["son"] < TRAIN_SON].copy()
    print("train-only olum ayi:")
    print(olu["son"].dt.to_period("M").value_counts().sort_index().to_string())

    print("\n=== 2. SOGUK DOGUM ZAMANLAMASI ===")
    print(soguk["ilk"].dt.to_period("M").value_counts().sort_index().to_string())
    print("\n2026-05-01'de ilk gorulen soguk trafo:", int((soguk["ilk"] == "2026-05-01").sum()))
    print("test'te gun sayisi dagilimi (soguk):")
    print(soguk["n"].describe().to_string())

    print("\n=== 3. GERCEK vs PERMUTASYON eslesme sayisi ===")
    olu_say = olu["anahtar"].value_counts()
    gercek_tekil = 0
    gercek_kapsanan = 0
    s_say = soguk["anahtar"].value_counts()
    ortak = olu_say.index.intersection(s_say.index)
    gercek_tekil = int(sum(1 for k in ortak if olu_say[k] == 1 and s_say[k] == 1))
    gercek_kapsanan = int(s_say.reindex(ortak).sum())

    # NULL: soguk trafolarin anahtarlarini karistir (kVA+lokasyon marjinallerini
    # koru: anahtar etiketlerini permute et)
    n_null = 200
    tekil_null = np.zeros(n_null, dtype=int)
    kaps_null = np.zeros(n_null, dtype=int)
    anahtarlar = soguk["anahtar"].to_numpy()
    for i in range(n_null):
        p = RNG.permutation(anahtarlar)
        s2 = pd.Series(p).value_counts()
        ort2 = olu_say.index.intersection(s2.index)
        tekil_null[i] = sum(1 for k in ort2 if olu_say[k] == 1 and s2[k] == 1)
        kaps_null[i] = int(s2.reindex(ort2).sum())
    print(
        f"tekil 1-1 : gercek {gercek_tekil}  null ort {tekil_null.mean():.1f} sd {tekil_null.std():.1f}"
    )
    print(
        f"kapsanan  : gercek {gercek_kapsanan}  null ort {kaps_null.mean():.1f} sd {kaps_null.std():.1f}"
    )
    if tekil_null.std() > 0:
        print(f"  z(tekil) = {(gercek_tekil - tekil_null.mean()) / tekil_null.std():+.2f}")
    if kaps_null.std() > 0:
        print(f"  z(kapsanan) = {(gercek_kapsanan - kaps_null.mean()) / kaps_null.std():+.2f}")

    print("\n=== 4. YANLIS POZITIF vekili: hucre kalabaligi ===")
    # Bir soguk trafo icin ayni (lokasyon,kVA) hucresinde kac olu aday var?
    soguk["aday_olu"] = olu_say.reindex(soguk["anahtar"]).fillna(0).astype(int).to_numpy()
    print(soguk["aday_olu"].value_counts().sort_index().to_string())
    en_az_bir = int((soguk["aday_olu"] >= 1).sum())
    tam_bir = int((soguk["aday_olu"] == 1).sum())
    print(f"en az 1 aday olusu olan soguk: {en_az_bir} / {len(soguk)}")
    print(f"tam 1 aday olusu olan soguk : {tam_bir}")

    print("\n=== 5. ZAMANLAMA KISITI EKLE (olum, dogumdan onceki 0-120 gun) ===")
    o_g = olu.groupby("anahtar")["son"].apply(list).to_dict()
    hit0, hit60, hit120 = 0, 0, 0
    farklar = []
    for anah, dogum in zip(soguk["anahtar"], soguk["ilk"]):
        for son in o_g.get(anah, []):
            d = (dogum - son).days
            farklar.append(d)
            if 0 <= d <= 30:
                hit0 += 1
            if 0 <= d <= 60:
                hit60 += 1
            if 0 <= d <= 120:
                hit120 += 1
    farklar = np.array(farklar)
    print(f"toplam (soguk,olu) aday cifti: {len(farklar)}")
    print(f"  fark 0-30 gun : {hit0}")
    print(f"  fark 0-60 gun : {hit60}")
    print(f"  fark 0-120 gun: {hit120}")
    if len(farklar):
        print("fark (gun) dagilimi:", np.percentile(farklar, [5, 25, 50, 75, 95]).round(0))


if __name__ == "__main__":
    main()
