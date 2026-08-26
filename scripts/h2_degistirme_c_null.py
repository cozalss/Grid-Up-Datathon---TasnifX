"""H2-C: DOGRU null + soguk dogum yapisinin teshisi.

B'deki permutasyon bozuktu: soguk trafolarin anahtar etiketlerini kendi
aralarinda karistirmak anahtar COKLUGUNU degistirmiyor, o yuzden sayilar
birebir ayni cikti (sd=0). Dogru null: soguk olmayan (SICAK) test trafolarindan
ayni buyuklukte ornek cek ve ayni kurallarla kac eslesme cikiyor say. Soguk
olmak, olu bir trafoyla ayni hucrede olma olasiligini ARTIRIYOR mu?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KOK = "C:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
RNG = np.random.default_rng(20260825)


def main() -> None:
    olu = pd.read_parquet(f"{KOK}/data/interim/h2_olu_havuz.parquet")
    soguk = pd.read_parquet(f"{KOK}/data/interim/h2_soguk_ozet.parquet")
    o_tr = pd.read_parquet(f"{KOK}/data/interim/h2_train_ozet.parquet")

    te = pd.read_csv(f"{KOK}/data/raw/test.csv", parse_dates=["tarih"])
    g = te.groupby("tanim")
    o_te = pd.DataFrame({"ilk": g["tarih"].min(), "son": g["tarih"].max(), "n": g["tarih"].size()})
    o_te["guc"] = g["guc"].median()
    o_te["lokasyon"] = g["lokasyon"].agg(lambda s: s.mode().iloc[0])
    o_te = o_te.reset_index()
    o_te["anahtar"] = o_te["lokasyon"] + "|" + o_te["guc"].astype(int).astype(str)
    soguk_set = set(soguk["tanim"])
    o_te["soguk"] = o_te["tanim"].isin(soguk_set)

    print("=== SOGUK DOGUM YAPISI ===")
    print("en sik ilk gunler (soguk):")
    print(soguk["ilk"].value_counts().head(10).to_string())
    print("\nen sik ilk gunler (SICAK test trafolari):")
    print(o_te.loc[~o_te["soguk"], "ilk"].value_counts().head(5).to_string())
    print("\nsoguk: (ilk gun, son gun, n) en sik uclu:")
    print(
        soguk.assign(
            u=soguk["ilk"].dt.strftime("%m-%d")
            + "_"
            + soguk["son"].dt.strftime("%m-%d")
            + "_n"
            + soguk["n"].astype(str)
        )["u"]
        .value_counts()
        .head(8)
        .to_string()
    )

    print("\n=== DOGRU NULL: soguk vs sicak, olu-hucre komsulugu ===")
    olu_say = olu["anahtar"].value_counts()
    o_te["aday_olu"] = olu_say.reindex(o_te["anahtar"]).fillna(0).astype(int).to_numpy()
    s = o_te[o_te["soguk"]]
    h = o_te[~o_te["soguk"]]
    print(
        f"SOGUK  n={len(s)}  en az 1 olu adayi: {(s['aday_olu'] >= 1).mean():.4f}  ort aday {s['aday_olu'].mean():.3f}"
    )
    print(
        f"SICAK  n={len(h)}  en az 1 olu adayi: {(h['aday_olu'] >= 1).mean():.4f}  ort aday {h['aday_olu'].mean():.3f}"
    )

    # sicak'tan 2024 kisilik ornekler cekerek null dagilimi
    n_null = 400
    pay = np.zeros(n_null)
    for i in range(n_null):
        idx = RNG.choice(len(h), size=len(s), replace=False)
        pay[i] = (h["aday_olu"].to_numpy()[idx] >= 1).mean()
    gercek = (s["aday_olu"] >= 1).mean()
    print(
        f"null(sicak ornek) ort {pay.mean():.4f} sd {pay.std():.4f} -> z = {(gercek - pay.mean()) / pay.std():+.2f}"
    )

    # kVA ve lokasyon marjinalleri esitlenmis null (tabakali)
    print("\n=== TABAKALI NULL (ayni kVA dagilimi) ===")
    s_kva = s["guc"].value_counts(normalize=True)
    print("soguk kVA dagilimi:")
    print(s_kva.head(8).to_string())
    print("sicak kVA dagilimi:")
    print(h["guc"].value_counts(normalize=True).head(8).to_string())

    print("\n=== 1-1 ESLESME KALITESI: tekil 7 cift ===")
    s_say = s["anahtar"].value_counts()
    ortak = olu_say.index.intersection(s_say.index)
    tekil = [k for k in ortak if olu_say[k] == 1 and s_say[k] == 1]
    o_i = olu.set_index("anahtar")
    s_i = s.set_index("anahtar")
    for k in tekil:
        print(
            f"  {k:52s} olu_son={o_i.loc[k, 'son'].date()} olu_ort={o_i.loc[k, 'ort_tuketim']:9.1f} "
            f"soguk_ilk={s_i.loc[k, 'ilk'].date()} fark={(s_i.loc[k, 'ilk'] - o_i.loc[k, 'son']).days}g"
        )


if __name__ == "__main__":
    main()
