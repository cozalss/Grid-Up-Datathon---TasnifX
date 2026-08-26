"""H2-D: IKIZDE DOGRULAMA (yaz25) -- olu esin seviyesi yeni trafoyu aciklar mi?

Kurulum (kural 8, AS-OF; kural 7, yaz25 zorunlu):
  * DOGUM: train icinde ilk kaydi 2025-04-01..07-31 arasinda olan trafolar.
    Bunlar soguk trafolarin ikizi -- gecmisleri yok, seviyeleri tahmin edilmeli.
  * OLU ES ADAYI: son kaydi dogum gununden ONCE biten trafolar (kesin as-of;
    pencere hedefin dogumundan once bitiyor).
  * ANAHTAR: (lokasyon, guc). `tanim` string yapisi KULLANILMIYOR.
  * HEDEF: yeni trafonun dogumdan sonraki ilk 60 gunundeki ort log1p(tuketim).

KARSILASTIRMA:
  taban  = hucre (lokasyon,kVA) ortalama log seviyesi -- as-of, yalnizca dogum
           gununden once kaydi olan trafolardan. Model bunu ZATEN biliyor
           (lokasyon + guc oznitelikleri).
  aday   = taban + eslesen OLU esin gecmis log seviyesi.
Olu es hucre ortalamasinin USTUNE bilgi katmiyorsa degistirme hipotezi olur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KOK = "C:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
YAZ_BAS = pd.Timestamp("2025-04-01")
YAZ_SON = pd.Timestamp("2025-07-31")
HEDEF_PENCERE = 60  # gun


def main() -> None:
    tr = pd.read_csv(f"{KOK}/data/raw/train.csv", parse_dates=["tarih"])
    tr["ll"] = np.log1p(tr["tuketim"].clip(lower=0))

    g = tr.groupby("tanim")
    ozet = pd.DataFrame({"ilk": g["tarih"].min(), "son": g["tarih"].max(), "n": g["tarih"].size()})
    ozet["guc"] = g["guc"].median()
    ozet["lokasyon"] = g["lokasyon"].agg(lambda s: s.mode().iloc[0])
    ozet = ozet.reset_index()
    ozet["anahtar"] = ozet["lokasyon"] + "|" + ozet["guc"].astype(int).astype(str)

    print("train ilk-gun dagilimi (ay):")
    print(ozet["ilk"].dt.to_period("M").value_counts().sort_index().to_string())

    dogum = ozet[(ozet["ilk"] >= YAZ_BAS) & (ozet["ilk"] <= YAZ_SON)].copy()
    dogum = dogum[dogum["n"] >= 20]
    print(f"\nyaz25 DOGUMLU trafo (>=20 kayit): {len(dogum)}")
    if len(dogum) < 30:
        print("!! ikiz orneklemi cok kucuk -- hukum verilemez")

    # hedef: dogumdan sonraki ilk 60 gun ort log seviyesi
    tr_idx = tr.set_index("tanim")
    hedefler = {}
    for t, ilk in zip(dogum["tanim"], dogum["ilk"]):
        s = tr_idx.loc[[t]]
        m = s["tarih"] <= ilk + pd.Timedelta(days=HEDEF_PENCERE)
        hedefler[t] = float(s.loc[m, "ll"].mean())
    dogum["y"] = dogum["tanim"].map(hedefler)
    dogum = dogum[np.isfinite(dogum["y"])]
    print(
        f"hedefi hesaplanan: {len(dogum)}  y ort {dogum['y'].mean():.3f} sd {dogum['y'].std():.3f}"
    )

    # --- AS-OF hucre ortalamasi ve olu es ---
    # her trafo icin: dogumdan ONCEKI kayitlarindan log seviyesi
    kayitlar = tr[["tanim", "tarih", "ll", "anahtar"]] if "anahtar" in tr.columns else None
    tr = tr.merge(ozet[["tanim", "anahtar"]], on="tanim", how="left")

    taban, olu_sev, olu_bilgi, olu_yas = [], [], [], []
    n_olu_var = 0
    for t, ilk, anah in zip(dogum["tanim"], dogum["ilk"], dogum["anahtar"]):
        # AS-OF: yalnizca dogum gununden ONCE biten kayitlar
        onceki = tr[(tr["anahtar"] == anah) & (tr["tarih"] < ilk) & (tr["tanim"] != t)]
        if len(onceki) == 0:
            taban.append(np.nan)
            olu_sev.append(np.nan)
            olu_bilgi.append(0)
            olu_yas.append(np.nan)
            continue
        taban.append(float(onceki["ll"].mean()))
        # olu es: bu hucrede son kaydi dogumdan ONCE biten trafolar
        son_kayit = onceki.groupby("tanim")["tarih"].max()
        adaylar = son_kayit[son_kayit < ilk - pd.Timedelta(days=7)]
        # sadece GERCEKTEN olmus olanlar: genel son kaydi da dogumdan once
        gercek_olu = [a for a in adaylar.index if ozet.set_index("tanim").loc[a, "son"] < ilk]
        if not gercek_olu:
            olu_sev.append(np.nan)
            olu_bilgi.append(0)
            olu_yas.append(np.nan)
            continue
        n_olu_var += 1
        # en YAKIN olen esi sec
        yakin = son_kayit.loc[gercek_olu].idxmax()
        sv = onceki.loc[onceki["tanim"] == yakin, "ll"].mean()
        olu_sev.append(float(sv))
        olu_bilgi.append(len(gercek_olu))
        olu_yas.append(int((ilk - son_kayit.loc[yakin]).days))

    dogum["taban"] = taban
    dogum["olu_sev"] = olu_sev
    dogum["n_olu"] = olu_bilgi
    dogum["olu_yas"] = olu_yas

    d = dogum.dropna(subset=["taban", "y"]).copy()
    print(f"\ntabani olan: {len(d)}   olu esi olan: {int(d['olu_sev'].notna().sum())}")
    print(f"olu esi TEKIL (n_olu==1) olan: {int((d['n_olu'] == 1).sum())}")

    def rmse(a, b):
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    print("\n=== TUM DOGUMLAR ===")
    print(f"  taban RMSE (hucre ort)      : {rmse(d['y'], d['taban']):.4f}")
    print(f"  global ort RMSE             : {rmse(d['y'], np.full(len(d), d['y'].mean())):.4f}")

    e = d.dropna(subset=["olu_sev"]).copy()
    print(f"\n=== OLU ESI OLANLAR (n={len(e)}) ===")
    print(f"  taban RMSE                  : {rmse(e['y'], e['taban']):.4f}")
    print(f"  olu es seviyesi RMSE        : {rmse(e['y'], e['olu_sev']):.4f}")
    for w in (0.25, 0.5, 0.75):
        h = (1 - w) * e["taban"] + w * e["olu_sev"]
        print(f"  harman w_olu={w:.2f} RMSE     : {rmse(e['y'], h):.4f}")

    # ARTIK REGRESYONU: olu es, hucre ortalamasinin USTUNE ne katiyor?
    r_y = e["y"] - e["taban"]
    r_x = e["olu_sev"] - e["taban"]
    if len(e) > 5 and r_x.std() > 0:
        beta = float(np.cov(r_x, r_y)[0, 1] / np.var(r_x))
        rr = float(np.corrcoef(r_x, r_y)[0, 1])
        n = len(e)
        se = np.sqrt((1 - rr**2) / (n - 2)) if n > 2 else np.nan
        t = rr / se if se and np.isfinite(se) else np.nan
        print(
            f"\n  artik regresyon: beta={beta:+.4f}  r={rr:+.4f}  R2={rr**2:.4f}  t={t:+.2f}  n={n}"
        )
        en_iyi = rmse(e["y"], e["taban"] + beta * r_x)
        print(
            f"  optimal beta ile RMSE       : {en_iyi:.4f}  (taban {rmse(e['y'], e['taban']):.4f})"
        )

    # TEKIL eslesmelerde ayni olcum
    t1 = e[e["n_olu"] == 1]
    if len(t1) > 5:
        r_y1 = t1["y"] - t1["taban"]
        r_x1 = t1["olu_sev"] - t1["taban"]
        rr1 = float(np.corrcoef(r_x1, r_y1)[0, 1])
        print(
            f"\n  TEKIL es (n={len(t1)}): r={rr1:+.4f} R2={rr1**2:.4f}  "
            f"taban RMSE {rmse(t1['y'], t1['taban']):.4f} olu RMSE {rmse(t1['y'], t1['olu_sev']):.4f}"
        )

    d.to_parquet(f"{KOK}/data/interim/h2_ikiz_yaz25.parquet", index=False)
    print("\nyazildi: data/interim/h2_ikiz_yaz25.parquet")


if __name__ == "__main__":
    main()
