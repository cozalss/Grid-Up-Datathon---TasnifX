"""H2-E: KIRPMA TABLOSU + coklu blok isaret tutarliligi + dMSE cevrimi.

D adiminda yaz25 ikizinde olu esin artik betasi -0,0403 (isaret TERS) cikti.
Burada kanit citasi gerekleri tamamlaniyor:
  * kural 1: trafo bazli ayrisim + KIRPMA TABLOSU K=0,1,5,10,25,50
  * kural 7/9: uc ortusmeyen dogum penceresinde (yaz25/guz25/kis26) isaret
  * beta ORNEK DISI kalibre edilir (guz25'te fit, yaz25'te uygulanir) --
    ici-ornek beta ile olculen kazanc uydurmadir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KOK = "C:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
HEDEF_PENCERE = 60
P_SOGUK = 0.22159
BLOKLAR = {
    "yaz25": ("2025-04-01", "2025-07-31"),
    "guz25": ("2025-08-01", "2025-11-30"),
    "kis26": ("2025-12-01", "2026-03-31"),
}


def blok_kur(tr: pd.DataFrame, ozet: pd.DataFrame, bas: str, son: str) -> pd.DataFrame:
    bas, son = pd.Timestamp(bas), pd.Timestamp(son)
    dogum = ozet[(ozet["ilk"] >= bas) & (ozet["ilk"] <= son) & (ozet["n"] >= 20)].copy()
    oz_i = ozet.set_index("tanim")
    satir = []
    for t, ilk, anah in zip(dogum["tanim"], dogum["ilk"], dogum["anahtar"]):
        kend = tr[tr["tanim"] == t]
        y = float(kend.loc[kend["tarih"] <= ilk + pd.Timedelta(days=HEDEF_PENCERE), "ll"].mean())
        onceki = tr[(tr["anahtar"] == anah) & (tr["tarih"] < ilk) & (tr["tanim"] != t)]
        if len(onceki) == 0 or not np.isfinite(y):
            continue
        taban = float(onceki["ll"].mean())
        son_kayit = onceki.groupby("tanim")["tarih"].max()
        adaylar = son_kayit[son_kayit < ilk - pd.Timedelta(days=7)]
        gercek_olu = [a for a in adaylar.index if oz_i.loc[a, "son"] < ilk]
        if not gercek_olu:
            continue
        yakin = son_kayit.loc[gercek_olu].idxmax()
        sv = float(onceki.loc[onceki["tanim"] == yakin, "ll"].mean())
        satir.append(
            {
                "tanim": t,
                "y": y,
                "taban": taban,
                "olu_sev": sv,
                "n_olu": len(gercek_olu),
                "olu_yas": int((ilk - son_kayit.loc[yakin]).days),
            }
        )
    return pd.DataFrame(satir)


def istatistik(d: pd.DataFrame, ad: str) -> float:
    rx = (d["olu_sev"] - d["taban"]).to_numpy()
    ry = (d["y"] - d["taban"]).to_numpy()
    if len(d) < 6 or rx.std() == 0:
        print(f"  {ad}: ornek yetersiz n={len(d)}")
        return np.nan
    beta = float(np.cov(rx, ry)[0, 1] / np.var(rx))
    r = float(np.corrcoef(rx, ry)[0, 1])
    n = len(d)
    se = np.sqrt((1 - r**2) / (n - 2))
    print(f"  {ad}: n={n:4d}  beta={beta:+.4f}  r={r:+.4f}  R2={r**2:.4f}  t={r / se:+.2f}")
    return beta


def main() -> None:
    tr = pd.read_csv(f"{KOK}/data/raw/train.csv", parse_dates=["tarih"])
    tr["ll"] = np.log1p(tr["tuketim"].clip(lower=0))
    g = tr.groupby("tanim")
    ozet = pd.DataFrame({"ilk": g["tarih"].min(), "son": g["tarih"].max(), "n": g["tarih"].size()})
    ozet["guc"] = g["guc"].median()
    ozet["lokasyon"] = g["lokasyon"].agg(lambda s: s.mode().iloc[0])
    ozet = ozet.reset_index()
    ozet["anahtar"] = ozet["lokasyon"] + "|" + ozet["guc"].astype(int).astype(str)
    tr = tr.merge(ozet[["tanim", "anahtar"]], on="tanim", how="left")

    veri = {}
    print("=== BLOK BAZLI ARTIK REGRESYONU (olu es, hucre ortalamasinin ustune) ===")
    betalar = {}
    for ad, (b, s) in BLOKLAR.items():
        d = blok_kur(tr, ozet, b, s)
        veri[ad] = d
        betalar[ad] = istatistik(d, ad)

    print("\n=== ORNEK DISI TRANSFER: guz25'te fit -> yaz25'te uygula ===")
    yaz = veri["yaz25"]
    b_dis = betalar["guz25"]
    if not np.isfinite(b_dis) or yaz.empty:
        print("  transfer olculemedi")
        return
    rx = (yaz["olu_sev"] - yaz["taban"]).to_numpy()
    ry = (yaz["y"] - yaz["taban"]).to_numpy()
    hata_taban = ry**2
    hata_aday = (ry - b_dis * rx) ** 2
    kazanc = hata_taban - hata_aday  # pozitif = iyilesme
    print(f"  guz25 beta = {b_dis:+.4f}  yaz25 n={len(yaz)}")
    print(f"  taban MSE(log seviye) = {hata_taban.mean():.4f}")
    print(f"  aday  MSE(log seviye) = {hata_aday.mean():.4f}")
    print(f"  dMSE(seviye) = {-kazanc.mean():+.4f}   (negatif = kazanc)")

    print("\n=== KIRPMA TABLOSU (yaz25, ornek disi beta) ===")
    print("  K    kalan_n   dMSE_seviye   dMSE_test_vekili")
    sirali = np.sort(kazanc)[::-1]  # en buyuk katki basta
    n = len(kazanc)
    for K in (0, 1, 5, 10, 25, 50):
        if n <= K:
            print(f"  {K:3d}  ornek tukendi")
            continue
        kalan = sirali[K:]
        d_sev = -float(kalan.mean())
        print(f"  {K:3d}  {len(kalan):6d}   {d_sev:+.5f}      {P_SOGUK * d_sev:+.6f}")

    print("\n=== ICI-ORNEK beta ile ayni tablo (uydurma tavan, referans) ===")
    b_ici = betalar["yaz25"]
    kaz_ici = ry**2 - (ry - b_ici * rx) ** 2
    s_ici = np.sort(kaz_ici)[::-1]
    for K in (0, 1, 5, 10, 25, 50):
        if n <= K:
            continue
        kalan = s_ici[K:]
        print(
            f"  {K:3d}  {len(kalan):6d}   {-float(kalan.mean()):+.5f}      {P_SOGUK * -float(kalan.mean()):+.6f}"
        )

    print("\n=== YANLIS POZITIF: hucrede kac olu aday vardi ===")
    for ad, d in veri.items():
        if len(d):
            print(
                f"  {ad}: n_olu medyan {d['n_olu'].median():.0f}  tekil orani {(d['n_olu'] == 1).mean():.3f}  "
                f"olu_yas medyan {d['olu_yas'].median():.0f}g"
            )

    print("\n=== TEKIL (n_olu==1) ALT KUMESI, blok bazli ===")
    for ad, d in veri.items():
        if len(d):
            istatistik(d[d["n_olu"] == 1], ad + "/tekil")

    print("\n=== YAKIN OLUM (olu_yas <= 60 gun) alt kumesi ===")
    for ad, d in veri.items():
        if len(d):
            alt = d[d["olu_yas"] <= 60]
            istatistik(alt, ad + "/yakin60")


if __name__ == "__main__":
    main()
