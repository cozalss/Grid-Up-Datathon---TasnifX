"""GRUP B SON ISLEMI -- raporu kesilip PANELE GERI DONEN olu trafolar.

BULGU (docs/43 §2)
------------------
Kuyrugu >=60 gun sifir olan 204 test trafosu TEK GRUP DEGIL. Panel onlari
kusursuz ikiye boluyor:

    A  train sonuna kadar rapor etti      145 trafo / 16.872 satir  -> OLU KALIYOR
    B  raporu kesildi, teste SONRADAN girdi 59 trafo /  4.727 satir  -> DIRILIYOR

B icin kanit uc bagimsiz yoldan geliyor:
  * train ici bosluk-donus taramasi: bosluk 200+ gun & bosluk oncesi SIFIR
    -> n=8 vaka, donus gunu sifir orani %0,0, ort log1p 5,685
  * ileri-pencere analogu (kesme 2025-07-31 / 09-30 / 11-30, "gecikme>30"):
    %100,0 / %100,0 / %100,0 sifir-olmayan; optimal log1p 6,313 / 7,103 / 7,922
  * uzun bosluktan donen CANLI trafolar (n=96): donus sonrasi seviye kendi
    onceki seviyesine esit (kayma -0,057), R^2 0,872; ILCE R^2 0,000

Yani "geri donus" bir DIRILME kanitidir ve trafo KENDI eski seviyesine doner.

NE YAPIYOR
----------
Her B trafosu icin hedef ofset seviyesi = olumden ONCEKI pozitif kayitlarin
ortalama ofseti (log1p(tuketim) - log1p(guc)), + yillik buyume. v55'in o
trafoya yazdigi ortalama ofsetle arasindaki fark, ``--s`` ile buzulup
trafonun BUTUN test satirlarina log uzayinda eklenir.

Yeterli pozitif kaydi olmayan trafolarda ilce x kVA kovasi ortalamasina
geri cekilir; o da yoksa trafo ATLANIR (islem yapilmaz).

KALICI KURAL 9: bu betik YALNIZCA grup B'ye dokunur. Grup A'ya (gercekten
olu, rapor devam eden 145 trafo) ASLA dokunmaz.

    python scripts/son_islem_grupb.py --giris submissions/tuketim_v55_gunolcek.csv \
        --cikis submissions/tuketim_v64_grupb.csv [--s 0.7]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
KUYRUK_ESIK = 60  # son N kaydin tamami sifir -> olu kuyruk
A_SINIRI = pd.Timestamp("2026-03-27")  # train son kaydi >= bu ise GRUP A
BUYUME = 0.0635  # docs/43 §4, hava-ayarli yillik buyume kuklasi
S_VARSAYILAN = 0.7


def yukle() -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str})
    tr["tarih"] = pd.to_datetime(tr["tarih"])
    te = pd.read_csv(KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str})
    te["tarih"] = pd.to_datetime(te["tarih"])
    for d in (tr, te):
        d["ilce"] = d["lokasyon"].str.split(">").str[-1]
    tr["ofs"] = np.log1p(tr["tuketim"].clip(lower=0)) - np.log1p(tr["guc"])
    return tr, te


def grup_b_bul(tr: pd.DataFrame, te: pd.DataFrame) -> pd.Index:
    """Kuyrugu >=KUYRUK_ESIK sifir VE train son kaydi A_SINIRI'ndan ONCE olanlar."""
    d = tr.sort_values(["tanim", "tarih"])
    son_n = d.groupby("tanim").tail(KUYRUK_ESIK)
    kuyruk_olu = son_n.groupby("tanim")["tuketim"].apply(lambda s: bool((s <= 0).all()))
    yeterli = d.groupby("tanim").size() >= KUYRUK_ESIK
    tr_son = d.groupby("tanim")["tarih"].max()
    b = kuyruk_olu & yeterli & (tr_son < A_SINIRI)
    return pd.Index(sorted(set(b.index[b]) & set(te["tanim"].unique())))


def hedef_seviye(tr: pd.DataFrame, trafolar: pd.Index) -> pd.Series:
    """Olumden ONCEKI pozitif kayitlarin ortalama ofseti."""
    poz = tr[(tr["tanim"].isin(trafolar)) & (tr["tuketim"] > 0)].sort_values(["tanim", "tarih"])
    return (
        poz.groupby("tanim")
        .tail(60)
        .groupby("tanim")
        .agg(seviye=("ofs", "mean"), n=("ofs", "size"))["seviye"]
        .where(poz.groupby("tanim").tail(60).groupby("tanim").size() >= 10)
    )


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--giris", required=True)
    a.add_argument("--cikis", required=True)
    a.add_argument("--s", type=float, default=S_VARSAYILAN, help="buzme (0=kapali, 1=tam)")
    a.add_argument("--buyume", type=float, default=BUYUME)
    ar = a.parse_args()

    tr, te = yukle()
    B = grup_b_bul(tr, te)
    print(f"GRUP B: {len(B)} trafo")

    sev = hedef_seviye(tr, B)
    # geri cekilme: ilce x kVA kovasi
    tr_c = tr[tr["tuketim"] > 0].copy()
    tr_c["kova"] = pd.cut(tr_c["guc"], [0, 100, 250, 400, 630, 1000, 1e9])
    kova_ort = tr_c.groupby(["ilce", "kova"], observed=True)["ofs"].mean()
    genel = tr_c["ofs"].mean()

    sub = pd.read_csv(ar.giris, encoding="utf-8")
    n0 = len(sub)
    m = te.merge(sub, on="id", how="left", validate="one_to_one")
    if len(m) != n0 or m["tuketim"].isna().any():
        raise RuntimeError(f"birlestirme bozuk: {n0} -> {len(m)}")
    m["lp"] = np.log1p(m["tuketim"].clip(lower=0))
    m["ofs"] = m["lp"] - np.log1p(m["guc"])
    m["kova"] = pd.cut(m["guc"], [0, 100, 250, 400, 630, 1000, 1e9])

    hedef = (
        m.loc[m["tanim"].isin(B)]
        .groupby("tanim")
        .agg(v55=("ofs", "mean"), n=("ofs", "size"), ilce=("ilce", "first"), kova=("kova", "first"))
    )
    hedef["kendi"] = hedef.index.map(sev)
    yedek = hedef.apply(lambda r: kova_ort.get((r["ilce"], r["kova"]), genel), axis=1)
    hedef["yedek_mi"] = hedef["kendi"].isna()
    hedef["seviye"] = hedef["kendi"].fillna(yedek) + ar.buyume
    hedef["kayma"] = ar.s * (hedef["seviye"] - hedef["v55"])

    print(
        f"  kendi gecmisinden seviye alan: {int((~hedef.yedek_mi).sum())}"
        f"  ilce x kVA yedegi: {int(hedef.yedek_mi.sum())}"
    )
    print(f"  v55 ort ofset  = {hedef.v55.mean():+.4f}")
    print(f"  hedef ort ofset= {hedef.seviye.mean():+.4f}")
    print(
        f"  kayma: ort={hedef.kayma.mean():+.4f} medyan={hedef.kayma.median():+.4f}"
        f" p10={hedef.kayma.quantile(0.1):+.3f} p90={hedef.kayma.quantile(0.9):+.3f}"
    )
    print(f"  etkilenen satir: {int(hedef.n.sum())} (%{100 * hedef.n.sum() / n0:.3f})")

    # ---- KAPILAR
    if len(B) == 0:
        raise RuntimeError("grup B bos -- tanim degismis olabilir")
    a_grup = tr.sort_values(["tanim", "tarih"]).groupby("tanim")["tarih"].max()
    cakisma = set(B) & set(a_grup.index[a_grup >= A_SINIRI])
    if cakisma:
        raise RuntimeError(f"KAPI: grup A ile cakisma var: {len(cakisma)} trafo")
    if hedef["kayma"].abs().max() > 5.0:
        raise RuntimeError(f"KAPI: asiri kayma {hedef['kayma'].abs().max():.2f}")
    if (hedef["kayma"] < -1.0).any():
        print(f"  UYARI: {int((hedef.kayma < -1).sum())} trafoda ASAGI kayma var")

    m["kayma"] = m["tanim"].map(hedef["kayma"]).fillna(0.0)
    yeni_lp = m["lp"] + m["kayma"]
    out = pd.DataFrame({"id": m["id"], "tuketim": np.expm1(yeni_lp).clip(lower=0.0)})

    degisen = int((m["kayma"] != 0).sum())
    dokunulmayan = m["kayma"] == 0
    sapma = float(
        np.abs(
            out.loc[dokunulmayan, "tuketim"].to_numpy() - m.loc[dokunulmayan, "tuketim"].to_numpy()
        ).max()
    )
    if sapma > 1e-9:
        raise RuntimeError(f"KAPI: dokunulmayan satirlar degismis (sapma {sapma:.2e})")
    if out["tuketim"].isna().any() or (out["tuketim"] < 0).any():
        raise RuntimeError("KAPI: cikti NaN ya da negatif")
    if len(out) != n0:
        raise RuntimeError("KAPI: satir sayisi degisti")

    out.to_csv(ar.cikis, index=False)
    print(f"  degisen satir {degisen} / {n0}   dokunulmayan sapma {sapma:.1e}")
    print(f"  toplam log1p ort {m['lp'].mean():.6f} -> {yeni_lp.mean():.6f}")
    print(f"  yazildi: {ar.cikis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
