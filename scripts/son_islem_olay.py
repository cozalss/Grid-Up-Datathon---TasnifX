"""OLAY GUNU SON ISLEMI -- ilk gun / donus gunu / son gun kismi-gun duzeltmesi.

BULGU (scripts/eksen2_kesin.py, panel gun etkisi cikarilmis, yerel referansli)
-----------------------------------------------------------------------------
Bir trafonun kaydinin BASLADIGI, KESILDIGI veya YENIDEN BASLADIGI gun KISMI bir
gundur ve log-tuketim o gun sistematik olarak dusuktur. Train'de olculdu:

    olay                          dusus    n     ikiz pencere (2025-04..07)
    dogum gun-0, parti <100      -0,604  2.256      -0,509
    dogum gun-0, parti 100+      -0,106    336      -0,111
    bosluk donus gunu            -0,529  1.698      -0,594
    son gun (canli trafo)        -1,203    638      -0,984

PARTI BUYUKLUGU BELIRLEYICI: ayni gun 100'den fazla trafo dogduysa dusus
neredeyse YOK (-0,11). Bu bir enerjilendirme dalgasi degil, veri setine TOPLU
KATILIM (geriye dolgu) -- olculen gun tamdir. 20-99 arasi partilerde dusus tam
(-0,52). Ayrim mevsimsel ikizde de ayakta: 100+ icin -0,111, <100 icin -0,509.

PANEL DUZELTMESI SART: yerel referans (sonraki 7 gun) panel gun etkisini
kaldirmaz. 2025-07-28'de (177 trafoluk parti) panel gun etkisi +0,4722 idi;
duzeltmesiz olcum "dusus yok" (-0,012) diyordu, duzeltilmis olcum -0,111.

v55 NE YAPIYOR: modelde ``yas``/``ilk_gun_mu`` kolonlari var ve YENI trafolarin
ilk gunune -0,337 uyguluyor -- parti buyuklugunu ayirt etmeden. Donus gunune ve
son gune HICBIR SEY uygulamiyor (-0,002 / +0,013).

    grup                          n     D_gercek  D_v55     kayma (s=0,6)
    YENI gun-0, parti 100+     1.634     -0,106  -0,337       +0,138
    YENI gun-0, parti <100       390     -0,604  -0,333       -0,163
    ESKI gun-0, bosluk 1-60g     557     -0,526  -0,021       -0,303
    ESKI gun-0, bosluk 60+g      534     -0,558  -0,002       -0,333
    IC BOSLUK donusu             745     -0,529  -0,001       -0,317
    SON gun (07-31 disi)         241     -1,203  +0,013       -0,729

4.081 satir (%0,571). Beklenen dMSE -0,00116 -> RMSLE 1,01591 -> 1,01534.

    python scripts/son_islem_olay.py --giris submissions/tuketim_v55_gunolcek.csv \
        --cikis submissions/tuketim_v56_olay.csv [--s 0.6]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
TE_SON = pd.Timestamp("2026-07-31")
S_VARSAYILAN = 0.6

#: (grup, D_gercek, D_v55) -- kaymanin tam kaynagi. eksen2_kesin.py ciktisi.
OLCUM = {
    "yeni_buyuk": (-0.1060, -0.3365),
    "yeni_kucuk": (-0.6042, -0.3334),
    "eski_kisa": (-0.5259, -0.0205),
    "eski_uzun": (-0.5576, -0.0020),
    "ic_bosluk": (-0.5289, -0.0011),
    "son_gun": (-1.2029, +0.0125),
}
#: Son gun dususu YALNIZCA canli trafolarda var: train'de canli -1,2029,
#: olu (onceki 14 gunun >=%50'si sifir) +0,0136. Olu trafolara dokunulmaz.
CANLI_ESIGI = 1.0


def main() -> int:
    a = argparse.ArgumentParser(description="olay gunu (kismi gun) duzeltmesi")
    a.add_argument("--giris", required=True)
    a.add_argument("--cikis", required=True)
    a.add_argument("--s", type=float, default=S_VARSAYILAN, help="buzme katsayisi")
    ar = a.parse_args()
    if not 0.0 <= ar.s <= 1.0:
        raise RuntimeError(f"s araligi disinda: {ar.s}")

    ornek = pd.read_csv(KOK / "data/raw/sample_submission.csv", encoding="utf-8")
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "guc", "tarih"],
        encoding="utf-8",
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    tr_son = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih"],
        encoding="utf-8",
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    tr_son = tr_son.groupby("tanim", observed=True)["tarih"].max()

    yol = Path(ar.giris)
    if not yol.is_absolute() and not yol.exists():
        yol = KOK / "submissions" / yol.name
    sub = pd.read_csv(yol, encoding="utf-8")
    if not sub["id"].equals(ornek["id"]):
        raise RuntimeError("id sirasi sample_submission ile ayni degil")

    m = sub.merge(te, on="id", how="left", validate="one_to_one")
    if len(m) != len(sub) or m["tanim"].isna().any():
        raise RuntimeError("birlestirme bozuldu")
    m = m.reset_index().rename(columns={"index": "_sira"})
    m = m.sort_values(["tanim", "tarih"], kind="mergesort")
    g = m.groupby("tanim", observed=True)
    m["ilk"] = g["tarih"].transform("min")
    m["son"] = g["tarih"].transform("max")
    m["yas"] = (m["tarih"] - m["ilk"]).dt.days
    m["kalan"] = (m["son"] - m["tarih"]).dt.days
    m["bosluk"] = ((m["tarih"] - g["tarih"].shift(1)).dt.days - 1.0).fillna(-1.0)
    m["tr_son"] = m["tanim"].map(tr_son)
    m["yeni"] = m["tr_son"].isna()
    m["gb"] = (m["tarih"] - m["tr_son"]).dt.days - 1.0  # yeni ise NaN

    parti = m.loc[m["yas"] == 0].groupby("ilk", observed=True).size()
    m["parti_n"] = m["ilk"].map(parti)
    # son gunde CANLI mi: son 14 gunun tahmin medyani
    canli = (
        m[m["kalan"].between(1, 14)].groupby("tanim", observed=True)["tuketim"].median()
        > CANLI_ESIGI
    )
    m["canli"] = m["tanim"].map(canli).fillna(True)

    ilk = m["yas"] == 0
    maskeler = {
        "yeni_buyuk": ilk & m["yeni"] & (m["parti_n"] >= 100),
        "yeni_kucuk": ilk & m["yeni"] & (m["parti_n"] < 100),
        "eski_kisa": ilk & (~m["yeni"]) & m["gb"].between(1, 60),
        "eski_uzun": ilk & (~m["yeni"]) & (m["gb"] > 60),
        "ic_bosluk": m["bosluk"] > 0,
        "son_gun": (m["kalan"] == 0) & (m["son"] < TE_SON) & m["canli"],
    }
    # Bir satir birden fazla gruba girebilir (ornek: tek gunluk kayit -- hem ilk
    # hem son gun). O gun TEK bir kismi olcumdur, iki dusus TOPLANMAZ; en buyuk
    # genlikli tek kayma uygulanir. Testte boyle 20 satir var.
    kayma = pd.Series(0.0, index=m.index)
    print(f"{'grup':<14} {'satir':>7} {'D_ger':>8} {'D_v55':>8} {'kayma':>9}")
    for ad, msk in maskeler.items():
        dg, dv = OLCUM[ad]
        k = ar.s * (dg - dv)
        cak = msk & (kayma.abs() > 0)
        kayma[msk & ~cak] = k
        kayma[cak] = np.where(np.abs(kayma[cak]) >= abs(k), kayma[cak], k)
        not_cak = f"  ({int(cak.sum())} cakisma, en buyuk genlik kazanir)" if cak.any() else ""
        print(f"{ad:<14} {int(msk.sum()):>7,} {dg:>+8.4f} {dv:>+8.4f} {k:>+9.4f}{not_cak}")
    dokunulan = int((kayma != 0).sum())
    print(f"{'TOPLAM':<14} {dokunulan:>7,}  ({dokunulan / len(m) * 100:.3f}% of test)")
    if float(kayma.abs().max()) > 1.0:
        raise RuntimeError(f"kayma cok buyuk: {float(kayma.abs().max()):.3f}")

    log_guc = np.log1p(m["guc"].to_numpy("float64"))
    r = np.log1p(m["tuketim"].to_numpy("float64")) - log_guc
    yeni_t = np.clip(np.expm1(r + kayma.to_numpy() + log_guc), 0.0, None)

    # ---- KAPILAR ----
    if np.isnan(yeni_t).any() or (yeni_t < 0).any():
        raise RuntimeError("NaN veya negatif tahmin")
    dok = kayma.to_numpy() == 0
    eski = m["tuketim"].to_numpy("float64")
    sap = float((np.abs(yeni_t[dok] - eski[dok]) / np.maximum(np.abs(eski[dok]), 1.0)).max())
    if sap > 1e-12:
        raise RuntimeError(f"dokunulmayan satir degisti: {sap:.3e}")
    if dokunulan > 0.01 * len(m):
        raise RuntimeError(f"cok fazla satira dokunuldu: {dokunulan:,}")
    gerc = np.log1p(yeni_t[~dok]) - np.log1p(eski[~dok])
    bek = kayma.to_numpy()[~dok]
    kirpma = float(np.abs(gerc - bek).max())
    print(
        f"  uygulanan kayma ile istenen arasindaki en buyuk fark: {kirpma:.2e} "
        f"(kirpilan satirlardan)"
    )
    print(f"  dokunulmayan {int(dok.sum()):,} satir aynen korundu (sapma {sap:.1e})")
    print(
        f"  min {yeni_t.min():.1f}  medyan {float(np.median(yeni_t)):.1f}  maks {yeni_t.max():.1f}"
    )
    print(f"  toplam log1p ortalamasi {np.log1p(eski).mean():.6f} -> {np.log1p(yeni_t).mean():.6f}")

    m["_yeni"] = yeni_t
    cik_df = m.sort_values("_sira")[["id", "_yeni"]].rename(columns={"_yeni": "tuketim"})
    if not cik_df["id"].reset_index(drop=True).equals(ornek["id"]):
        raise RuntimeError("cikis id sirasi bozuldu")
    cik = Path(ar.cikis)
    if not cik.is_absolute():
        cik = KOK / "submissions" / cik.name
    cik_df.to_csv(cik, index=False)
    print(f"  yazildi: {cik}  ({len(cik_df):,} satir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
