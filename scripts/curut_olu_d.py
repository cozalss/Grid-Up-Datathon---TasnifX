"""CURUTUCU D -- BASKA KESMELER. Model gerektirmeyen yapisal olcum.

Kova/grup sabitine cekme kuralinin tek dayanagi, olu-kuyruklu satirlarin
hedef seviyesinin KESMEDEN KESMEYE TASINMASIDIR. Burada bu dogrudan olculuyor:
sekiz farkli kesme, her biri icin ileri 122 gun, kova bazinda optimal sabit
ve sabitin capraz kesme TASIMA CEZASI.

Ayrica testin kendi kesmesinde (2026-03-31) etkilenen satir payi p cikarilir --
dMSE hesabinin paydasi.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

KESMELER = (
    "2025-03-31",  # yaz25 kesmesi (mevsimsel ikiz)
    "2025-04-30",  # YENI
    "2025-05-31",  # YENI
    "2025-06-30",  # YENI
    "2025-07-31",  # guz25 kesmesi
    "2025-09-30",  # YENI
    "2025-10-31",  # YENI
    "2025-11-30",  # kis26 kesmesi
)
UFUK = 122
KOVA_SINIR = [15, 30, 60, 90]
KOVA_ETIKET = ["1-14", "15-29", "30-59", "60-89", "90+"]


def kuyruk_tablosu(tr: pd.DataFrame, kesme: pd.Timestamp) -> pd.DataFrame:
    g = tr[tr["tarih"] <= kesme].sort_values(["tanim", "tarih"])
    sf = (g["tuketim"].to_numpy() <= 0).astype(np.int8)
    g = g.assign(_s=sf)
    out = []
    for ad, s in g.groupby("tanim", observed=True, sort=False):
        a = s["_s"].to_numpy()
        poz = np.flatnonzero(a == 0)
        kuy = len(a) if poz.size == 0 else len(a) - poz[-1] - 1
        out.append((ad, kuy, len(a)))
    return pd.DataFrame(out, columns=["tanim", "kuyruk", "gun"]).set_index("tanim")


def main() -> int:
    t0 = time.time()
    tr = pd.read_csv(KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str})
    tr["tarih"] = pd.to_datetime(tr["tarih"])
    tr["_ly"] = np.log1p(tr["tuketim"].clip(lower=0.0))

    kayit = {}
    print("=" * 104)
    print("D1) SEKIZ KESME -- olu-kuyruklu (kuyruk>=1) satirlarin ileri 122 gunu")
    print("=" * 104)
    print(
        f"  {'kesme':12}{'olu trafo':>11}{'ileri satir':>13}{'dirilme %':>11}"
        f"{'optimal c*':>12}{'c* MSE':>10}{'en buyuk trafo payi %':>23}"
    )
    for k in KESMELER:
        kt = pd.Timestamp(k)
        son = kt + pd.Timedelta(days=UFUK)
        tab = kuyruk_tablosu(tr, kt)
        olu = tab[tab["kuyruk"] >= 1]
        ileri = tr[(tr["tarih"] > kt) & (tr["tarih"] <= son)]
        ileri = ileri[ileri["tanim"].isin(olu.index)]
        if ileri.empty:
            continue
        ly = ileri["_ly"].to_numpy()
        c = float(ly.mean())
        mse = float(((ly - c) ** 2).mean())
        kuy = ileri["tanim"].map(olu["kuyruk"]).to_numpy()
        kov = np.digitize(kuy, KOVA_SINIR)
        # en buyuk tek trafonun toplam kare hataya payi
        kare = (ly - c) ** 2
        pay = pd.Series(kare).groupby(ileri["tanim"].to_numpy()).sum().max() / kare.sum()
        kayit[k] = {"ly": ly, "kov": kov, "tanim": ileri["tanim"].to_numpy(), "c": c}
        print(
            f"  {k:12}{len(olu):11,}{len(ileri):13,}{100 * (ly > 0).mean():11.2f}"
            f"{c:12.4f}{mse:10.4f}{100 * pay:23.2f}"
        )

    print()
    print("=" * 104)
    print("D2) KOVA BAZINDA OPTIMAL SABIT -- kesmeden kesmeye TASINIYOR MU?")
    print("=" * 104)
    print(f"  {'kesme':12}" + "".join(f"{e:>12}" for e in KOVA_ETIKET))
    kova_c = {}
    for k, v in kayit.items():
        satir = f"  {k:12}"
        kova_c[k] = {}
        for i in range(5):
            s = v["kov"] == i
            if s.sum() < 100:
                satir += f"{'-':>12}"
                kova_c[k][i] = np.nan
            else:
                c = float(v["ly"][s].mean())
                kova_c[k][i] = c
                satir += f"{c:12.4f}"
        print(satir)
    print("\n  kova bazinda kesmeler arasi std / ortalama:")
    for i in range(5):
        vals = np.array([kova_c[k][i] for k in kayit if not np.isnan(kova_c[k][i])])
        if vals.size >= 3:
            print(
                f"    {KOVA_ETIKET[i]:>8}: n={vals.size}  ort {vals.mean():.4f}  "
                f"std {vals.std(ddof=1):.4f}  min {vals.min():.4f}  max {vals.max():.4f}"
            )

    print()
    print("=" * 104)
    print("D3) CAPRAZ TASIMA CEZASI -- baska kesmenin sabitini kullanmanin MSE artisi")
    print("=" * 104)
    ks = list(kayit)
    print(
        f"  {'hedef kesme':14}{'kendi c* MSE':>14}{'digerlerinin c* ile ort MSE':>30}"
        f"{'ceza':>10}{'en kotu ceza':>14}"
    )
    for k in ks:
        v = kayit[k]
        ly = v["ly"]
        oz = float(((ly - v["c"]) ** 2).mean())
        cez = []
        for j in ks:
            if j == k:
                continue
            cez.append(float(((ly - kayit[j]["c"]) ** 2).mean()) - oz)
        print(f"  {k:14}{oz:14.4f}{oz + np.mean(cez):30.4f}{np.mean(cez):+10.4f}{max(cez):+14.4f}")

    print()
    print("=" * 104)
    print("D4) TESTIN KENDI KESMESI (2026-03-31) -- etkilenen satir payi p")
    print("=" * 104)
    kt = pd.Timestamp("2026-03-31")
    tab = kuyruk_tablosu(tr, kt)
    olu = tab[tab["kuyruk"] >= 1]
    te = pd.read_csv(
        KOK / "data/raw/test.csv", usecols=["id", "tanim"], encoding="utf-8", dtype={"tanim": str}
    )
    etkilenen = te["tanim"].isin(olu.index)
    print(f"  egitim sonunda kuyruk>=1 trafo   {len(olu):,}")
    print(
        f"  bunlarin testte gorunen satiri   {int(etkilenen.sum()):,} / {len(te):,}"
        f"  = p {etkilenen.mean():.5f}"
    )
    kuy = te.loc[etkilenen, "tanim"].map(olu["kuyruk"]).to_numpy()
    kov = np.digitize(kuy, KOVA_SINIR)
    for i in range(5):
        s = kov == i
        if s.sum():
            print(
                f"    kova {KOVA_ETIKET[i]:>8}: {int(s.sum()):7,} satir  "
                f"{te.loc[etkilenen, 'tanim'][s].nunique():4} trafo"
            )

    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
