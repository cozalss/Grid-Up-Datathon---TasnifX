"""OLCUMLE CURUTULMUS bileseni adaylardan cikarir + rejim capasini yeniler.

`docs/52` §1: "olu trafo" tezi LB'de CURUDU. Train'de kesilmis / hep sifir olan
trafolar (GRUP B) test penceresinde DIRIDIR; gercek log1p'leri 5,9-7,9.
Onlari ~0'a cekmek olculdu ve ZEHIRLI cikti (v87 1.14297, v88 1.11810,
sota_v1 tez-yanlis ucu 1.15740).

Klasik / amnezik / SOTA adaylarinin yon enerjisinin %62-65'i tam da bu
satirlardan geliyor -- yani yonun cogu ZATEN OLCULMUS ve YANLIS cikmis bir
hipotezi tekrar ediyor. Bu satirlari m6'ya esitlemek gurultu temizligi degil,
OLCUME dayali bir duzeltmedir; yonun acisini gercekten iyilestirir.

Sonra her rejimde (soguk / kuyruk / cekirdek) NOTR OLMAYAN satirlarin ortalama
farki sifirlanir -- seviye yalnizca LB'de olculur, aday SEKIL tasisin.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))
S = os.path.join(KOK, "submissions")

tr = pd.read_csv(
    os.path.join(KOK, "data/raw/train.csv"),
    parse_dates=["tarih"],
    dtype={"tanim": str},
    usecols=["tanim", "tarih", "tuketim"],
)
te = pd.read_csv(
    os.path.join(KOK, "data/raw/test.csv"),
    parse_dates=["tarih"],
    dtype={"tanim": str},
    usecols=["id", "tanim", "tarih"],
)

_mx = tr.groupby("tanim").tuketim.max()
_son = tr.groupby("tanim").tarih.max()
_ilk = tr.groupby("tanim").tarih.min()
_s28 = tr[tr.tarih > pd.Timestamp("2026-03-03")].groupby("tanim").tuketim.max()
OLU = (
    set(_mx[_mx == 0].index)
    | set(_son[_son < pd.Timestamp("2026-02-01")].index)
    | set(_s28[_s28 == 0].index)
)
NOTR = te.tanim.isin(OLU).to_numpy()

SOGUK = (~te.tanim.isin(set(tr.tanim))).to_numpy()
KUYRUK = (~SOGUK) & (te.tanim.map(_ilk) >= pd.Timestamp("2026-03-26")).to_numpy()
CEK = (~SOGUK) & (~KUYRUK)
REJIM = [("soguk", SOGUK), ("kuyruk", KUYRUK), ("cekirdek", CEK)]

A6 = np.log1p(pd.read_csv(os.path.join(S, "tuketim_m6_ikiyon.csv")).tuketim.values)
SS = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
print(f"notr (olculmus-curuk) trafo {len(OLU)}, satir {NOTR.sum():,} (%{100 * NOTR.mean():.1f})")


def temizle(giris, cikis):
    b = np.log1p(pd.read_csv(os.path.join(S, giris)).tuketim.values)
    f = b - A6
    pay = float((f[NOTR] ** 2).sum() / (f**2).sum())
    f = np.where(NOTR, 0.0, f)
    rap = {"curuk_Q_payi": pay, "capa": {}}
    for nm, m in REJIM:
        mm = m & ~NOTR
        if mm.sum() == 0:
            continue
        d = float(f[mm].mean())
        f[mm] -= d
        rap["capa"][nm] = dict(satir=int(mm.sum()), kaydirma=-d)
    y = np.clip(np.expm1(A6 + f), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    out.to_csv(os.path.join(S, cikis), index=False)
    kapi = dict(
        satir=len(out),
        id_birebir=bool((out.id.values == SS.iloc[:, 0].values).all()),
        nan=int(out.tuketim.isna().sum()),
        negatif=int((out.tuketim < 0).sum()),
    )
    assert (
        kapi["satir"] == 714688 and kapi["id_birebir"] and not kapi["nan"] and not kapi["negatif"]
    )
    rap["kapi"] = kapi
    print(f"  {giris} -> {cikis}  curuk pay {pay:.3f}  Q {float((f**2).mean()):.5f}")
    return rap


if __name__ == "__main__":
    ISLER = [
        ("tuketim_y30_sota.csv", "tuketim_y40_sota_temiz.csv"),
        ("tuketim_y31_amnezik.csv", "tuketim_y41_amnezik_temiz.csv"),
        ("tuketim_y32_kapasite.csv", "tuketim_y42_kapasite_temiz.csv"),
        ("tuketim_y34_mevsimsel.csv", "tuketim_y43_mevsimsel_temiz.csv"),
    ]
    if len(sys.argv) > 2:
        ISLER = [(sys.argv[1], sys.argv[2])]
    rap = {c: temizle(g, c) for g, c in ISLER}
    json.dump(rap, open(os.path.join(BURA, "y1_temizle.json"), "w"), indent=1)
