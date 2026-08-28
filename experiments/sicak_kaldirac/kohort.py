"""G1 / G2 KOHORT AYRIMI -- sicak tarafta ayri muamele hakli mi?

Testteki tanim (docs/50 §2):
    G1 = 2025 yazinda (01-04..31-07 2025) gercek tuketimi OLAN trafo
    G2 = 2025 sonbahar/kisinda aktif, 2025 yazinda verisi OLMAYAN trafo

CV bloklarinda yeniden kurulabilirligi:
    yaz25  blok basi 2025-04-01 -> ONCESINDE yaz penceresi YOK. TANIMSIZ.
    guz25  blok basi 2025-08-01 -> yaz penceresi TAM ve blogun ONCESINDE. TANIMLI.
    kis26  blok basi 2025-12-01 -> ayni. TANIMLI.

Yani bu eksen UC blokta olculemez; en fazla IKI blokta. Kapi (uc blokta
ayni isaret) bu yuzden yapisal olarak GECILEMEZ -- ama isaretin iki blokta
tutup tutmadigi yine de bilgidir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ortak import KOK, SICAK_PAY, bloklari_kur, mse, taban_r  # noqa: E402

YAZ = ("2025-04-01", "2025-07-31")
OLCULEBILIR = ("guz25", "kis26")


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih", "tuketim"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    tr["tarih"] = pd.to_datetime(tr["tarih"])
    yaz = tr[(tr["tarih"] >= YAZ[0]) & (tr["tarih"] <= YAZ[1]) & (tr["tuketim"] > 0)]
    g1 = set(yaz["tanim"].unique())
    print(f"  2025 yazinda gercek tuketimi olan trafo: {len(g1):,}")

    bl = bloklari_kur()
    taban = {k: taban_r(b) for k, b in bl.items()}

    print()
    print("=" * 96)
    print("1) KOHORT KIRILIMI (yalniz guz25 / kis26 -- yaz25'te tanimsiz)")
    print("=" * 96)
    ofset = {}
    for k in OLCULEBILIR:
        b = bl[k]
        e = b.lgy - np.maximum(taban[k] + b.lgc, 0.0)
        koh = np.where(b.cerceve["tanim"].isin(g1).to_numpy(), "G1", "G2")
        d = pd.DataFrame({"koh": koh, "e": e, "e2": e * e, "sifir": (b.y == 0).astype(float)})
        t = d.groupby("koh").agg(
            n=("e", "size"), yanlilik=("e", "mean"), mse=("e2", "mean"), sifir=("sifir", "mean")
        )
        t["satir%"] = 100 * t["n"] / len(d)
        t["pay%"] = 100 * d.groupby("koh")["e2"].sum() / d["e2"].sum()
        print(f"\n-- {k} --")
        print(t.round(4).to_string())
        ofset[k] = t["yanlilik"].to_dict()

    print()
    print("=" * 96)
    print("2) KOHORT OFSETI TASINIYOR MU? (kaynak blokta ogren, digerine uygula)")
    print("=" * 96)
    for kay, hed in (("guz25", "kis26"), ("kis26", "guz25")):
        b = bl[hed]
        koh = np.where(b.cerceve["tanim"].isin(g1).to_numpy(), "G1", "G2")
        h = ofset[kay]
        v = np.array([h.get(x, 0.0) for x in koh])
        v = v - v.mean()
        d = mse(b, taban[hed] + v) - mse(b, taban[hed])
        print(
            f"  {kay:6} -> {hed:6}  ofset G1 {h['G1']:+.4f} G2 {h['G2']:+.4f}  "
            f"dMSE {d:+.5f}  (test {d * bl[hed].n / sum(bl[x].n for x in bl) * SICAK_PAY:+.5f})"
        )

    print()
    print("=" * 96)
    print("3) HILE UST SINIRI -- ofset AYNI bloktan ogrenilse ne kazandirirdi?")
    print("=" * 96)
    for k in OLCULEBILIR:
        b = bl[k]
        koh = np.where(b.cerceve["tanim"].isin(g1).to_numpy(), "G1", "G2")
        h = ofset[k]
        v = np.array([h[x] for x in koh])
        v = v - v.mean()
        print(f"  {k}: dMSE {mse(b, taban[k] + v) - mse(b, taban[k]):+.5f}")

    print()
    print("=" * 96)
    print("4) TESTTE KOHORT BUYUKLUKLERI (v83 dosyasi)")
    print("=" * 96)
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "guc"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    soguk = ~te["tanim"].isin(set(tr["tanim"])).to_numpy()
    sicak = ~soguk
    ing1 = te["tanim"].isin(g1).to_numpy()
    print(f"  sicak toplam {int(sicak.sum()):,}")
    print(
        f"  G1 (yaz2025 tuketimi var) {int((sicak & ing1).sum()):,} satir, {te.loc[sicak & ing1, 'tanim'].nunique():,} trafo"
    )
    print(
        f"  G2 (yok)                  {int((sicak & ~ing1).sum()):,} satir, {te.loc[sicak & ~ing1, 'tanim'].nunique():,} trafo"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
