"""PANEL SINIR -- PARTI BUYUKLUGU ISTISNASI (son_islem_olay.py ile ortusme).

scripts/son_islem_olay.py zaten AYNI mekanizmayi (kismi gun) v67 zincirine
uyguladi (1.832 sicak satir) ve KRITIK bir istisna kaydetti:

    "PARTI BUYUKLUGU BELIRLEYICI: ayni gun 100'den fazla trafo dogduysa
     dusus neredeyse YOK (-0,11). Bu bir enerjilendirme dalgasi degil,
     veri setine TOPLU KATILIM (geriye dolgu) -- olculen gun TAMDIR."

Bu, panel sinir duzeltmesinin EN BUYUK parcasini vurur: testteki 3.860
giris satirinin 2.370'i 2026-05-11'de, yani 100+'lik DEV bir partide.

Bu betik sinir yanliligini PARTI BUYUKLUGUNE gore ayirir ve duzeltmenin
gercekten uygulanabilir kismini olcer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "experiments" / "sicak_kaldirac"))
import ortak as S  # noqa: E402

CIK = Path(__file__).resolve().parent
GUN = pd.Timedelta(days=1)
TRAIN_BAS = pd.Timestamp("2025-01-01")
TRAIN_SON = pd.Timestamp("2026-03-31")
TEST_BAS = pd.Timestamp("2026-04-01")
TEST_SON = pd.Timestamp("2026-07-31")
BLOKLAR = ("yaz25", "guz25", "kis26")
pd.set_option("display.width", 240)


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih", "tuketim"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
        encoding="utf-8",
    )
    tr = tr.sort_values(["tanim", "tarih"], kind="mergesort")
    onc = tr.groupby("tanim", observed=True)["tarih"].shift(1)
    son = tr.groupby("tanim", observed=True)["tarih"].shift(-1)
    tr["giris"] = (onc.isna() | ((tr["tarih"] - onc) > GUN)) & (tr["tarih"] != TRAIN_BAS)
    tr["cikis"] = (son.isna() | ((son - tr["tarih"]) > GUN)) & (tr["tarih"] != TRAIN_SON)
    # ayni gun kac trafo GIRIYOR / CIKIYOR
    gp = tr.loc[tr["giris"]].groupby("tarih").size()
    cp = tr.loc[tr["cikis"]].groupby("tarih").size()
    tr["g_parti"] = tr["tarih"].map(gp).fillna(0).astype(int)
    tr["c_parti"] = tr["tarih"].map(cp).fillna(0).astype(int)

    tr["lg"] = np.log1p(np.clip(tr["tuketim"], 0, None))
    tr["lg_onc"] = tr.groupby("tanim", observed=True)["lg"].shift(1)
    tr["lg_son"] = tr.groupby("tanim", observed=True)["lg"].shift(-1)

    print("=" * 100)
    print("MODELSIZ: sinir gunu - komsu gun farki, PARTI BUYUKLUGUNE gore")
    print("=" * 100)
    kenar = [0, 20, 100, 1e9]
    etik = ["<20", "20-99", "100+"]
    for tur, mas, komsu, pk in (
        ("GIRIS", tr["giris"] & tr["lg_son"].notna(), "lg_son", "g_parti"),
        ("CIKIS", tr["cikis"] & tr["lg_onc"].notna(), "lg_onc", "c_parti"),
    ):
        d = tr.loc[mas].copy()
        d["fark"] = d["lg"] - d[komsu]
        d["p"] = pd.cut(d[pk], kenar, labels=etik)
        g = d.groupby("p", observed=True).agg(
            n=("fark", "size"), gun=("tarih", "nunique"), fark=("fark", "mean"), std=("fark", "std")
        )
        g["t"] = g["fark"] / (g["std"] / np.sqrt(g["n"]))
        print(f"\n-- {tur} --")
        print(g.round(4).to_string())

    print("\n" + "=" * 100)
    print("MODEL ARTIGI: uc blokta, PARTI BUYUKLUGUNE gore (uretim tabani, kirpmali)")
    print("=" * 100)
    tb = tr[["tanim", "tarih", "giris", "cikis", "g_parti", "c_parti"]]
    sb = S.bloklari_kur()
    kayit = []
    for ad in BLOKLAR:
        b = sb[ad]
        r0 = S.taban_r(b)
        sol = pd.DataFrame(
            {
                "tanim": b.cerceve["tanim"].to_numpy(),
                "tarih": pd.to_datetime(b.cerceve["tarih"].to_numpy()),
            }
        )
        sol["_i"] = np.arange(len(sol))
        j = sol.merge(tb, on=["tanim", "tarih"], how="left").sort_values("_i")
        e = b.lgy - np.maximum(r0 + b.lgc, 0.0)
        print(f"\n-- {ad} --")
        for tur, mk, pk in (("GIRIS", "giris", "g_parti"), ("CIKIS", "cikis", "c_parti")):
            m = j[mk].fillna(False).to_numpy()
            if m.sum() == 0:
                continue
            p = pd.cut(j.loc[m, pk].to_numpy(), kenar, labels=etik)
            d = pd.DataFrame({"p": p, "e": e[m]})
            g = d.groupby("p", observed=True).agg(
                n=("e", "size"), yanlilik=("e", "mean"), std=("e", "std")
            )
            g["t"] = g["yanlilik"] / (g["std"] / np.sqrt(g["n"]))
            print(f"  {tur}:")
            print("   " + g.round(4).to_string().replace("\n", "\n   "))
            for pp, row in g.iterrows():
                kayit.append(
                    {
                        "blok": ad,
                        "tur": tur,
                        "parti": str(pp),
                        "n": int(row["n"]),
                        "yanlilik": float(row["yanlilik"]),
                        "t": float(row["t"]),
                    }
                )

    print("\n" + "=" * 100)
    print("TESTTE PARTI DAGILIMI -- duzeltmenin gercekten uygulanabilir kismi")
    print("=" * 100)
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
        encoding="utf-8",
    )
    ort = pd.concat([tr[["tanim", "tarih"]], te], ignore_index=True).sort_values(
        ["tanim", "tarih"], kind="mergesort"
    )
    o2 = ort.groupby("tanim", observed=True)["tarih"].shift(1)
    s2 = ort.groupby("tanim", observed=True)["tarih"].shift(-1)
    ort["giris"] = o2.isna() | ((ort["tarih"] - o2) > GUN)
    ort["cikis"] = s2.isna() | ((s2 - ort["tarih"]) > GUN)
    t = ort[ort["tarih"] >= TEST_BAS].copy()
    t["cikis"] &= t["tarih"] != TEST_SON
    for tur in ("giris", "cikis"):
        pp = t.loc[t[tur]].groupby("tarih").size()
        v = t.loc[t[tur], "tarih"].map(pp)
        kes = pd.cut(v, kenar, labels=etik)
        print(f"\n  TEST {tur.upper()} ({int(t[tur].sum()):,} satir) parti dagilimi:")
        print("   " + kes.value_counts().sort_index().to_string().replace("\n", "\n   "))

    (CIK / "panel_sinir_parti.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in kayit), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
