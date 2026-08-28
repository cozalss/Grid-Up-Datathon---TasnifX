"""DOGUM DALGALARI -- kis26 ile guz25 neden ters isaret veriyor?

kuyruk_adaylar.py: "genc trafo" duzeltmesi kis26'da -0,0075..-0,0122,
guz25'te +0,0028..+0,0064. Ayni yapi, ters isaret. Bu betik nedenini
arar: genc kohortlar TOPLU DOGUM dalgasi mi, tekil dogum mu?

Toplu dogum = veri setine geriye-dolgu ile giren parti (test 2026-05-11'de
2.222 trafo). Tekil dogum = gercek enerjilendirme. Ikisinin yanliligi
farkli olabilir ve test TOPLU dalgayla dolu.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parents[2]
pd.set_option("display.width", 240)


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
        encoding="utf-8",
    )
    ilk = tr.groupby("tanim")["tarih"].min()
    say = ilk.value_counts().sort_index()
    print("TRAIN'DE DOGUM GUNU HISTOGRAMI -- en kalabalik 20 gun")
    print(say.sort_values(ascending=False).head(20).to_string())
    print(f"\ntoplam trafo {len(ilk):,}   farkli dogum gunu {say.size:,}")
    print(f"medyan gunluk dogum {say.median():.0f}   ort {say.mean():.1f}")

    esik = 20
    toplu_gun = set(say[say >= esik].index)
    print(
        f"\nTOPLU dalga esigi >= {esik} trafo/gun -> {len(toplu_gun)} gun, "
        f"{int(say[say >= esik].sum()):,} trafo ({say[say >= esik].sum() / len(ilk):.3f})"
    )

    print("\n" + "=" * 90)
    print("BLOK BASLANGICINDAN ONCEKI 90 GUNDE DOGANLAR: toplu mu tekil mi?")
    print("=" * 90)
    for ad, bas in (
        ("yaz25", "2025-04-01"),
        ("guz25", "2025-08-01"),
        ("kis26", "2025-12-01"),
        ("TEST", "2026-04-01"),
    ):
        b = pd.Timestamp(bas)
        for alt, ust in ((0, 6), (7, 30), (31, 90)):
            m = ilk[(ilk >= b - pd.Timedelta(days=ust)) & (ilk <= b - pd.Timedelta(days=alt))]
            if len(m) == 0:
                print(f"  {ad:6} {alt:3}-{ust:3}g : YOK")
                continue
            t = int(sum(1 for v in m.values if pd.Timestamp(v) in toplu_gun))
            print(
                f"  {ad:6} {alt:3}-{ust:3}g : {len(m):5,} trafo   "
                f"TOPLU dogumlu {t:5,} ({t / len(m):.3f})   "
                f"dogum gunu araligi {pd.Timestamp(m.min()).date()}..{pd.Timestamp(m.max()).date()}"
            )
        print()

    print("=" * 90)
    print("TEST'TE PANELE GIRIS (test.csv icinde ilk gun) -- toplu dalga")
    print("=" * 90)
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
        encoding="utf-8",
    )
    tilk = te.groupby("tanim")["tarih"].min()
    tsay = tilk.value_counts().sort_values(ascending=False).head(10)
    print(tsay.to_string())
    yeni = tilk[~tilk.index.isin(ilk.index)]
    print(f"\ntestte train'de HIC OLMAYAN trafo: {len(yeni):,}")
    print("bunlarin test'e giris gunu (en kalabalik 6):")
    print(yeni.value_counts().sort_values(ascending=False).head(6).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
