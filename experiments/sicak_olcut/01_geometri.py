"""KAT GEOMETRISI -- her CV blogunun egitim katlarinin hedefe gore konumu.

Olculen iki kirlilik olcusu (SICAK tarafta anlamli olanlar):

  (1) ILERI KAT PAYI : egitim satirlarinin, etiketi hedef blogun etiket
      penceresinin BASLANGICINDAN SONRA olan orani. TEST'te bu oran 0'dir
      (tum egitim gecmiste). Sifirdan buyuk her deger, hedef blokta
      TEST'te bulunmayan bir zaman yonu demektir.

  (2) OZET-KAPSAMA PAYI : egitim satirlarinin, kendi ``t_*`` ozet
      penceresi hedef blogun etiket penceresiyle KESISEN orani. Bu
      satirlarda oznitelikler hedef donemin tuketimini iceriyor; model
      f(t_*) -> y esleme fonksiyonunu, hedef donemin cevabini goren bir
      oznitelik rejiminde kaliyor. TEST'te bu oran da 0'dir.

Not: soguktaki "tanim ezberlenebilirligi" olcusunun SICAK tarafta karsiligi
yoktur -- sicak trafo tanimi geregi egitimde vardir (%100), o yuzden o sayi
bilgi tasimaz. Sicakta kirliligi tasiyan sey KIMLIK degil, ZAMAN YONUDUR.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "scripts"))

BLOKLAR = {
    "yaz25": ("2025-04-01", "2025-07-31"),
    "guz25": ("2025-08-01", "2025-11-30"),
    "kis26": ("2025-12-01", "2026-03-31"),
}
EK = {
    "sub25": ("2025-02-01", "2025-03-31"),
    "bah25": ("2025-05-01", "2025-08-31"),
    "yaz25b": ("2025-07-01", "2025-10-31"),
    "guz25b": ("2025-09-01", "2025-12-31"),
    "kis26b": ("2025-11-01", "2026-02-28"),
    "bah26": ("2026-01-01", "2026-03-31"),
}
TUM = {**BLOKLAR, **EK}
EGITIM_BASI = pd.Timestamp("2025-01-01")
TEST = ("2026-04-01", "2026-07-31")


def ozet(bas: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    return EGITIM_BASI, pd.Timestamp(bas) - pd.Timedelta(days=1)


def tut(hedef: str) -> list[str]:
    hb, hs = (pd.Timestamp(x) for x in BLOKLAR[hedef])
    return [ad for ad, (b, s) in TUM.items() if pd.Timestamp(s) < hb or pd.Timestamp(b) > hs]


def main() -> int:
    eg = pd.read_parquet(
        KOK / "data/interim/deney/egitim.parquet", columns=["_blok", "tarih", "soguk_mu", "tanim"]
    )
    ek = pd.read_parquet(
        KOK / "data/interim/deney/ek_kokenler.parquet",
        columns=["_blok", "tarih", "soguk_mu", "tanim"],
    )
    ek = ek[ek["_blok"].isin(EK)]
    hepsi = pd.concat([eg, ek], ignore_index=True)
    sicak = hepsi[hepsi["soguk_mu"] == 0]

    print("=" * 96)
    print("1) KAT GEOMETRISI -- her hedef blok icin hangi kokenler egitime giriyor")
    print("=" * 96)
    print(f"{'hedef':8}{'etiket penceresi':26}{'egitim kokenleri (ozet penceresi sonu)'}")
    print("-" * 96)
    ozet_satir = []
    for hedef in (*BLOKLAR, "TEST"):
        if hedef == "TEST":
            kokler = list(TUM)
            hb = pd.Timestamp(TEST[0])
            hs = pd.Timestamp(TEST[1])
        else:
            kokler = tut(hedef)
            hb, hs = (pd.Timestamp(x) for x in BLOKLAR[hedef])
        etiketler = []
        for k in sorted(kokler, key=lambda a: TUM[a][0]):
            b, s = (pd.Timestamp(x) for x in TUM[k])
            _, oz_son = ozet(TUM[k][0])
            yon = "ILERI" if b > hb else "gecmis"
            kaps = "KAPSAR" if (oz_son >= hb) else "-"
            etiketler.append(f"{k}[{yon},ozet<={oz_son:%m/%d},{kaps}]")
        print(f"{hedef:8}{hb:%Y-%m-%d} .. {hs:%Y-%m-%d}   {' '.join(etiketler)}")

        # satir agirlikli
        alt = sicak[sicak["_blok"].isin(kokler)]
        n = len(alt)
        ileri = 0
        kapsayan = 0
        for k in kokler:
            m = int((alt["_blok"] == k).sum())
            b, _s = (pd.Timestamp(x) for x in TUM[k])
            _, oz_son = ozet(TUM[k][0])
            if b > hb:
                ileri += m
            if oz_son >= hb:
                kapsayan += m
        ozet_satir.append((hedef, n, ileri, kapsayan, len(kokler)))
    print()
    print("=" * 96)
    print("2) KIRLILIK OLCULERI (SICAK satirlar, satir agirlikli)")
    print("=" * 96)
    print(f"{'hedef':8}{'egitim satiri':>16}{'koken':>7}{'ILERI KAT PAYI':>18}{'OZET-KAPSAMA':>16}")
    print("-" * 96)
    for hedef, n, ileri, kap, nk in ozet_satir:
        print(f"{hedef:8}{n:>16,}{nk:>7}{100 * ileri / n:>17.1f}%{100 * kap / n:>15.1f}%")
    print()
    print("=" * 96)
    print("3) GEOMETRIK MESAFE -- TEST'e benzerlik")
    print("=" * 96)
    print(
        f"{'hedef':8}{'ozet gun':>10}{'ufuk gun':>10}{'bosluk':>8}{'ileri kat':>11}{'ozet kaps.':>12}"
    )
    print("-" * 96)
    for hedef in (*BLOKLAR, "TEST"):
        if hedef == "TEST":
            hb, hs = (pd.Timestamp(x) for x in TEST)
        else:
            hb, hs = (pd.Timestamp(x) for x in BLOKLAR[hedef])
        oz_gun = (hb - EGITIM_BASI).days
        ufuk = (hs - hb).days + 1
        rec = next(r for r in ozet_satir if r[0] == hedef)
        print(
            f"{hedef:8}{oz_gun:>10}{ufuk:>10}{0:>8}{100 * rec[2] / rec[1]:>10.1f}%{100 * rec[3] / rec[1]:>11.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
