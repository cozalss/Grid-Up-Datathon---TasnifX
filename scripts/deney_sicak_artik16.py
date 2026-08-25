# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""SICAK ARTIK -- ADIM O: aylik kalicilik BLOK ICI mi, YOKSA GERCEK ZAMAN mi?

Adim N 'aralik 1 ay -> kor +0,682' buldu ve test tam o aralikta oturuyor.
Ama a_i, hedef blogu HIC gormeyen kat modelinin artigi. Ayni blogun iki ayi
AYNI kat modelini paylasir; farkli bloklarin aylari PAYLASMAZ. Test ise
hicbir katin icinde degil -- yani testin durumu BLOKLAR ARASI cifte benzer.
Ayrimi yapmadan aylik egri ANLAMSIZ.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import deney as d
import deney_sicak_artik as sa  # noqa: E402
import tuketim_model as tm


def main() -> int:
    egitim, test = d.cerceveleri_kur()
    A, AY_BLOK = {}, {}
    for b in tm.BLOKLAR:
        v = sa.blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        e = pd.Series(v["g"] - v["r"])
        t = pd.Series(dg["tanim"].to_numpy())
        ay = pd.Series(pd.to_datetime(dg["tarih"]).dt.to_period("M").astype(str))
        for a_ in sorted(ay.unique()):
            m = (ay == a_).to_numpy()
            A[a_] = e[m].groupby(t[m]).mean()
            AY_BLOK[a_] = b.ad
    aylar = sorted(A)
    idx = {a_: i for i, a_ in enumerate(aylar)}

    kayit = []
    for i in range(len(aylar)):
        for j in range(i + 1, len(aylar)):
            x = pd.concat([A[aylar[i]], A[aylar[j]]], axis=1, join="inner").dropna()
            x.columns = ["a", "b"]
            if len(x) < 300:
                continue
            kayit.append(
                {
                    "i": aylar[i],
                    "j": aylar[j],
                    "aralik": idx[aylar[j]] - idx[aylar[i]],
                    "ayni_blok": AY_BLOK[aylar[i]] == AY_BLOK[aylar[j]],
                    "kor": float(x["a"].corr(x["b"])),
                    "n": len(x),
                }
            )
    df = pd.DataFrame(kayit)

    print("  a_i KORELASYONU -- ARALIK x (AYNI KAT MI?)")
    print(
        f"  {'aralik':>7}{'AYNI BLOK cift':>16}{'kor':>9}{'|':>3}{'FARKLI BLOK cift':>18}{'kor':>9}"
    )
    for g in sorted(df["aralik"].unique()):
        gr = df[df["aralik"] == g]
        ai = gr[gr["ayni_blok"]]
        fa = gr[~gr["ayni_blok"]]
        s1 = f"{len(ai):>16}{ai['kor'].mean():>+9.3f}" if len(ai) else f"{0:>16}{'-':>9}"
        s2 = f"{len(fa):>18}{fa['kor'].mean():>+9.3f}" if len(fa) else f"{0:>18}{'-':>9}"
        print(f"  {g:>7}{s1}{'|':>3}{s2}")

    print("\n  BLOK SINIRINI GECEN BITISIK AY CIFTLERI (aralik 1, farkli kat):")
    for _, r in df[(df["aralik"] == 1) & (~df["ayni_blok"])].iterrows():
        print(f"    {r['i']} -> {r['j']}   kor {r['kor']:+.3f}   n={r['n']:,}")
    print("  AYNI KATTAKI BITISIK AY CIFTLERI (aralik 1, ayni blok):")
    for _, r in df[(df["aralik"] == 1) & (df["ayni_blok"])].iterrows():
        print(f"    {r['i']} -> {r['j']}   kor {r['kor']:+.3f}   n={r['n']:,}")

    print("\n  OZET: testin durumu 'FARKLI KAT' sutunudur (test hicbir katta yok).")
    fa = df[~df["ayni_blok"]]
    for lo, hi in ((1, 2), (3, 4), (5, 8), (9, 11)):
        gr = fa[fa["aralik"].between(lo, hi)]
        if len(gr):
            print(
                f"    aralik {lo}-{hi:<3} cift={len(gr):3}  ort kor {gr['kor'].mean():+.3f}"
                f"  (SH {gr['kor'].std(ddof=1) / np.sqrt(len(gr)):.3f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
