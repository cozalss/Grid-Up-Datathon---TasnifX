# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""SICAK ARTIK -- ADIM N: a_i KALICILIGI ZAMAN ARALIGINA gore nasil sonuyor?

Bloklar kaba (4 ay). Etiket ayina gore a_i kestirip, AY CIFTLERI arasi
korelasyonu ARALIGA (ay) gore toplamak, 'eksen kapali mi' sorusunu TEST
UFKUNA tasir: test 2026-04..07, en yakin egitim ayi 2026-03 (aralik 1..4 ay),
mevsimsel esi 2025-04..07 (aralik 9..12 ay).
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
    tt = pd.to_datetime(test["tarih"])
    print(f"  TEST tarih araligi {tt.min().date()} .. {tt.max().date()}  n={len(test):,}")
    tsicak = test[test["soguk_mu"] == 0]
    print(f"  TEST sicak {len(tsicak):,}  trafo={tsicak['tanim'].nunique():,}")

    A, N = {}, {}
    kapsam_test = {}
    for b in tm.BLOKLAR:
        v = sa.blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        e = pd.Series(v["g"] - v["r"])
        t = pd.Series(dg["tanim"].to_numpy())
        ay = pd.Series(pd.to_datetime(dg["tarih"]).dt.to_period("M").astype(str))
        for a_ in sorted(ay.unique()):
            m = (ay == a_).to_numpy()
            A[a_] = e[m].groupby(t[m]).mean()
            N[a_] = e[m].groupby(t[m]).size()
        kapsam_test[b.ad] = float(tsicak["tanim"].isin(pd.Series(dg["tanim"]).unique()).mean())
    aylar = sorted(A)
    print(f"  etiket aylari: {aylar[0]} .. {aylar[-1]}  (n={len(aylar)})")
    print(
        "  TEST sicak satirlarin blok trafolariyla kapsanma orani: "
        + "  ".join(f"{k} %{v * 100:.1f}" for k, v in kapsam_test.items())
    )

    idx = {a_: i for i, a_ in enumerate(aylar)}
    kayit = []
    for i in range(len(aylar)):
        for j in range(i + 1, len(aylar)):
            x = pd.concat([A[aylar[i]], A[aylar[j]]], axis=1, join="inner").dropna()
            x.columns = ["a", "b"]
            if len(x) < 300:
                continue
            kayit.append((idx[aylar[j]] - idx[aylar[i]], float(x["a"].corr(x["b"])), len(x)))
    df = pd.DataFrame(kayit, columns=["aralik", "kor", "n"])
    print("\n  a_i KORELASYONU  ARALIGA gore (aylik dilimler, ortak trafolar)")
    print(
        f"  {'aralik(ay)':>11}{'cift':>6}{'ort kor':>10}{'SH':>9}{'min':>9}{'max':>9}{'ort n':>9}"
    )
    for g, gr in df.groupby("aralik"):
        sh = float(gr["kor"].std(ddof=1) / np.sqrt(len(gr))) if len(gr) > 1 else np.nan
        print(
            f"  {g:>11}{len(gr):>6}{gr['kor'].mean():>+10.3f}{sh:>9.3f}"
            f"{gr['kor'].min():>+9.3f}{gr['kor'].max():>+9.3f}{gr['n'].mean():>9.0f}"
        )

    print("\n  MEVSIMSEL EKO var mi?  aralik 9..11 (mevsimsel es, 1 yil eksigi) vs aralik 5..8")
    for etiket, sec in (
        ("aralik 1-2", df["aralik"] <= 2),
        ("aralik 3-4", df["aralik"].between(3, 4)),
        ("aralik 5-8", df["aralik"].between(5, 8)),
        ("aralik 9-11", df["aralik"] >= 9),
    ):
        gr = df[sec]
        print(f"    {etiket:12} cift={len(gr):3}  ort kor {gr['kor'].mean():+.3f}")

    print("\n  TEST UFKU: test ayi x en yakin egitim ayi araliklari")
    tay = sorted(tt.dt.to_period("M").astype(str).unique())
    print(f"    test aylari: {tay}   son egitim ayi: {aylar[-1]}")
    for a_ in tay:
        g = (pd.Period(a_, "M") - pd.Period(aylar[-1], "M")).n
        print(f"    {a_}: son egitim ayina aralik {g} ay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
