"""SOGUK HATANIN NEREDE OLDUGU -- sifir/kuyruk ayrisimi ve gozlenebilir
oznitelikler (varlik deseni, giris tarihi) uzerinden seviye kestirimi.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ortak import BLOKLAR, SOGUK_PAY, mse, taban_r, tum_bloklar  # noqa: E402

pd.set_option("display.width", 220)


def main() -> int:
    bloklar = tum_bloklar()

    print("=" * 92)
    print("1) MSE'NIN GERCEK SEVIYEYE GORE AYRISIMI (r_ger = log1p(y) - log1p(guc))")
    print("=" * 92)
    for ad in BLOKLAR:
        b = bloklar[ad]
        r = taban_r(b)
        e2 = (b.lgy - (r + b.lgc)) ** 2
        rg = b.lgy - b.lgc
        kes = pd.cut(
            rg,
            [-0.001, 0.001, 0.5, 1.0, 1.5, 2.0, 99],
            labels=["y=0", "0-0.5", "0.5-1", "1-1.5", "1.5-2", ">2"],
        )
        d = pd.DataFrame({"k": kes, "e2": e2})
        g = d.groupby("k", observed=True).agg(n=("e2", "size"), mse=("e2", "mean"))
        g["n%"] = 100 * g["n"] / len(d)
        g["MSE_pay%"] = 100 * d.groupby("k", observed=True)["e2"].sum() / d["e2"].sum()
        print(f"\n-- {ad}  (toplam MSE {mse(b, r):.4f}) --")
        print(g.round(3).to_string())

    print()
    print("=" * 92)
    print("2) GOZLENEBILIR TRAFO OZNITELIKLERI trafo SEVIYESINI ne kadar aciklar?")
    print("   (hedef: trafonun blok icindeki ortalama r_ger'i)")
    print("=" * 92)
    for ad in BLOKLAR:
        b = bloklar[ad]
        rg = b.lgy - b.lgc
        d = pd.DataFrame(
            {
                "t": b.tanim,
                "rg": rg,
                "guc": b.guc,
                "ilce": b.ilce,
                "yas": b.yas,
                "r0": taban_r(b),
            }
        )
        tr = d.groupby("t").agg(
            rg=("rg", "mean"),
            r0=("r0", "mean"),
            guc=("guc", "first"),
            ilce=("ilce", "first"),
            gun=("rg", "size"),
        )
        ilk = pd.Series(b.giris).groupby(pd.Series(b.tanim)).first()
        tr["giris_gun"] = pd.to_datetime(ilk.reindex(tr.index)).dt.dayofyear
        print(f"\n-- {ad}: trafo ({len(tr)}) duzeyinde --")
        print(f"   var(r_ger_trafo)          {tr['rg'].var():.4f}")
        print(f"   korelasyon( r0 , r_ger )  {tr['r0'].corr(tr['rg']):+.3f}")
        print(f"   korelasyon( log(guc), rg) {np.log(tr['guc']).corr(tr['rg']):+.3f}")
        print(f"   korelasyon( gun sayisi )  {tr['gun'].corr(tr['rg']):+.3f}")
        print(f"   korelasyon( giris gunu )  {tr['giris_gun'].corr(tr['rg']):+.3f}")
        # ilce ANOVA
        gm = tr.groupby("ilce")["rg"].transform("mean")
        print(
            f"   ilce R^2                  {1 - ((tr['rg'] - gm) ** 2).mean() / tr['rg'].var():.3f}"
            f"   (ilce sayisi {tr['ilce'].nunique()})"
        )

    print()
    print("=" * 92)
    print("3) GUN SAYISI (varlik deseni) ile SEVIYE -- testte GOZLENEBILIR")
    print("=" * 92)
    for ad in BLOKLAR:
        b = bloklar[ad]
        rg = b.lgy - b.lgc
        r0 = taban_r(b)
        e = b.lgy - (r0 + b.lgc)
        gun = pd.Series(b.tanim).map(pd.Series(b.tanim).value_counts()).to_numpy()
        kes = pd.cut(gun, [0, 7, 21, 45, 75, 200], labels=["1-7", "8-21", "22-45", "46-75", "76+"])
        d = pd.DataFrame({"k": kes, "e": e, "e2": e * e, "rg": rg})
        g = d.groupby("k", observed=True).agg(
            n=("e", "size"), yanlilik=("e", "mean"), mse=("e2", "mean"), r_ger=("rg", "mean")
        )
        print(f"\n-- {ad} --")
        print(g.round(4).to_string())

    print()
    print(f"NOT: genel dMSE = soguk dMSE x {SOGUK_PAY:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
