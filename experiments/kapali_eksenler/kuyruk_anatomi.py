"""KUYRUK REJIMI ANATOMISI -- ucuncu rejim hic acilmadi.

v83 zincirinde kuyruk rejimi SABIT +0,16640 aliyor (29.873 test satiri,
%4,18). Bu, uc rejim sabiti icinde EN BUYUGU. docs/47 §1 bu sabitin
dayanagi olan olcumu VOID ilan etti:

    "docs/45 tik6'nin kuyruk sayilari (+0,4754 / +0,3531) uretim modeline
     ait degil (kos_lgbm 'Aplus' kolonu). Uretim harmaniyla guz25 +0,0409,
     t=+0,62 (NULL). Kuyruk ekseninin 'iki blokta dogrulandi' hukmu gecersiz."

Yani uretimde duran en buyuk rejim sabiti, curutulmus bir olcumden geliyor.
Bu betik URETIM HARMANI ile (sicak_kaldirac/ortak.py onbellegi, kirpmali
olcut) kuyruk rejiminin anatomisini cikarir.

Kullanim:  uv run python experiments/kapali_eksenler/kuyruk_anatomi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "experiments" / "sicak_kaldirac"))

from ortak import (  # noqa: E402
    BLOKLAR,
    KUYRUK_DELTA,
    SICAK_PAY,
    bloklari_kur,
    kuresel_delta,
    mse,
    taban_r,
)

pd.set_option("display.width", 240)


def taban_kuyruksuz(b) -> np.ndarray:
    """Uretim tabani AMA kuyruk sabiti UYGULANMAMIS (delta=0).

    Boylece kuyruk sabitinin kendisi bir ADAY gibi olculebilir. Kuresel
    seviye her iki halde de yeniden kalibre edilir -- yani olculen sey
    kuresel kayma degil, KUYRUK ILE DIGERLERI ARASINDAKI FARK.
    """
    return taban_r(b, kuyruk=0.0)


def main() -> int:
    bloklar = bloklari_kur()

    print("=" * 100)
    print("0) KUYRUK NUFUSU -- uc blokta")
    print("=" * 100)
    for ad in BLOKLAR:
        b = bloklar[ad]
        k = b.cerceve["kuyruk"].to_numpy()
        nt = b.cerceve.loc[k, "tanim"].nunique()
        print(
            f"{ad:8} sicak satir {b.n:>7,}   kuyruk satir {int(k.sum()):>6,} "
            f"({k.mean():.4f})   kuyruk trafo {nt:>5,}   "
            f"gecmis_gun medyan {b.cerceve.loc[k, 'gecmis_gun'].median() if k.any() else float('nan')}"
        )

    print()
    print("=" * 100)
    print("1) HATA ANATOMISI -- kuyruk vs kuyruk-disi (taban: uretim, kuyruk delta=0)")
    print("   e = log1p(y) - max(r + log1p(guc), 0)   [URETIM KIRPMASI]")
    print("=" * 100)
    for ad in BLOKLAR:
        b = bloklar[ad]
        r0 = taban_kuyruksuz(b)
        e = b.lgy - np.maximum(r0 + b.lgc, 0.0)
        k = b.cerceve["kuyruk"].to_numpy()
        d = pd.DataFrame(
            {
                "grup": np.where(k, "KUYRUK", "diger"),
                "e": e,
                "e2": e * e,
                "y0": (b.y <= 0).astype(float),
            }
        )
        g = d.groupby("grup").agg(
            n=("e", "size"),
            yanlilik=("e", "mean"),
            std=("e", "std"),
            mse=("e2", "mean"),
            y0_pay=("y0", "mean"),
        )
        g["MSE_pay%"] = 100 * d.groupby("grup")["e2"].sum() / d["e2"].sum()
        print(f"\n-- {ad}  (blok sicak MSE {mse(b, r0):.5f}) --")
        print(g.round(5).to_string())

    print()
    print("=" * 100)
    print("2) DOZ-TEPKI -- gecmis_gun kovalarina gore yanlilik (trafo etkisi ICINDE)")
    print("=" * 100)
    kenar = [-1, 6, 30, 90, 180, 400, 10_000]
    etiket = ["<=6g", "7-30g", "31-90g", "91-180g", "181-400g", ">400g"]
    for ad in BLOKLAR:
        b = bloklar[ad]
        r0 = taban_kuyruksuz(b)
        e = b.lgy - np.maximum(r0 + b.lgc, 0.0)
        kes = pd.cut(b.cerceve["gecmis_gun"].to_numpy(), kenar, labels=etiket)
        d = pd.DataFrame({"k": kes, "e": e, "e2": e * e})
        g = d.groupby("k", observed=True).agg(
            n=("e", "size"), trafo=("e", "size"), yanlilik=("e", "mean"), mse=("e2", "mean")
        )
        g["trafo"] = (
            pd.DataFrame({"k": kes, "t": b.cerceve["tanim"].to_numpy()})
            .groupby("k", observed=True)["t"]
            .nunique()
        )
        print(f"\n-- {ad} --")
        print(g.round(5).to_string())

    print()
    print("=" * 100)
    print("3) EN KOTU %1 -- kuyruk bu kuyrugun neresinde?")
    print("=" * 100)
    for ad in BLOKLAR:
        b = bloklar[ad]
        r0 = taban_kuyruksuz(b)
        e2 = (b.lgy - np.maximum(r0 + b.lgc, 0.0)) ** 2
        k = b.cerceve["kuyruk"].to_numpy()
        esik = np.quantile(e2, 0.99)
        kotu = e2 >= esik
        print(
            f"{ad:8} en kotu %1 MSE payi {100 * e2[kotu].sum() / e2.sum():5.1f}%   "
            f"kuyruk satirlarin en-kotu-%1 icindeki payi {100 * k[kotu].mean():5.2f}% "
            f"(taban pay {100 * k.mean():5.2f}%)   "
            f"kuyruk satirlarinin %{100 * kotu[k].mean() if k.any() else 0:4.2f}'i en kotu %1'de"
        )

    print()
    print("=" * 100)
    print("4) KIRPMA TABLOSU (kalici kural 1) -- kuyruk yanliligi trafo bazinda,")
    print("   trafo ortalamasi K en buyuk/kucuk kirpilarak")
    print("=" * 100)
    for ad in BLOKLAR:
        b = bloklar[ad]
        r0 = taban_kuyruksuz(b)
        e = b.lgy - np.maximum(r0 + b.lgc, 0.0)
        k = b.cerceve["kuyruk"].to_numpy()
        if k.sum() == 0:
            print(f"{ad:8} kuyruk satiri YOK")
            continue
        tr = pd.Series(e[k]).groupby(pd.Series(b.cerceve.loc[k, "tanim"].to_numpy())).mean()
        satir = [f"{ad:8} n_trafo={len(tr):4d}"]
        for K in (0, 1, 5, 10, 25, 50):
            v = tr.sort_values()
            v = v.iloc[K : len(v) - K] if len(v) > 2 * K else pd.Series(dtype=float)
            satir.append(f"K={K}: {v.mean():+.4f} (n={len(v)})" if len(v) else f"K={K}: --")
        print("   ".join(satir))

    print()
    print("=" * 100)
    print("5) URETIMDEKI +0,16640 SABITI DOGRU SEVIYE MI?")
    print("   her blokta kuyruk satirlarindaki OPTIMUM delta (1-b arama, kirpmali)")
    print("=" * 100)
    for ad in BLOKLAR:
        b = bloklar[ad]
        r0 = taban_kuyruksuz(b)
        k = b.cerceve["kuyruk"].to_numpy(dtype="float64")
        if k.sum() == 0:
            print(f"{ad:8} kuyruk satiri YOK -- bu blok ekseni GOREMEZ")
            continue
        en_iyi, en_iyi_m = 0.0, None
        for d in np.arange(-0.60, 1.201, 0.005):
            rr = r0 + d * k
            m = mse(b, rr + kuresel_delta(b, rr))
            if en_iyi_m is None or m < en_iyi_m:
                en_iyi, en_iyi_m = float(d), m
        r_uret = r0 + KUYRUK_DELTA * k
        m_uret = mse(b, r_uret + kuresel_delta(b, r_uret))
        r_sifir = r0 + kuresel_delta(b, r0)
        m0 = mse(b, r_sifir)
        print(
            f"{ad:8} optimum delta {en_iyi:+.4f} (MSE {en_iyi_m:.6f})   "
            f"delta=0 {m0:.6f}   delta=0,16640 {m_uret:.6f}   "
            f"uretimin kazanci {m_uret - m0:+.6f}   optimumun {en_iyi_m - m0:+.6f}"
        )
    print()
    print(f"NOT: sicak dMSE -> test dMSE carpani {SICAK_PAY:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
