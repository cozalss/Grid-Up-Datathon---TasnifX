"""H8d -- gun ekseni onbellek panelleri KIMIN paneli, ve c formulunun AGIRLIGI.

IKI SORU
--------
(1) yaz25 paneli "pencerede dogmus trafolar" (SOGUK ikiz) ise, guz25 paneli
    de oyle mi? Degilse guz25 SOGUK hipotezi icin IKINCI BLOK SAYILMAZ ve
    kural 7/9 saglanmamis olur.

(2) h8c'de formul c* = kor * sigma_g/sigma_m = 3,44 verdi ama IZGARA optimumu
    2,1 cikti ve 3,4'te sonuc KOTULESIYOR. Neden?
    Kuadratik kayipta optimum:
        c - 1 = SUM_i r_i * bm[g_i] / SUM_i bm[g_i]^2
    Bu, gunleri o gundeki SATIR SAYISIYLA (n_d) agirliklar. h8c'nin
    np.corrcoef'i gunleri ESIT agirlikladi. Panel asiri dengesiz oldugu icin
    (erken gunlerde 3 satir, gec gunlerde 1.000+) ikisi COK farkli.
    Bu betik dogru agirlikli tahmincileri hesaplar ve izgarayla karsilastirir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
ONBELLEK = KOK / "data/interim/gun_ekseni"


def iki_yonlu(v, bi, gi, nb, ng, tur=400):
    mu = float(v.mean())
    a = np.zeros(nb)
    b = np.zeros(ng)
    cb = np.maximum(np.bincount(bi, minlength=nb), 1)
    cg = np.maximum(np.bincount(gi, minlength=ng), 1)
    for _ in range(tur):
        a = np.bincount(bi, v - mu - b[gi], minlength=nb) / cb
        b = np.bincount(gi, v - mu - a[bi], minlength=ng) / cg
        b -= b.mean()
    return a, b, mu


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    ilk = tr.groupby("tanim")["tarih"].min()
    son = tr.groupby("tanim")["tarih"].max()

    print("=" * 78)
    print("1. PANEL KIMLIGI -- bu trafolar pencerede mi DOGDU?")
    print("=" * 78)
    for ad, bas in (("yaz25", "2025-04-01"), ("guz25", "2025-08-01")):
        m = pd.read_parquet(ONBELLEK / f"{ad}_meta.parquet")
        t = pd.Index(m["tanim"].unique())
        i = ilk.reindex(t)
        s = son.reindex(t)
        bas_ts = pd.Timestamp(bas)
        icinde = (i >= bas_ts).sum()
        print(f"\n{ad}: {len(t)} trafo, pencere {m.tarih.min().date()}..{m.tarih.max().date()}")
        print(
            f"  ilk kaydi pencere BASINDAN sonra (= pencerede DOGMUS): {icinde}/{len(t)}"
            f"  ({icinde / len(t):.1%})"
        )
        print(
            f"  ilk kayit tarihi dagilimi: min {i.min().date()} "
            f"medyan {i.median().date()} max {i.max().date()}"
        )
        print(f"  son kayit tarihi: min {s.min().date()} max {s.max().date()}")
        # gun basi satir dengesizligi
        n_d = m.groupby("tarih").size()
        print(
            f"  gun basi satir: min {n_d.min()} q25 {int(n_d.quantile(0.25))} "
            f"medyan {int(n_d.median())} q75 {int(n_d.quantile(0.75))} max {n_d.max()}"
        )
        say = m.groupby("tanim")["tarih"].nunique()
        print(
            f"  trafo basi gun: min {say.min()} medyan {int(say.median())} max {say.max()}"
            f"  |  tam pencere olan trafo: {(say == n_d.index.nunique()).sum()}"
        )

    print("\n" + "=" * 78)
    print("2. DOGRU AGIRLIKLI c TAHMINCISI vs IZGARA")
    print("=" * 78)
    for ad in ("yaz25", "guz25"):
        m = pd.read_parquet(ONBELLEK / f"{ad}_meta.parquet").reset_index(drop=True)
        lgy = np.log1p(np.clip(m["y"].to_numpy(dtype="float64"), 0, None))
        bi, _ = pd.factorize(m["tanim"])
        gi, _ = pd.factorize(m["tarih"])
        nb, ng = int(bi.max()) + 1, int(gi.max()) + 1
        n_d = np.bincount(gi, minlength=ng).astype(float)
        ag, bg, mug = iki_yonlu(lgy, bi, gi, nb, ng)

        print(
            f"\n--- {ad}   GERCEK gun ekseni sigma: "
            f"esit-agirlik {bg.std():.4f}   n_d-agirlikli "
            f"{np.sqrt(np.average((bg - np.average(bg, weights=n_d)) ** 2, weights=n_d)):.4f}"
        )
        sat = []
        for p in sorted(ONBELLEK.glob(f"{ad}_*_taban.npy")):
            pr = np.load(p).astype("float64")
            am, bm, mum = iki_yonlu(pr, bi, gi, nb, ng)
            r = lgy - pr
            # KESIN optimum: c-1 = <r, bm> / <bm, bm>  (satir uzerinde)
            pay = float(np.dot(np.bincount(gi, r, minlength=ng), bm))
            payda = float(np.dot(n_d, bm**2))
            c_kesin = 1.0 + pay / payda
            # esit-agirlikli formul (h8c'nin yaptigi)
            kor = float(np.corrcoef(bg, bm)[0, 1])
            c_esit = kor * bg.std() / bm.std()
            # n_d-agirlikli formul
            bgw = bg - np.average(bg, weights=n_d)
            bmw = bm - np.average(bm, weights=n_d)
            c_ag = float(np.average(bgw * bmw, weights=n_d) / np.average(bmw**2, weights=n_d))
            mse0 = float((r**2).mean())
            mse_k = float(((r - (c_kesin - 1) * bm[gi]) ** 2).mean())
            sat.append(
                {
                    "tohum": p.stem.split("_")[1],
                    "c_esit": c_esit,
                    "c_ndagirlikli": c_ag,
                    "c_KESIN": c_kesin,
                    "dMSE_at_c_KESIN": mse_k - mse0,
                }
            )
        d = pd.DataFrame(sat)
        print(d.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
        v = d["dMSE_at_c_KESIN"].to_numpy()
        sh = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        print(
            f"  ort c_KESIN {d.c_KESIN.mean():.4f} (std {d.c_KESIN.std():.4f})"
            f"   dMSE {v.mean():+.5f}  eslenik_SH {sh:.5f}  t {v.mean() / sh:+.2f}"
            f"  pozitif {int((v < 0).sum())}/{len(v)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
