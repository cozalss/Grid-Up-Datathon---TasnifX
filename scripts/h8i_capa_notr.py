"""H8i -- SEVIYE-NOTR SOGUK GENLIK CAPASI + KIRPMA.

h8h SONUCU
----------
Gun bileseni SATIR-AGIRLIKLI merkezlenince (mudahale seviyeyi TANIM GEREGI
degistirmez) tablo degisti:

    panel            dMSE_SEVIYE   dMSE_GENLIK   genlik payi   c_genlik
    yaz25 T0           -0,0037       -0,0792         95,5%       3,03
    yaz25 T3           -0,0024       -0,0852         97,3%       3,10
    guz25 T0           -0,0022       -0,0003         10,5%       0,96
    guz25 T3           -0,0059       -0,0001          1,9%       1,01

Iki sonuc:
(1) yaz25'te kazanc SEVIYE DEGIL GENLIK (t=-88, 6/6 tohum) ve seviye
    sizintisi TEMIZLENINCE BUYUDU (-0,0556 -> -0,0792).
(2) guz25'te genlik ekseni NOTR (c ~ 1,0), ZIT DEGIL. Onceki "guz25 buzme
    istiyor" okumasi seviye konfaundiydi. Yani GENLIK EKSENINDE bloklar
    arasi ISARET CELISKISI YOK: guz25 sifir, yaz25 buyuk pozitif.
    Fizik de bunu soyluyor -- gunluk sogutma yuku salinimi yazin buyuk,
    guzun kucuk; model yalnizca buyuk oldugunda az yayiyor.

BU BETIK
--------
1. SEVIYE-NOTR capa formulunu kurar ve ETIKETLI c_genlik'i uretip
   uretmedigini iki blokta sinar (kalibrasyon).
2. Test soguk satirlari icin capayi hesaplar (test etiketi KULLANILMAZ).
3. Secilen c'de KIRPMA TABLOSU (kural 1) -- seviye-notr surumde.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
ONBELLEK = KOK / "data/interim/gun_ekseni"
SAMPIYON = "tuketim_v67_c1335_olay.csv"
P_SOGUK = 0.22159
MIN_YAS, MIN_GUN = 7, 60


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
    return b


def merkezle(b, n_d):
    return b - float(np.dot(n_d, b) / n_d.sum())


def ag_std(b, n_d):
    bc = merkezle(b, n_d)
    return float(np.sqrt(np.dot(n_d, bc**2) / n_d.sum()))


def t3(d):
    ilk = d.groupby("tanim")["tarih"].transform("min")
    yas = (d["tarih"] - ilk).dt.days.to_numpy()
    say = d.groupby("tanim")["tanim"].transform("size").to_numpy()
    return (yas >= MIN_YAS) & (say >= MIN_GUN)


def kalibrasyon(ad: str) -> None:
    m = pd.read_parquet(ONBELLEK / f"{ad}_meta.parquet").reset_index(drop=True)
    mask = t3(m)
    a = m.loc[mask].reset_index(drop=True)
    lgy = np.log1p(np.clip(a["y"].to_numpy(dtype="float64"), 0, None))
    bi, _ = pd.factorize(a["tanim"])
    gi, gun = pd.factorize(a["tarih"])
    nb, ng = int(bi.max()) + 1, int(gi.max()) + 1
    n_d = np.bincount(gi, minlength=ng).astype(float)
    bg = merkezle(iki_yonlu(lgy, bi, gi, nb, ng), n_d)

    sat = []
    for p in sorted(ONBELLEK.glob(f"{ad}_*_taban.npy")):
        pr = np.load(p).astype("float64")[mask]
        bm = merkezle(iki_yonlu(pr, bi, gi, nb, ng), n_d)
        w = n_d / n_d.sum()
        kor = float(np.sum(w * bg * bm) / np.sqrt(np.sum(w * bg**2) * np.sum(w * bm**2)))
        c_capa = kor * float(np.sqrt(np.sum(w * bg**2) / np.sum(w * bm**2)))
        r = lgy - pr
        rg = np.bincount(gi, r, minlength=ng)
        c_et = 1.0 + float(np.dot(rg, bm)) / float(np.dot(n_d, bm**2))
        sat.append(
            {
                "tohum": p.stem.split("_")[1],
                "sig_g": ag_std(bg, n_d),
                "sig_m": ag_std(bm, n_d),
                "kor": kor,
                "c_capa": c_capa,
                "c_ETIKETLI": c_et,
                "oran": c_capa / c_et,
            }
        )
    d = pd.DataFrame(sat)
    print(f"\n--- {ad} T3  {len(a):,} satir, {a.tanim.nunique()} trafo, {ng} gun")
    print(d.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
    print(
        f"  ort c_capa {d.c_capa.mean():.3f}  c_ETIKETLI {d.c_ETIKETLI.mean():.3f}"
        f"  ORAN {d.oran.mean():.3f}"
    )


def kirpma(ad: str, c: float) -> None:
    m = pd.read_parquet(ONBELLEK / f"{ad}_meta.parquet").reset_index(drop=True)
    for etiket, mask in (("T0 ham", np.ones(len(m), bool)), ("T3 temiz", t3(m))):
        a = m.loc[mask].reset_index(drop=True)
        lgy = np.log1p(np.clip(a["y"].to_numpy(dtype="float64"), 0, None))
        bi, _ = pd.factorize(a["tanim"])
        gi, _ = pd.factorize(a["tarih"])
        nb, ng = int(bi.max()) + 1, int(gi.max()) + 1
        n_d = np.bincount(gi, minlength=ng).astype(float)
        print(f"\n  KIRPMA {ad} {etiket}  c={c:.2f}  (SEVIYE-NOTR)")
        print(f"    {'K':>4} {'dMSE':>10} {'SH':>9} {'t':>8}  kazanan")
        for K in (0, 1, 5, 10, 25, 50, 100):
            per, kaz = [], None
            for p in sorted(ONBELLEK.glob(f"{ad}_*_taban.npy")):
                pr = np.load(p).astype("float64")[mask]
                bm = merkezle(iki_yonlu(pr, bi, gi, nb, ng), n_d)
                r = lgy - pr
                d = (r - (c - 1) * bm[gi]) ** 2 - r**2
                katki = np.bincount(bi, d, minlength=nb)
                if kaz is None:
                    kaz = (int((katki < 0).sum()), nb)
                at = np.argsort(katki)[:K]
                tut = ~np.isin(bi, at) if K else np.ones(len(d), bool)
                per.append(float(d[tut].mean()))
            if nb <= K:
                break
            v = np.array(per)
            sh = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
            ek = f"  {kaz[0]}/{kaz[1]} ({kaz[0] / kaz[1]:.1%})" if K == 0 else ""
            print(f"    {K:>4} {v.mean():+10.5f} {sh:9.5f} {v.mean() / sh:+8.2f}{ek}")


def main() -> int:
    print("=" * 92)
    print("1. SEVIYE-NOTR CAPA KALIBRASYONU")
    print("=" * 92)
    for ad in ("yaz25", "guz25"):
        kalibrasyon(ad)

    print("\n" + "=" * 92)
    print("2. TEST SOGUK CAPASI (test etiketi KULLANILMADI)")
    print("=" * 92)
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih", "tuketim"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    ilk_tum = tr.groupby("tanim")["tarih"].min()
    g25 = tr[(tr["tarih"] >= "2025-04-01") & (tr["tarih"] <= "2025-07-31")].copy()
    g25["it"] = g25["tanim"].map(ilk_tum)
    g25 = g25[g25["it"] >= pd.Timestamp("2025-04-01")].reset_index(drop=True)
    a25 = g25.loc[t3(g25)].reset_index(drop=True)
    lg25 = np.log1p(np.clip(a25["tuketim"].to_numpy(dtype="float64"), 0, None))
    b1, _ = pd.factorize(a25["tanim"])
    g1, gun1 = pd.factorize(a25["tarih"])
    n1 = np.bincount(g1, minlength=int(g1.max()) + 1).astype(float)
    p25 = merkezle(iki_yonlu(lg25, b1, g1, int(b1.max()) + 1, int(g1.max()) + 1), n1)
    print(f"\n2025 GERCEK soguk ikiz T3: {len(a25):,} satir, {a25.tanim.nunique()} trafo")
    print(f"  sigma_gercek (satir-agirlikli) = {ag_std(p25, n1):.4f}")

    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    sam = pd.read_csv(KOK / "submissions" / SAMPIYON)
    assert (sam["id"].values == te["id"].values).all()
    te["lg"] = np.log1p(np.clip(sam["tuketim"].to_numpy(dtype="float64"), 0, None))
    tc = te[~te["tanim"].isin(set(tr["tanim"].unique()))].reset_index(drop=True)
    a26 = tc.loc[t3(tc)].reset_index(drop=True)
    b2, _ = pd.factorize(a26["tanim"])
    g2, gun2 = pd.factorize(a26["tarih"])
    n2 = np.bincount(g2, minlength=int(g2.max()) + 1).astype(float)
    p26 = merkezle(
        iki_yonlu(
            a26["lg"].to_numpy(dtype="float64"), b2, g2, int(b2.max()) + 1, int(g2.max()) + 1
        ),
        n2,
    )
    print(f"2026 MODEL test soguk T3 ({SAMPIYON}): {len(a26):,} satir, {a26.tanim.nunique()} trafo")
    print(f"  sigma_model (satir-agirlikli)  = {ag_std(p26, n2):.4f}")

    s25 = pd.Series(p25, index=pd.Index(gun1).dayofyear).groupby(level=0).mean()
    s26 = pd.Series(p26, index=pd.Index(gun2).dayofyear).groupby(level=0).mean()
    w26 = pd.Series(n2, index=pd.Index(gun2).dayofyear).groupby(level=0).sum()
    ortak = s25.index.intersection(s26.index)
    w = w26.loc[ortak].to_numpy()
    w = w / w.sum()
    x, y = s25.loc[ortak].to_numpy(), s26.loc[ortak].to_numpy()
    x = x - np.sum(w * x)
    y = y - np.sum(w * y)
    kor = float(np.sum(w * x * y) / np.sqrt(np.sum(w * x**2) * np.sum(w * y**2)))
    oran = float(np.sqrt(np.sum(w * x**2) / np.sum(w * y**2)))
    print(f"\n  ortak gun {len(ortak)}   oran {oran:.4f}   korelasyon {kor:+.4f}")
    print(f"  >>> c_soguk_capa (seviye-notr) = {kor * oran:.4f}")

    print("\n" + "=" * 92)
    print("3. KIRPMA (kural 1) -- SEVIYE-NOTR, secilen c")
    print("=" * 92)
    for c in (2.20, 2.60, 3.00):
        kirpma("yaz25", c)
    kirpma("guz25", 2.60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
