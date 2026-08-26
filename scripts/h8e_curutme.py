"""H8e -- SOGUK GUN EKSENI BULGUSUNU YIKMAYA CALIS.

BULGU (h8c/h8d)
---------------
yaz25 SOGUK ikiz panelinde (678 trafo, HEPSI pencerede dogmus) gun ekseni
olceginin kesin optimumu c=2,13 ve dMSE=-0,0556 (6/6 tohum, t=-28).
Soguk tarafa bugun HIC gun ekseni olceklemesi uygulanmiyor (0/158.369 satir).

SUPHE -- bulgu SAHTE olabilir, uc mekanizma:
(a) GIRIS/DOGUM ESERI: panel asiri dengesiz (trafo basi medyan 32/116 gun,
    TAM pencere olan trafo SIFIR). Trafolarin ilk gunleri KISMI gun ve
    sistematik dusuk (-0,52 kucuk parti). Gun etkisi bunu emiyor olabilir.
(b) KUCUK-n TRAFOLAR: 1-5 gunluk trafolarin sabit etkisi kestirilemiyor;
    artiklari gun eksenine sizarak sahte genlik uretiyor.
(c) YOGUNLASMA: c=2,1'de K=25 kirpmasi kazancin %85'ini siliyor, K=50'de
    isaret donuyor. 678 trafonun 25'i (%3,7) kazanci tasiyorsa bu bir
    seviye/olay eseri olabilir, mevsimsel genlik degil.

BU BETIK
--------
Ayni olcumu GIDEREK TEMIZLENEN alt panellerde tekrarlar:
    T0 ham panel
    T1 her trafonun ilk 7 gunu atilir      (dogum/olay eseri)
    T2 >=60 gunluk trafolar                 (kucuk-n)
    T3 T1 + T2
    T4 ilk 14 gun atilir + >=60 gun         (en sert)
Kazanc T3/T4'te de duruyorsa mekanizma MEVSIMSEL GENLIK'tir.
Sonuyorsa bulgu CURUKTUR ve uretime GIREMEZ.

Ayrica gun ekseninin DUSUK FREKANS (mevsimsel rampa) bileseni yalniz basina
olceklenirse ne oluyor -- H1'in soguk taraftaki hali.
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


def kesin_c(r, bm, gi, ng):
    n_d = np.bincount(gi, minlength=ng).astype(float)
    pay = float(np.dot(np.bincount(gi, r, minlength=ng), bm))
    payda = float(np.dot(n_d, bm**2))
    return 1.0 + pay / payda if payda > 0 else 1.0


def dusuk_frekans(b: np.ndarray, pencere: int = 15) -> np.ndarray:
    """Merkezi hareketli ortalama -- mevsimsel rampa bileseni."""
    s = pd.Series(b)
    return s.rolling(pencere, center=True, min_periods=1).mean().to_numpy()


def kos(
    ad: str,
    m: pd.DataFrame,
    mask: np.ndarray,
    etiket: str,
    tohum_yollari: list[Path],
    sabit_c: float | None = None,
) -> dict:
    a = m.loc[mask].reset_index(drop=True)
    lgy = np.log1p(np.clip(a["y"].to_numpy(dtype="float64"), 0, None))
    bi, _ = pd.factorize(a["tanim"])
    gi, _ = pd.factorize(a["tarih"])
    nb, ng = int(bi.max()) + 1, int(gi.max()) + 1
    say = a.groupby("tanim")["tarih"].nunique()

    cs, ds, ds_sabit = [], [], []
    for p in tohum_yollari:
        pr = np.load(p).astype("float64")[mask]
        am, bm, mum = iki_yonlu(pr, bi, gi, nb, ng)
        r = lgy - pr
        c = kesin_c(r, bm, gi, ng)
        mse0 = float((r**2).mean())
        cs.append(c)
        ds.append(float(((r - (c - 1) * bm[gi]) ** 2).mean()) - mse0)
        if sabit_c is not None:
            ds_sabit.append(float(((r - (sabit_c - 1) * bm[gi]) ** 2).mean()) - mse0)

    v = np.array(ds)
    sh = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    satir = (
        f"  {etiket:34s} n={mask.sum():>6,} trafo={a.tanim.nunique():>4} "
        f"gun/trafo_med={int(say.median()):>3}  c={np.mean(cs):6.3f}"
        f"  dMSE={v.mean():+.5f} SH={sh:.5f} t={v.mean() / sh:+7.2f} "
        f"poz={int((v < 0).sum())}/{len(v)}"
    )
    if sabit_c is not None and ds_sabit:
        vs = np.array(ds_sabit)
        shs = vs.std(ddof=1) / np.sqrt(len(vs)) if len(vs) > 1 else float("nan")
        satir += f"  | c={sabit_c} sabit: dMSE={vs.mean():+.5f} t={vs.mean() / shs:+6.2f}"
    print(satir)
    return {
        "etiket": etiket,
        "n": int(mask.sum()),
        "c": float(np.mean(cs)),
        "dmse": float(v.mean()),
        "sh": float(sh),
        "poz": int((v < 0).sum()),
        "ntohum": len(v),
        "bi": bi,
        "gi": gi,
        "ng": ng,
        "nb": nb,
        "lgy": lgy,
        "mask": mask,
        "tohumlar": tohum_yollari,
    }


def kirpma(m, sonuc, c, tohum_yollari):
    mask, bi, gi, lgy = sonuc["mask"], sonuc["bi"], sonuc["gi"], sonuc["lgy"]
    nb, ng = sonuc["nb"], sonuc["ng"]
    print(f"\n  KIRPMA  ({sonuc['etiket']}, c={c:.2f})")
    print(f"    {'K':>4} {'dMSE':>10} {'SH':>9} {'t':>7}  kazanan")
    for K in (0, 1, 5, 10, 25, 50):
        per, kaz = [], None
        for p in tohum_yollari:
            pr = np.load(p).astype("float64")[mask]
            am, bm, mum = iki_yonlu(pr, bi, gi, nb, ng)
            r = lgy - pr
            d = (r - (c - 1) * bm[gi]) ** 2 - r**2
            katki = np.bincount(bi, d, minlength=nb)
            if kaz is None:
                kaz = (int((katki < 0).sum()), nb)
            at = np.argsort(katki)[:K]
            tut = ~np.isin(bi, at) if K else np.ones(len(d), bool)
            per.append(float(d[tut].mean()))
        v = np.array(per)
        sh = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        ek = f"  {kaz[0]}/{kaz[1]} ({kaz[0] / kaz[1]:.1%})" if K == 0 else ""
        print(f"    {K:>4} {v.mean():+10.5f} {sh:9.5f} {v.mean() / sh:+7.2f}{ek}")


def frekans_ayrim(m, mask, tohum_yollari, etiket):
    """Gun eksenini dusuk/yuksek frekansa ayirip AYRI olcekle (H1'in soguk hali)."""
    a = m.loc[mask].reset_index(drop=True)
    lgy = np.log1p(np.clip(a["y"].to_numpy(dtype="float64"), 0, None))
    bi, _ = pd.factorize(a["tanim"])
    gi, gun = pd.factorize(a["tarih"])
    nb, ng = int(bi.max()) + 1, int(gi.max()) + 1
    sira = np.argsort(gun.values)
    ters = np.empty(ng, int)
    ters[sira] = np.arange(ng)

    print(f"\n  FREKANS AYRIMI  ({etiket})")
    print(
        f"    {'pencere':>8} {'c_dusuk':>9} {'c_yuksek':>9} {'dMSE_ayri':>11} "
        f"{'dMSE_tek':>10} {'kazanc':>9}"
    )
    for pen in (7, 15, 31):
        cd, cy, d_ayri, d_tek = [], [], [], []
        for p in tohum_yollari:
            pr = np.load(p).astype("float64")[mask]
            am, bm, mum = iki_yonlu(pr, bi, gi, nb, ng)
            bs = bm[sira]
            lo_s = dusuk_frekans(bs, pen)
            lo = np.empty(ng)
            lo[sira] = lo_s
            hi = bm - lo
            r = lgy - pr
            n_d = np.bincount(gi, minlength=ng).astype(float)
            rg = np.bincount(gi, r, minlength=ng)
            # iki bilesenli en kucuk kareler (satir uzerinde)
            A = np.array(
                [
                    [float(np.dot(n_d, lo * lo)), float(np.dot(n_d, lo * hi))],
                    [float(np.dot(n_d, lo * hi)), float(np.dot(n_d, hi * hi))],
                ]
            )
            y = np.array([float(np.dot(rg, lo)), float(np.dot(rg, hi))])
            try:
                sol = np.linalg.solve(A, y)
            except np.linalg.LinAlgError:
                continue
            k_lo, k_hi = 1 + sol[0], 1 + sol[1]
            mse0 = float((r**2).mean())
            d_ayri.append(float(((r - sol[0] * lo[gi] - sol[1] * hi[gi]) ** 2).mean()) - mse0)
            c1 = kesin_c(r, bm, gi, ng)
            d_tek.append(float(((r - (c1 - 1) * bm[gi]) ** 2).mean()) - mse0)
            cd.append(k_lo)
            cy.append(k_hi)
        if not d_ayri:
            continue
        print(
            f"    {pen:>8} {np.mean(cd):9.3f} {np.mean(cy):9.3f} "
            f"{np.mean(d_ayri):+11.5f} {np.mean(d_tek):+10.5f} "
            f"{np.mean(d_ayri) - np.mean(d_tek):+9.5f}"
        )


def main() -> int:
    for ad in ("yaz25", "guz25"):
        m = pd.read_parquet(ONBELLEK / f"{ad}_meta.parquet").reset_index(drop=True)
        tohumlar = sorted(ONBELLEK.glob(f"{ad}_*_taban.npy"))
        ilk = m.groupby("tanim")["tarih"].transform("min")
        yas = (m["tarih"] - ilk).dt.days.to_numpy()
        say = m.groupby("tanim")["tanim"].transform("size").to_numpy()

        print("=" * 100)
        print(f"{ad}  ({len(m):,} satir, {m.tanim.nunique()} trafo, {len(tohumlar)} tohum)")
        print("=" * 100)
        t0 = kos(ad, m, np.ones(len(m), bool), "T0 ham panel", tohumlar)
        c_ref = round(t0["c"], 2)
        t1 = kos(ad, m, yas >= 7, "T1 ilk 7 gun atildi", tohumlar, c_ref)
        t2 = kos(ad, m, say >= 60, "T2 >=60 gunluk trafolar", tohumlar, c_ref)
        t3 = kos(ad, m, (yas >= 7) & (say >= 60), "T3 = T1 + T2", tohumlar, c_ref)
        t4 = kos(ad, m, (yas >= 14) & (say >= 60), "T4 ilk14 + >=60 (en sert)", tohumlar, c_ref)

        if t3["n"] > 1000:
            kirpma(m, t3, t3["c"], tohumlar)
            frekans_ayrim(m, t3["mask"], tohumlar, "T3")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
