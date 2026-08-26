"""H8l -- c=2,20 RISK-AYARLI dogru nokta mi?

SORU
----
Tik 2'nin kirpma tablosu bir odunlesme gosterdi:
    c=2,20  K=0 -0,0671   K=25 -0,0011   K=50 +0,0142
    c=1,41  K=0 -0,0297   K=25 -0,0065   K=50 -0,0012
Yani BUYUK c daha cok kazandiriyor ama trafo kirpmasina daha kirilgan.

Gun ekseni duzeltmesinin dogal birimi GUN ve orada c=2,20 saglam (K=50'de
-0,0209, t=-36). Ama testte gerceklesen davranis trafo-kirpilmis gorunume
yakin cikarsa, daha kucuk bir c daha iyi bir BEKLENEN sonuc verebilir.

BU BETIK
--------
c izgarasinda dMSE'yi UC senaryoda birden verir:
    IYIMSER  K=0   (tam panel -- olculen)
    ORTA     K=25  (en buyuk 25 trafo atilmis)
    KOTUMSER K=50  (en buyuk 50 trafo atilmis)
ve her c icin bu ucunun ORTALAMASINI ve EN KOTUSUNU raporlar.

Karar kurali ACIK: gonderilecek c, kotumser senaryoda ZARAR VERMEYEN
noktalar arasinda iyimser kazanci en buyuk olandir. Bu, "en buyuk kazanc"
degil "kaybetmeyen en buyuk kazanc" secimidir -- projede uc kez fazla
iyimser secim LB'de para kaybettirdi.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
ONBELLEK = KOK / "data/interim/gun_ekseni"
P_SOGUK = 0.22159


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


def main() -> int:
    m = pd.read_parquet(ONBELLEK / "yaz25_meta.parquet").reset_index(drop=True)
    lgy = np.log1p(np.clip(m["y"].to_numpy(dtype="float64"), 0, None))
    bi, _ = pd.factorize(m["tanim"])
    gi, _ = pd.factorize(m["tarih"])
    nb, ng = int(bi.max()) + 1, int(gi.max()) + 1
    n_d = np.bincount(gi, minlength=ng).astype(float)
    tohumlar = sorted(ONBELLEK.glob("yaz25_*_taban.npy"))

    # tohum basina gun bileseni ve artik -- bir kez hesapla
    hazir = []
    for p in tohumlar:
        pr = np.load(p).astype("float64")
        b = iki_yonlu(pr, bi, gi, nb, ng)
        bc = b - float(np.dot(n_d, b) / n_d.sum())
        hazir.append((lgy - pr, bc))

    print(f"yaz25 T0, {len(m):,} satir, {nb} trafo, {len(tohumlar)} tohum, SEVIYE-NOTR")
    print(f"Panel dMSE -> test etkisi carpani p_soguk = {P_SOGUK}\n")
    print(
        f"  {'c':>5} | {'K=0 (iyimser)':>14} {'K=25 (orta)':>13} "
        f"{'K=50 (kotumser)':>16} | {'ortalama':>10} {'EN KOTU':>10} | karar"
    )
    print("  " + "-" * 92)

    en_iyi = None
    for c in np.round(np.arange(1.0, 3.01, 0.1), 2):
        senaryo = {}
        for K in (0, 25, 50):
            per = []
            for r, bc in hazir:
                d = (r - (c - 1) * bc[gi]) ** 2 - r**2
                katki = np.bincount(bi, d, minlength=nb)
                at = np.argsort(katki)[:K]
                tut = ~np.isin(bi, at) if K else np.ones(len(d), bool)
                per.append(float(d[tut].mean()))
            senaryo[K] = float(np.mean(per))
        ort = float(np.mean(list(senaryo.values())))
        kotu = max(senaryo.values())
        guvenli = kotu <= 0.0
        etiket = "GUVENLI" if guvenli else ""
        if guvenli and (en_iyi is None or senaryo[0] < en_iyi[1]):
            en_iyi = (float(c), senaryo[0], ort, kotu)
        print(
            f"  {c:5.2f} | {senaryo[0]:+14.5f} {senaryo[25]:+13.5f} "
            f"{senaryo[50]:+16.5f} | {ort:+10.5f} {kotu:+10.5f} | {etiket}"
        )

    print("\n" + "=" * 96)
    print("KARAR")
    print("=" * 96)
    if en_iyi is None:
        print("  Hicbir c kotumser senaryoda guvenli degil -> en dusuk EN KOTU'yu sec.")
    else:
        c, iyi, ort, kotu = en_iyi
        print(f"  Kotumser senaryoda ZARAR VERMEYEN en buyuk kazanc:  c = {c:.2f}")
        print(f"    K=0  {iyi:+.5f}  ->  test {P_SOGUK * iyi:+.6f}")
        print(f"    ort  {ort:+.5f}  ->  test {P_SOGUK * ort:+.6f}")
        print(f"    K=50 {kotu:+.5f}  ->  test {P_SOGUK * kotu:+.6f}")
        print("\n  Karsilastirma c=2,20:")
        s = {}
        for K in (0, 25, 50):
            per = []
            for r, bc in hazir:
                d = (r - 1.2 * bc[gi]) ** 2 - r**2
                katki = np.bincount(bi, d, minlength=nb)
                at = np.argsort(katki)[:K]
                tut = ~np.isin(bi, at) if K else np.ones(len(d), bool)
                per.append(float(d[tut].mean()))
            s[K] = float(np.mean(per))
        print(
            f"    K=0 {s[0]:+.5f} (test {P_SOGUK * s[0]:+.6f}) | "
            f"K=25 {s[25]:+.5f} | K=50 {s[50]:+.5f} (test {P_SOGUK * s[50]:+.6f})"
        )
        print(
            f"\n  ODUNLESME: c={c:.2f} kotumserde guvenli ama iyimserde "
            f"{(1 - iyi / s[0]) * 100:.0f}% daha az kazanc."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
