"""H17 -- b_soguk YANLIS NUFUSA MI CAPALANMIS? (S3'e YAZILACAK SAYI)

DENETLENEN SAYI
---------------
On kayitli ``b_soguk = +0,16``. YOL 1'i tam olarak su:
    "kis26 soguk yanlilik +0,3017, sicak +0,1899 -> SOGUK FAZLASI +0,1118"
Bu, kis26 soguk nufusunda olculdu: **%59 tekil / %39 orta / %1 TOPLU**.
TEST soguk nufusu ise **%11,9 tekil / %7,4 orta / %80,7 TOPLU**.

Yani delta'miz H8'in basina gelenin BIREBIR aynisina acik olabilir: gercek
etiketlerden, ama YANLIS NUFUSUN ikizinden. (H8'de duzeltince 2,20 -> 1,60.)

NEDEN BEDAVA DEGIL
------------------
Ozdeslik her delta icin b'yi TAM cozer -- ama BANKAYA YATIRILAN kazanc
delta'nin b'ye yakinligina baglidir:
    kazanc(delta, b) = 2*p*delta*b - p*delta^2
    gercek b=0,25 ise  delta=0,16 -> 0,01206   delta=0,25 -> 0,01385
    fark 0,0018 -- kalibrasyon orakul tavaniyla (0,002) AYNI BUYUKLUKTE.

OLCUM
-----
data/interim/kis26_soguk_meta.parquet (61.918 satir, y ile)
data/interim/deney/soguk_tahmin_kis26.npz (3 tohum x cat/xgb/lgbm)
    yanlilik b = ort( log1p(y) - tahmin )
Parti sinifina gore ayristirilir, kirpma tablosu verilir, test karisimina
agirliklandirilir.

YON BILINMIYOR. Toplu katilim "zaten calisan trafolarin veri setine alinmasi"
ise model onlari daha az yanlis fiyatliyor olabilir (b KUCUK); ya da tam
tersi. Sonuca gore yazilir, beklentiye gore degil.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
P_SOGUK = 0.22159


def main() -> int:
    m = pd.read_parquet(KOK / "data/interim/kis26_soguk_meta.parquet").reset_index(drop=True)
    z = np.load(KOK / "data/interim/deney/soguk_tahmin_kis26.npz")
    lgy = np.log1p(np.clip(m["y"].to_numpy(dtype="float64"), 0, None))

    tohumlar = sorted({k.split("_")[0] for k in z.files})
    print(
        f"kis26 soguk: {len(m):,} satir, {m.tanim.nunique():,} trafo, "
        f"{len(tohumlar)} tohum x {len(z.files) // len(tohumlar)} model"
    )

    # tohum basina uc modelin ortalamasi (uretim harmani icin makul vekil)
    tahmin = {}
    for t in tohumlar:
        kollar = [z[k] for k in z.files if k.startswith(t + "_")]
        tahmin[t] = np.mean(kollar, axis=0)
    genel = np.mean(list(tahmin.values()), axis=0)
    print(
        f"  GENEL yanlilik b = {float((lgy - genel).mean()):+.4f}   (docs/43 YOL 1: soguk +0,3017)"
    )

    # ---- parti sinifi
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    ilk = tr.groupby("tanim")["tarih"].min()
    parti = ilk.groupby(ilk).size()
    bs = m["tanim"].map(ilk.map(parti))
    sinif = pd.cut(bs, [-1, 19, 99, 10**9], labels=["tekil/kucuk <20", "orta 20-99", "TOPLU >=100"])
    m["sinif"] = sinif

    # ---- test karisimi
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    tc = te[~te["tanim"].isin(set(tr["tanim"].unique()))]
    ilk_te = tc.groupby("tanim")["tarih"].min()
    p_te = ilk_te.groupby(ilk_te).size()
    st = pd.cut(
        tc["tanim"].map(ilk_te.map(p_te)),
        [-1, 19, 99, 10**9],
        labels=["tekil/kucuk <20", "orta 20-99", "TOPLU >=100"],
    )
    agirlik = st.value_counts(normalize=True).to_dict()

    print("\n" + "=" * 92)
    print("1. YANLILIK, PARTI SINIFINA GORE")
    print("=" * 92)
    print(
        f"\n  {'sinif':<18} {'satir':>8} {'trafo':>7} {'kis26 payi':>11} "
        f"{'TEST payi':>10} {'b':>9} {'tohum SH':>9}"
    )
    b_sinif = {}
    for s in ("tekil/kucuk <20", "orta 20-99", "TOPLU >=100"):
        msk = (m["sinif"] == s).to_numpy()
        if msk.sum() == 0:
            continue
        per = [float((lgy[msk] - tahmin[t][msk]).mean()) for t in tohumlar]
        b = float(np.mean(per))
        sh = float(np.std(per, ddof=1) / np.sqrt(len(per)))
        b_sinif[s] = b
        print(
            f"  {s:<18} {int(msk.sum()):>8,} {m.loc[msk, 'tanim'].nunique():>7,} "
            f"{msk.mean():>11.4f} {agirlik.get(s, 0):>10.4f} {b:>+9.4f} {sh:>9.4f}"
        )

    print(
        "\n  UYARI: TOPLU sinifinin kis26'daki orneklemi cok kucuk "
        f"({int((m['sinif'] == 'TOPLU >=100').sum()):,} satir). Asagida gradyan sinanir."
    )

    # ---- parti buyuklugu SUREKLI degisken olarak: gradyan var mi?
    print("\n" + "=" * 92)
    print("2. GRADYAN -- yanlilik parti buyuklugu ile sistematik degisiyor mu?")
    print("=" * 92)
    kova = pd.qcut(bs, 6, duplicates="drop")
    print(f"\n  {'parti kovasi':<24} {'satir':>8} {'ort parti':>10} {'b':>9}")
    xs, ys, ws = [], [], []
    for k, g in m.assign(kova=kova, bs=bs).groupby("kova", observed=True):
        msk = m.index.isin(g.index)
        b = float((lgy[msk] - genel[msk]).mean())
        print(f"  {str(k):<24} {len(g):>8,} {g['bs'].mean():>10.1f} {b:>+9.4f}")
        xs.append(float(np.log1p(g["bs"].mean())))
        ys.append(b)
        ws.append(len(g))
    if len(xs) >= 3:
        w = np.array(ws, float)
        x = np.array(xs)
        yv = np.array(ys)
        xm = np.average(x, weights=w)
        ym = np.average(yv, weights=w)
        egim = float(
            np.average((x - xm) * (yv - ym), weights=w) / np.average((x - xm) ** 2, weights=w)
        )
        print(f"\n  log(parti) egimi = {egim:+.4f} / log birim")
        print(f"  -> parti 10 kat buyudukce yanlilik {egim * np.log(10):+.4f} degisiyor")

    # ---- kirpma, sinif bazinda
    print("\n" + "=" * 92)
    print("3. KIRPMA TABLOSU -- sinif bazinda (kalici kural 1)")
    print("=" * 92)
    for s in ("tekil/kucuk <20", "orta 20-99"):
        msk = (m["sinif"] == s).to_numpy()
        if msk.sum() < 1000:
            continue
        tan = m.loc[msk, "tanim"].to_numpy()
        bi, _ = pd.factorize(tan)
        nb = int(bi.max()) + 1
        print(f"\n  {s}  ({int(msk.sum()):,} satir, {nb} trafo)")
        print(f"    {'K':>4} {'b':>9} {'SH':>8} {'kalan trafo':>12} {'kalan satir':>12}")
        for K in (0, 1, 5, 10, 25, 50):
            if nb <= K:
                break
            per = []
            kalan_t = kalan_s = 0
            for t in tohumlar:
                d = lgy[msk] - tahmin[t][msk]
                katki = np.bincount(bi, d, minlength=nb) / np.maximum(
                    np.bincount(bi, minlength=nb), 1
                )
                at = np.argsort(-np.abs(katki))[:K]
                tut = ~np.isin(bi, at) if K else np.ones(len(d), bool)
                per.append(float(d[tut].mean()))
                kalan_t, kalan_s = nb - K, int(tut.sum())
            v = np.array(per)
            print(
                f"    {K:>4} {v.mean():>+9.4f} "
                f"{v.std(ddof=1) / np.sqrt(len(v)):>8.4f} {kalan_t:>12,} {kalan_s:>12,}"
            )

    # ---- test karisimina agirliklandir
    print("\n" + "=" * 92)
    print("4. TEST KARISIMINA AGIRLIKLANDIRILMIS b_soguk")
    print("=" * 92)
    gecerli = {k: v for k, v in b_sinif.items() if not np.isnan(v)}
    b_kis = float((lgy - genel).mean())
    pay = sum(agirlik.get(k, 0) * v for k, v in gecerli.items())
    w = sum(agirlik.get(k, 0) for k in gecerli)
    print(f"\n  kis26'nin KENDI karisimiyla   b = {b_kis:+.4f}")
    if w > 0:
        b_test = pay / w
        print(f"  TEST karisimiyla              b = {b_test:+.4f}  (agirlik toplami {w:.3f})")
        print(f"  ORAN test/kis26               {b_test / max(abs(b_kis), 1e-9):.3f}")
        print("\n  On kayitli b_soguk = 0,16 (SOGUK FAZLASI +0,1118 uzerinden)")
        olcek = b_test / max(b_kis, 1e-9)
        print(f"  Ayni olcekle duzeltilmis      b_soguk ~ {0.16 * olcek:.4f}")
        for b_ger in (0.16 * olcek, 0.16, 0.20, 0.25):
            k16 = 2 * P_SOGUK * 0.16 * b_ger - P_SOGUK * 0.16**2
            kopt = P_SOGUK * b_ger**2
            print(
                f"    gercek b={b_ger:.3f} -> delta=0,16 kazanci {k16:+.5f} | "
                f"optimum {-kopt:+.5f} | kacan {kopt - abs(k16) if k16 < 0 else kopt + k16:+.5f}"
            )
    print("\n" + "=" * 92)
    print("HUKUM")
    print("=" * 92)
    print("  ORAN ~1 ise b_soguk nufustan bagimsiz -> 0,16'da KAL.")
    print("  ORAN belirgin farkliysa S3'un delta'si duzeltilir.")
    print("  TOPLU sinifi olculemiyorsa gradyan (bolum 2) yon verir; o da")
    print("  belirsizse 0,16'da kalinir ve gerekcesi yazilir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
