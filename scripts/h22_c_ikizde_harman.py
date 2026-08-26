"""H22 -- SOGUK GUN EKSENI c*'i IKIZDE, URETIM HARMANIYLA yeniden capala.

NEDEN
-----
H8/H9b'nin c'si ``gun_ekseni/*_taban.npy`` uzerinden turetildi ve tik 7'de
ortaya cikti ki o dosyalar YALNIZ ``cat`` (deney_gun_ekseni_dogrula.py:121).
Uretim harmani cat/xgb/lgbm = 3/1/1. Seviye olcumlerinde bu %44'luk fark
yaratmisti (b: +0,0595 vs +0,1056).

Gun ekseni GENLIGI icin aile farkinin sadelesmesi BEKLENIR (genlik, seviyeden
farkli olarak ortak bir olcek) ama VARSAYIM DEGIL OLCUM ister. Bu betik
tik 7'de uretilen ``soguk_tahmin_yaz25.npz`` / ``guz25.npz`` ile ayni
olcumu URETIM HARMANIYLA tekrarlar.

S2 saat 03:07'de c=1,60 ile gidiyor. Bu olcum onu degistirebilir.

YONTEM (h8h/h8i ile ayni, SEVIYE-NOTR)
--------------------------------------
    b_gun = iki yonlu ayristirmadan gun bileseni (kural 6: trafo etkisi cikarilir)
    b_c   = satir-agirlikli merkezlenmis  ->  mudahale seviyeyi DEGISTIRMEZ
    c*    = 1 + <r, b_c> / <b_c, b_c>      (kesin MSE optimumu)
Ayrica ETIKETSIZ CAPA (kural 5: test etiketi kullanilmaz) ve capanin
etiketli optimumu uretip uretmedigi (kalibrasyon orani).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
HARMAN = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0}
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


def blok(blok_ad: str):
    npz = KOK / f"data/interim/deney/soguk_tahmin_{blok_ad}.npz"
    meta = KOK / f"data/interim/{blok_ad}_soguk_meta.parquet"
    if not npz.exists():
        print(f"{blok_ad}: onbellek yok\n")
        return
    z = np.load(npz)
    m = pd.read_parquet(meta).reset_index(drop=True)
    tohum = sorted({k.split("_")[0] for k in z.files})
    pay = sum(HARMAN.values())
    tah = {
        t: sum(HARMAN[a] * z[f"{t}_{a}"].astype("float64") for a in HARMAN) / pay
        for t in tohum
        if all(f"{t}_{a}" in z.files for a in HARMAN)
    }

    lgy = np.log1p(np.clip(m["y"].to_numpy(dtype="float64"), 0, None))
    # T3 temiz alt panel
    ilk = m.groupby("tanim")["tarih"].transform("min")
    yas = (m["tarih"] - ilk).dt.days.to_numpy()
    say = m.groupby("tanim")["tanim"].transform("size").to_numpy()
    for etiket, msk in (
        ("T0 ham", np.ones(len(m), bool)),
        ("T3 temiz", (yas >= MIN_YAS) & (say >= MIN_GUN)),
    ):
        a = m.loc[msk].reset_index(drop=True)
        y2 = lgy[msk]
        bi, _ = pd.factorize(a["tanim"])
        gi, _ = pd.factorize(a["tarih"])
        nb, ng = int(bi.max()) + 1, int(gi.max()) + 1
        n_d = np.bincount(gi, minlength=ng).astype(float)
        bg = merkezle(iki_yonlu(y2, bi, gi, nb, ng), n_d)

        cs, ds, capalar = [], [], []
        for t, v in tah.items():
            pr = v[msk]
            bm = merkezle(iki_yonlu(pr, bi, gi, nb, ng), n_d)
            r = y2 - pr
            rg = np.bincount(gi, r, minlength=ng)
            payda = float(np.dot(n_d, bm**2))
            c = 1.0 + float(np.dot(rg, bm)) / payda
            mse0 = float((r**2).mean())
            ds.append(float(((r - (c - 1) * bm[gi]) ** 2).mean()) - mse0)
            cs.append(c)
            w = n_d / n_d.sum()
            kor = float(np.sum(w * bg * bm) / np.sqrt(np.sum(w * bg**2) * np.sum(w * bm**2)))
            capalar.append(kor * float(np.sqrt(np.sum(w * bg**2) / np.sum(w * bm**2))))
        v1, v2 = np.array(cs), np.array(ds)
        print(
            f"  {blok_ad} {etiket:<10} n={int(msk.sum()):>7,} trafo={a.tanim.nunique():>5} "
            f"| c*(etiketli) {v1.mean():6.3f} (std {v1.std(ddof=1):.3f}) "
            f"| c_capa {np.mean(capalar):6.3f} "
            f"| oran capa/etiketli {np.mean(capalar) / v1.mean():.3f} "
            f"| dMSE {v2.mean():+.5f}"
        )
    print()


def main() -> int:
    print("=" * 100)
    print("SOGUK GUN EKSENI c* -- URETIM HARMANI (cat/xgb/lgbm = 3/1/1), SEVIYE-NOTR")
    print("=" * 100)
    print("  (H8/H9b'nin dayandigi taban dosyalari YALNIZ cat idi -- karsilastirma icin)\n")
    for b in ("yaz25", "guz25"):
        blok(b)

    print("=" * 100)
    print("KARSILASTIRMA")
    print("=" * 100)
    print("  H8  (taban=cat, yaz25 T3, etiketli)         c* = 3,127")
    print("  H8  (taban=cat, yaz25 T3, capa)             c  = 2,503")
    print("  H9b (nufus eslesmis capa)                   c  = 1,547")
    print("  h8l (kirpma risk ayari, K=25 argmin)        c  = 1,60   <- SECILEN")
    print("\n  Yukaridaki HARMAN sayilari 1,60'a yakinsa secim dogrulanir.")
    print("  Belirgin farkliysa S2 03:07'den ONCE yeniden uretilir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
