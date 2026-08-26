"""H8h -- OLCULEN SOGUK KAZANC SEVIYE MI, GENLIK MI?

YAKALANAN KONFAUND
------------------
h8c/h8e'de gun bileseni ``b`` GUNLER uzerinde ortalamasi 0 olacak sekilde
normalize edildi (``b -= b.mean()``), sonra SATIRLARA ``(c-1)*b[g_i]`` diye
yazildi. Panel DENGESIZ oldugu icin (trafolar pencere boyunca dogar, gec
gunlerde cok daha fazla satir var) satir-agirlikli ortalama SIFIR DEGIL.

Uretim betigi bunu kapida yakaladi: c=2,2 uygulanınca soguk satirlarin
ortalama log1p kaymasi **+0,0714** -- yani mudahale gizlice bir SEVIYE
KAYMASI tasiyor. Seviye ise AYRI bir knob ve LB problariyla cozuluyor
(b_soguk ~ +0,16). Ikisi karisirsa iki olcum de bozulur.

O halde su soru KRITIK:
    olculen -0,0553'un ne kadari GENLIK, ne kadari SEVIYE?

Genlik kaybolursa bulgu CURUKTUR -- "gun ekseni genligi" diye raporlanan sey
aslinda zaten bilinen soguk seviye acigidir ve H8 yeni bir sey EKLEMEZ.

YONTEM
------
Gun bilesenini SATIR-AGIRLIKLI merkezle:
    b_c = b - (SUM_d n_d * b_d) / (SUM_d n_d)
Bu durumda SUM_i b_c[g_i] = 0, yani (c-1)*b_c mudahalesi seviyeyi
TANIM GEREGI degistirmez.

Uc mudahale ayri ayri ve birlikte olculur:
    (S) yalniz seviye:   +delta            (optimal delta = ort artik)
    (G) yalniz genlik:   (c-1)*b_c         (optimal c, seviye-notr)
    (SG) ikisi birlikte  -- ortogonal olduklari icin dMSE toplanmali
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


def olc(ad: str, etiket: str, mask: np.ndarray | None = None) -> None:
    m = pd.read_parquet(ONBELLEK / f"{ad}_meta.parquet").reset_index(drop=True)
    if mask is None:
        mask = np.ones(len(m), bool)
    a = m.loc[mask].reset_index(drop=True)
    lgy = np.log1p(np.clip(a["y"].to_numpy(dtype="float64"), 0, None))
    bi, _ = pd.factorize(a["tanim"])
    gi, _ = pd.factorize(a["tarih"])
    nb, ng = int(bi.max()) + 1, int(gi.max()) + 1
    n_d = np.bincount(gi, minlength=ng).astype(float)
    N = float(n_d.sum())

    sat = []
    for p in sorted(ONBELLEK.glob(f"{ad}_*_taban.npy")):
        pr = np.load(p).astype("float64")[mask]
        b = iki_yonlu(pr, bi, gi, nb, ng)
        bc = b - float(np.dot(n_d, b) / N)  # SATIR-agirlikli merkez
        r = lgy - pr
        mse0 = float((r**2).mean())

        # (S) yalniz seviye
        delta = float(r.mean())
        d_s = float(((r - delta) ** 2).mean()) - mse0

        # (G) yalniz genlik, seviye-notr
        rg = np.bincount(gi, r, minlength=ng)
        pay = float(np.dot(rg, bc))
        payda = float(np.dot(n_d, bc**2))
        c_g = 1.0 + pay / payda
        d_g = float(((r - (c_g - 1) * bc[gi]) ** 2).mean()) - mse0

        # (SG) birlikte
        d_sg = float(((r - delta - (c_g - 1) * bc[gi]) ** 2).mean()) - mse0

        # KARSILASTIRMA: eski (gun-merkezli, seviye sizdiran) yontem
        pay2 = float(np.dot(rg, b))
        payda2 = float(np.dot(n_d, b**2))
        c_eski = 1.0 + pay2 / payda2
        d_eski = float(((r - (c_eski - 1) * b[gi]) ** 2).mean()) - mse0
        kayma = (c_eski - 1) * float(np.dot(n_d, b) / N)

        sat.append(
            {
                "tohum": p.stem.split("_")[1],
                "artik_ort": delta,
                "dMSE_SEVIYE": d_s,
                "c_genlik": c_g,
                "dMSE_GENLIK": d_g,
                "dMSE_BIRLIKTE": d_sg,
                "c_eski": c_eski,
                "dMSE_eski": d_eski,
                "gizli_kayma": kayma,
            }
        )
    d = pd.DataFrame(sat)
    print(
        f"\n--- {etiket}   {int(mask.sum()):,} satir, {a.tanim.nunique()} trafo, "
        f"{ng} gun, {len(d)} tohum"
    )
    print(d.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
    for k in ("dMSE_SEVIYE", "dMSE_GENLIK", "dMSE_BIRLIKTE", "dMSE_eski"):
        v = d[k].to_numpy()
        sh = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        t = v.mean() / sh if sh and sh > 0 else float("nan")
        print(
            f"  {k:16s} ort {v.mean():+.5f}  SH {sh:.5f}  t {t:+7.2f}  "
            f"poz {int((v < 0).sum())}/{len(v)}   -> test etkisi "
            f"{P_SOGUK * v.mean():+.6f}"
        )
    g = d["dMSE_GENLIK"].mean()
    s = d["dMSE_SEVIYE"].mean()
    print(f"  >>> GENLIK payi {g / (g + s) * 100:5.1f}%   SEVIYE payi {s / (g + s) * 100:5.1f}%")
    print(
        f"  >>> eski yontemin gizli seviye kaymasi ort "
        f"{d.gizli_kayma.mean():+.4f}  (uretimde kapida yakalandi: +0,0714)"
    )


def main() -> int:
    for ad, adi in (("yaz25", "yaz25 SOGUK IKIZ"), ("guz25", "guz25 SOGUK")):
        m = pd.read_parquet(ONBELLEK / f"{ad}_meta.parquet").reset_index(drop=True)
        ilk = m.groupby("tanim")["tarih"].transform("min")
        yas = (m["tarih"] - ilk).dt.days.to_numpy()
        say = m.groupby("tanim")["tanim"].transform("size").to_numpy()
        print("=" * 96)
        print(adi)
        print("=" * 96)
        olc(ad, f"{adi} T0 ham")
        olc(ad, f"{adi} T3 temiz (ilk7 atildi, >=60 gun)", (yas >= 7) & (say >= 60))
    print("\n" + "=" * 96)
    print("HUKUM")
    print("=" * 96)
    print("  GENLIK payi kucukse (dMSE_GENLIK ~ 0) H8 yeni bir sey EKLEMIYOR:")
    print("  olculen kazanc zaten bilinen SOGUK SEVIYE acigi (b_soguk ~ +0,16),")
    print("  ki o 03:00 gonderim penceresinde LB probuyla cozulecek. -> CURUDU")
    print("  GENLIK payi buyukse H8 seviyeden BAGIMSIZ, EK bir kazanctir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
