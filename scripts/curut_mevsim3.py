# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""CURUTME 3: mevsim agirligi = TAKVIM mi, yoksa `sub25` KOKENININ agirligi mi?

kis26 egitiminde hedef takvim aylari (2,3) YALNIZCA `sub25` kokeninden gelir
(ozet_pencere_gun=31, hedefin ozeti 334, TESTin 455). mevsim agirligi bu kokeni
%9,4 -> %28,2'ye cikariyor (3,01 kat) -- her uc blok icindeki EN BUYUK koken
kaymasi. Uretimde ise `sub25` yalnizca 1,18 kat aliyor.

Kipler:
  mevsim         : uretim esli taban kip (exp(-u/tau))
  sub25notr      : mevsim, ama `sub25`in TOPLAM agirlik payi dogal payina
                   sabitlenir (takvim yakinligi korunur, koken kaymasi silinir)
  sub25x3        : TAKVIM YOK. Yalnizca `sub25` 3,01 kat; digerleri duz.
  sub25sifir     : `sub25` tamamen atilir (agirlik 0) -- ust sinir kontrolu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import deney as d
import deney_ileri as di
import olcut as ol
import tuketim_model as tm  # noqa: E402

ONB = KOK / "data" / "interim" / "aile_onbellek"
KAYIT = KOK / "experiments" / "curut_mevsim.jsonl"
SICAK_CAT = {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}
MASKE = 0.15


def dairesel_uzaklik(tarih, hedef_gunler):
    doy = pd.to_datetime(tarih).dt.dayofyear.to_numpy()
    h = np.unique(hedef_gunler)
    f = np.abs(doy[:, None] - h[None, :])
    return np.minimum(f, 365 - f).min(axis=1).astype("float64")


def cat_agirlikli(parca, dogrulama, kolonlar, tohum, w):
    maskeli = tm.soguk_maskele(parca, kolonlar, tohum, MASKE)
    y = tm.ofsetli_hedef(maskeli)
    model = tm.aile_modeli("cat", tohum, hizli=False, cat_ustyazim=SICAK_CAT)
    xe, xh = maskeli[kolonlar].copy(), dogrulama[kolonlar].copy()
    kat = [k for k in tm.KATEGORIK if k in xe.columns]
    for k in kat:
        xe[k] = xe[k].astype(str)
        xh[k] = xh[k].astype(str)
    model.fit(xe, y, cat_features=kat, sample_weight=w)
    return tm.ofseti_geri_ekle(model.predict(xh), dogrulama)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=45.0)
    ap.add_argument("--tohumlar", default="1000,1001,1002")
    ap.add_argument(
        "--kip", default="sub25notr", choices=("mevsim", "sub25notr", "sub25x3", "sub25sifir")
    )
    ap.add_argument("--bloklar", default="kis26")
    ar = ap.parse_args()
    tohumlar = tuple(int(x) for x in ar.tohumlar.split(","))
    t0 = time.time()

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    gk = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]

    ek = d._ek_kokenler_kur(False)
    ek = ek[ek["_blok"].isin([a for a, _, _ in tm.EK_KOKENLER])]
    ortak = [k for k in egitim.columns if k in ek.columns]
    genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
    for k in tm.KATEGORIK:
        genis[k] = pd.Categorical(genis[k], categories=egitim[k].cat.categories)

    farklar, satirlar = [], []
    secilen = [x for x in ar.bloklar.split(",") if x] or [b.ad for b in tm.BLOKLAR]
    for b in [x for x in tm.BLOKLAR if x.ad in secilen]:
        _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        sicak = ~soguk
        dg = dogrulama[sicak]
        y = gercek[sicak]
        parca = tm.kokenleri_ayikla(genis, b.ad)
        hd = pd.to_datetime(dg["tarih"])
        sub = parca["_blok"].to_numpy() == "sub25"
        u = dairesel_uzaklik(parca["tarih"], hd.dt.dayofyear.unique())
        w_mev = np.exp(-u / ar.tau)
        if ar.kip == "mevsim":
            w = w_mev
        elif ar.kip == "sub25notr":
            w = w_mev.copy()
            # sub25'in TOPLAM agirlik payi dogal satir payina esitlenir
            dogal = sub.mean()
            hedef_top = dogal / (1 - dogal) * w[~sub].sum()
            w[sub] = w[sub] * (hedef_top / w[sub].sum())
        elif ar.kip == "sub25x3":
            w = np.ones(len(parca))
            kat = (0.2822 / 0.7178) / (sub.mean() / (1 - sub.mean()))
            w[sub] = kat
        else:  # sub25sifir
            w = np.ones(len(parca))
            w[sub] = 0.0
        w = np.asarray(w, dtype="float64")
        w = w / w.mean()
        pay = 100 * w[sub].sum() / w.sum()
        print(
            f"  {b.ad} {ar.kip}: egitim {len(parca):,}  sub25 satir payi %{100 * sub.mean():.1f}"
            f"  AGIRLIK payi %{pay:.1f}  ESS %{100 * (w.sum() ** 2 / (w**2).sum()) / len(w):.1f}"
        )
        wt, tani = ol.test_agirliklari(dg, te_s, gk)
        for tohum in tohumlar:
            taban_log = np.load(ONB / f"{b.ad}_{tohum}_cat_uretim.npy").astype("float64")
            t1 = time.time()
            yeni_log = cat_agirlikli(parca, dg, kol, tohum, w)
            s0 = ol.agirlikli_rmsle(y, np.expm1(taban_log), wt)
            s1 = ol.agirlikli_rmsle(y, np.expm1(yeni_log), wt)
            farklar.append(s0 - s1)
            satirlar.append((b.ad, tohum, s0, s1, s0 - s1))
            print(
                f"    {b.ad} t{tohum}  taban {s0:.5f}  yeni {s1:.5f}"
                f"  fark {s0 - s1:+.5f}   ({time.time() - t1:.0f} sn)"
            )
    f = np.array(farklar)
    sh = float(f.std(ddof=1) / np.sqrt(len(f))) if len(f) > 1 else float("nan")
    print(
        f"\n  {ar.kip}: HAVUZLANMIS {f.mean():+.5f}  SH {sh:.5f}  t={f.mean() / sh:+.2f}"
        f"  pozitif {int((f > 0).sum())}/{len(f)}"
    )
    with KAYIT.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "kip": ar.kip,
                    "tau": ar.tau,
                    "fark": float(f.mean()),
                    "sh": sh,
                    "t": float(f.mean() / sh),
                    "satirlar": satirlar,
                }
            )
            + "\n"
        )
    print(f"  {(time.time() - t0) / 60:.1f} dk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
