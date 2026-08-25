# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""SICAK: MEVSIMSEL YAKINLIK AGIRLIKLANDIRMASI (egitim agirligi, kolon DEGIL).

GEREKCE (bu gece olculdu, deney_sicak_artik7/8/9.py):
  * Sicak hatanin oznitelik->hata haritasi MEVSIME BAGLI: blok ICINDE (ayni
    mevsim, gorulmemis trafo + gorulmemis gun) ikinci asama 3/3 kazaniyor
    (-0,009 / -0,029 / -0,038 @a=0,25); bloklar ARASI 3/3 cokuyor
    (+0,027 / +0,052 / +0,008 @a=0,25).
  * Yani eksik olan kolon degil, MEVSIM KOSULLAMASI. Uretim 12 ayi havuzluyor
    ve `tk_` YALIN_CIKARILAN ile atildigi icin takvim konumuna kosullanamiyor.

Kolon eklemek yerine EGITIM AGIRLIGI: hedef pencereye takvimsel olarak yakin
egitim satirlari agirlikli. Kolon eklemedigi icin doluluk deseni sorunu YOK --
agirligin girdisi yalnizca `tarih` (egitimde %100, testte %100).

Taban BEDAVA: onbellekteki `{blok}_{tohum}_cat_uretim.npy`.
Yalnizca agirlikli kol egitilir. Eslenik (blok, tohum) ciftleri.

    python scripts/deney_sicak_mevsim_agirlik.py --tau 45 --tohumlar 1000,1001,1002
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
KAYIT = KOK / "experiments" / "sicak_mevsim_agirlik.jsonl"
SICAK_CAT = {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}
MASKE = 0.15


def dairesel_uzaklik(tarih: pd.Series, hedef_gunler: np.ndarray) -> np.ndarray:
    doy = pd.to_datetime(tarih).dt.dayofyear.to_numpy()
    h = np.unique(hedef_gunler)
    fark = np.abs(doy[:, None] - h[None, :])
    return np.minimum(fark, 365 - fark).min(axis=1).astype("float64")


def cat_agirlikli(parca, dogrulama, kolonlar, tohum, w):  # noqa: ANN001
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
        "--kip", default="mevsim", choices=("mevsim", "guncel", "ayni_ay", "ters_ay", "rastgele")
    )
    ap.add_argument("--bloklar", default="")
    ar = ap.parse_args()
    tohumlar = tuple(int(x) for x in ar.tohumlar.split(","))
    t0 = time.time()

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    gk = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]
    print(f"  {len(kol)} kolon  tau={ar.tau:.0f}  tohum {tohumlar}")

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
        if ar.kip == "mevsim":
            u = dairesel_uzaklik(parca["tarih"], hd.dt.dayofyear.unique())
            w = np.exp(-u / ar.tau)
        elif ar.kip == "guncel":
            pt = pd.to_datetime(parca["tarih"])
            u = np.minimum(np.abs((pt - hd.min()).dt.days), np.abs((pt - hd.max()).dt.days))
            u = np.where((pt >= hd.min()) & (pt <= hd.max()), 0, u).astype("float64")
            w = np.exp(-u / ar.tau)
        elif ar.kip == "ters_ay":  # KONTROL: hedefin TAM KARSI aylari yukseltilir
            aylar = {((m + 5) % 12) + 1 for m in hd.dt.month.unique()}
            u = (~pd.to_datetime(parca["tarih"]).dt.month.isin(aylar)).to_numpy().astype("float64")
            w = np.where(u == 0, ar.tau, 1.0)
        elif ar.kip == "rastgele":  # KONTROL: rastgele %9,4 satir yukseltilir
            rg = np.random.default_rng(5)
            u = (rg.random(len(parca)) >= 0.094).astype("float64")
            w = np.where(u == 0, ar.tau, 1.0)
        else:  # ayni_ay: yalnizca hedef TAKVIM AYLARINDAKI satirlar yukseltilir
            aylar = set(hd.dt.month.unique())
            u = (~pd.to_datetime(parca["tarih"]).dt.month.isin(aylar)).to_numpy().astype("float64")
            w = np.where(u == 0, ar.tau, 1.0)
        w = np.asarray(w, dtype="float64")
        w = w / w.mean()
        print(
            f"  {b.ad}: egitim {len(parca):,}  uzaklik medyan {np.median(u):.0f}g"
            f"  agirlik p10={np.quantile(w, 0.1):.3f} p90={np.quantile(w, 0.9):.3f}"
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
                f"    {b.ad} t{tohum}  taban {s0:.5f}  agirlikli {s1:.5f}"
                f"  fark {s0 - s1:+.5f}   ({time.time() - t1:.0f} sn)"
            )
    f = np.array(farklar)
    sh = float(f.std(ddof=1) / np.sqrt(len(f)))
    print(
        f"\n  HAVUZLANMIS fark {f.mean():+.5f}  SH {sh:.5f}  t={f.mean() / sh:+.2f}"
        f"  pozitif {int((f > 0).sum())}/{len(f)}"
    )
    for b in [x for x in tm.BLOKLAR if x.ad in secilen]:
        bf = np.array([x[4] for x in satirlar if x[0] == b.ad])
        print(f"    {b.ad}: {bf.mean():+.5f}  ({int((bf > 0).sum())}/{len(bf)})")
    KAYIT.parent.mkdir(exist_ok=True)
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
