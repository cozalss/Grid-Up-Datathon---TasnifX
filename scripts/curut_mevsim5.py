# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""CURUTME 5: mevsim agirligi -- TRAFO BAZINDA yogunlasma + KIRPILMIS tablo.

docs/41 §1 hukmunu veren testin AYNISI: `sicak kapasite` kis26'da t=+4,38 / 3-3
idi ve K=5'te sifira, K=25'te NEGATIFE dondu. mevsim agirligi kis26'da
t=+3,18 / 3-3. Ayni testten gecmeli.

Tahminler diske yazilir (curut_mevsim_tahmin/) -- kesintiye dayanikli.
"""

from __future__ import annotations

import argparse
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
CIK = KOK / "data" / "interim" / "curut_mevsim_tahmin"
SICAK_CAT = {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}
MASKE = 0.15
TOHUMLAR = (1000, 1001, 1002)
SICAK_KATSAYI = 0.536


def dairesel(tarih, hedef_doy):
    doy = pd.to_datetime(tarih).dt.dayofyear.to_numpy()
    h = np.unique(np.asarray(hedef_doy))
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


def agirlikli_mse(e, w):
    return float((w * e * e).sum() / w.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=45.0)
    ap.add_argument("--bloklar", default="kis26")
    ar = ap.parse_args()
    CIK.mkdir(parents=True, exist_ok=True)
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

    secilen = [x for x in ar.bloklar.split(",") if x]
    V = {}
    for b in [x for x in tm.BLOKLAR if x.ad in secilen]:
        _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        sicak = ~soguk
        dg = dogrulama[sicak]
        y = gercek[sicak]
        parca = tm.kokenleri_ayikla(genis, b.ad)
        hd = pd.to_datetime(dg["tarih"])
        u = dairesel(parca["tarih"], hd.dt.dayofyear.unique())
        w = np.exp(-u / ar.tau)
        w = w / w.mean()
        wt, tani = ol.test_agirliklari(dg, te_s, gk)
        g = np.log1p(np.clip(y, 0, None))
        r0, r1 = {}, {}
        for tohum in TOHUMLAR:
            r0[tohum] = np.load(ONB / f"{b.ad}_{tohum}_cat_uretim.npy").astype("float64")
            yol = CIK / f"{b.ad}_{tohum}_mevsim{int(ar.tau)}.npy"
            if yol.exists():
                r1[tohum] = np.load(yol).astype("float64")
                print(f"    {b.ad} t{tohum} onbellekten")
            else:
                t1 = time.time()
                p = cat_agirlikli(parca, dg, kol, tohum, w)
                np.save(yol, p.astype("float32"))
                r1[tohum] = p
                print(f"    {b.ad} t{tohum} egitildi ({time.time() - t1:.0f} sn)")
        V[b.ad] = {"g": g, "w": wt, "r0": r0, "r1": r1, "trafo": dg[tm.GRUP].astype(str).to_numpy()}

    print("\n" + "=" * 96)
    print("YOGUNLASMA (torbalanmis d(MSE), pozitif = mevsim KAZANIYOR)")
    print("=" * 96)
    print(
        f"  {'blok':8}{'trafo':>8}{'toplam d(MSE)':>16}{'EN BUYUK':>11}{'ilk5':>9}{'poz trafo':>11}"
    )
    for ad, v in V.items():
        e0 = v["g"] - np.mean([v["r0"][t] for t in TOHUMLAR], axis=0)
        e1 = v["g"] - np.mean([v["r1"][t] for t in TOHUMLAR], axis=0)
        dmse = v["w"] * (e0 * e0 - e1 * e1)
        pay = pd.Series(dmse).groupby(pd.Series(v["trafo"])).sum().sort_values(ascending=False)
        top = pay.sum()
        print(
            f"  {ad:8}{pay.size:>8,}{top:>16.1f}{100 * pay.iloc[0] / top:>10.1f}%"
            f"{100 * pay.iloc[:5].sum() / top:>8.1f}%{100 * (pay > 0).mean():>10.1f}%"
        )
        print(f"           en buyuk 5 trafo: {list(pay.index[:5])}")

    print("\n" + "=" * 96)
    print("KIRPILMIS HUKUM (en cok KAZANDIRAN K trafo atilarak; blok x tohum eslenik)")
    print("=" * 96)
    print(f"  {'K':>5}{'kalan':>9}{'fark':>11}{'SH':>10}{'t':>8}{'tohum':>8}{'genele':>10}")
    for K in (0, 1, 5, 10, 25, 50, 100):
        f = []
        kalan = 0
        for ad, v in V.items():
            e0 = v["g"] - np.mean([v["r0"][t] for t in TOHUMLAR], axis=0)
            e1 = v["g"] - np.mean([v["r1"][t] for t in TOHUMLAR], axis=0)
            dmse = v["w"] * (e0 * e0 - e1 * e1)
            pay = pd.Series(dmse).groupby(pd.Series(v["trafo"])).sum()
            kotu = set(pay.nlargest(K).index) if K else set()
            tut = ~pd.Series(v["trafo"]).isin(kotu).to_numpy()
            kalan = pay.size - len(kotu)
            for t in TOHUMLAR:
                a = v["g"] - v["r0"][t]
                z = v["g"] - v["r1"][t]
                f.append(
                    np.sqrt(agirlikli_mse(a[tut], v["w"][tut]))
                    - np.sqrt(agirlikli_mse(z[tut], v["w"][tut]))
                )
        fa = np.array(f)
        sh = float(fa.std(ddof=1) / np.sqrt(len(fa)))
        print(
            f"  {K:>5}{kalan:>9,}{fa.mean():>+11.5f}{sh:>10.5f}{fa.mean() / sh:>+8.2f}"
            f"{int((fa > 0).sum()):>5}/{len(fa)}{-fa.mean() * SICAK_KATSAYI:>+10.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
