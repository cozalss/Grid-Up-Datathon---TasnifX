# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""KUVVET (power) ORTALAMASI EKSENI -- dusmanca sinama.

IDDIA: p=0 (uretim) optimumda, eksen kapali.
BU BETIK IDDIAYI KIRMAYA CALISIR. Uc kol:

  --adim sicak   aile ekseni p taramasi (yeniden uretim) + KIRPILMIS tablo
                 + seviye/uyusmazlik ayrisimi (kazanc sabit kayma mi?)
  --adim tohum   TOHUM ekseni q taramasi -- rigde HIC olculmedi
                 (deney_ileri.torba_deneyi her zaman np.mean kullaniyor)
  --adim soguk   kis26 soguk tarafta TOHUM ekseni (aile ekseni orada
                 tanimsiz: soguk harman = yalniz cat)

Onbellekten okur, FIT YOK.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402

ONB = KOK / "data" / "interim" / "aile_onbellek"
SOG = KOK / "data" / "interim" / "soguk_temiz"
TOHUMLAR = (1000, 1001, 1002)
AGIRLIK = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 1.4}
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907
SOGUK_KATSAYI = 0.2216 * 1.82133 / 1.07907
BETA = 0.60


def kuvvet_ort(loglar, agir, p: float) -> np.ndarray:
    """M_p uzerinde (1+tahmin); girdi ve cikti LOG1P uzayinda.

    p->0 : agirlikli aritmetik log ortalama (URETIM).
    """
    top = float(sum(agir))
    if abs(p) < 1e-12:
        return sum(w * L for w, L in zip(agir, loglar)) / top
    yig = np.stack([p * L + np.log(w / top) for w, L in zip(agir, loglar)])
    m = yig.max(axis=0)
    return (m + np.log(np.exp(yig - m).sum(axis=0))) / p


def sicak_veri():
    egitim, test = d.cerceveleri_kur()
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]
    veri = {}
    for b in tm.BLOKLAR:
        _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        sicak = ~soguk
        dg = dogrulama[sicak]
        w, tani = ol.test_agirliklari(dg, te_s, guc_kenar, eksenler=("bayatlik",))
        veri[b.ad] = {
            "y": np.load(ONB / f"{b.ad}_gercek.npy").astype("float64"),
            "w": w,
            "tanim": dg["tanim"].astype(str).to_numpy(),
            "ess": tani["ess_orani"],
            "n": int(sicak.sum()),
        }
        for t in TOHUMLAR:
            for a in AGIRLIK:
                veri[b.ad][(t, a)] = np.load(ONB / f"{b.ad}_{t}_{a}_uretim.npy").astype("float64")
        print(f"  {b.ad}  sicak {veri[b.ad]['n']:,}  ESS %{100 * tani['ess_orani']:.1f}")
    return veri


def _skor(v, log_t):
    return ol.agirlikli_rmsle(v["y"], np.clip(np.expm1(log_t), 0.0, None), v["w"])


def _t_tablo(fark):
    ort = float(fark.mean())
    sh = float(fark.std(ddof=1) / np.sqrt(len(fark))) if len(fark) > 1 else 0.0
    return ort, sh, (ort / sh if sh > 0 else 0.0), int((fark > 0).sum())


def adim_sicak(veri) -> None:
    ciftler = [(b.ad, t) for b in tm.BLOKLAR for t in TOHUMLAR]
    aileler = list(AGIRLIK)
    agir = [AGIRLIK[a] for a in aileler]

    def harman(bad, t, p):
        return kuvvet_ort([veri[bad][(t, a)] for a in aileler], agir, p)

    b0, t0 = ciftler[0]
    uretim = sum(AGIRLIK[a] * veri[b0][(t0, a)] for a in aileler) / sum(agir)
    fark0 = float(np.abs(harman(b0, t0, 0.0) - uretim).max())
    print(f"\n  OZDESLIK p=0 vs uretim: max|fark| = {fark0:.3e}")

    taban = {c: _skor(veri[c[0]], harman(*c, 0.0)) for c in ciftler}
    print(f"\n  {'p':>7}{'fark':>11}{'SH':>9}{'t':>7}{'tohum':>7}   bloklar (yaz/guz/kis)")
    for p in (-1.0, -0.25, -0.10, -0.05, 0.0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.50, 1.0):
        s = {c: _skor(veri[c[0]], harman(*c, p)) for c in ciftler}
        f = np.array([taban[c] - s[c] for c in ciftler])
        ort, sh, td, kaz = _t_tablo(f)
        blok = "  ".join(
            f"{np.mean([taban[(b.ad, t)] - s[(b.ad, t)] for t in TOHUMLAR]):+.5f}"
            for b in tm.BLOKLAR
        )
        print(f"  {p:+7.2f}{ort:+11.5f}{sh:9.5f}{td:+7.2f}{kaz:5d}/9   {blok}")

    print("\n  --- p=+0,10 kazancinin ayrisimi (sabit kayma vs varyansla orantili) ---")
    p = 0.10
    for ad in ("kuvvet", "sabit", "varyans"):
        s = {}
        for bad, t in ciftler:
            v = veri[bad]
            L = [v[(t, a)] for a in aileler]
            m = sum(w * x for w, x in zip(agir, L)) / sum(agir)
            if ad == "kuvvet":
                yy = harman(bad, t, p)
            else:
                var = sum(w * (x - m) ** 2 for w, x in zip(agir, L)) / sum(agir)
                delta = (p / 2.0) * var
                yy = m + (float(delta.mean()) if ad == "sabit" else delta)
            s[(bad, t)] = _skor(v, yy)
        f = np.array([taban[c] - s[c] for c in ciftler])
        ort, sh, td, kaz = _t_tablo(f)
        print(f"    {ad:9}{ort:+11.5f}{sh:9.5f}{td:+7.2f}{kaz:5d}/9")

    for p in (0.05, 0.10):
        print(f"\n  --- KIRPILMIS (p={p:+.2f}): en buyuk K trafo atilarak ---")
        for b in tm.BLOKLAR:
            v = veri[b.ad]
            g = np.log1p(np.clip(v["y"], 0, None))
            dm = np.zeros(len(g))
            for t in TOHUMLAR:
                dm += v["w"] * ((g - harman(b.ad, t, 0.0)) ** 2 - (g - harman(b.ad, t, p)) ** 2)
            s_tr = pd.Series(dm).groupby(v["tanim"]).sum().sort_values(ascending=False)
            top = float(s_tr.sum())
            eb = 100 * s_tr.iloc[0] / top if top != 0 else float("nan")
            i5 = 100 * s_tr.iloc[:5].sum() / top if top != 0 else float("nan")
            print(
                f"    {b.ad}  trafo {len(s_tr):,}  toplam d(MSE) {top:+.1f}  "
                f"en buyuk %{eb:.1f}  ilk5 %{i5:.1f}"
            )
            veri[b.ad][f"sira{p}"] = s_tr
        print(f"    {'K':>4}{'fark':>11}{'SH':>9}{'t':>7}{'tohum':>7}{'genele':>10}")
        for K in (0, 1, 5, 10, 25, 50):
            f = []
            for bad, t in ciftler:
                v = veri[bad]
                at = set(v[f"sira{p}"].index[:K])
                msk = ~pd.Series(v["tanim"]).isin(at).to_numpy()
                yy, ww = v["y"][msk], v["w"][msk]
                a0 = ol.agirlikli_rmsle(
                    yy, np.clip(np.expm1(harman(bad, t, 0.0)[msk]), 0, None), ww
                )
                a1 = ol.agirlikli_rmsle(yy, np.clip(np.expm1(harman(bad, t, p)[msk]), 0, None), ww)
                f.append(a0 - a1)
            f = np.array(f)
            ort, sh, td, kaz = _t_tablo(f)
            print(
                f"    {K:>4}{ort:+11.5f}{sh:9.5f}{td:+7.2f}{kaz:5d}/9{-ort * SICAK_KATSAYI:+10.5f}"
            )


def adim_tohum(veri) -> None:
    """TOHUM ekseni: uc tohumu log-aritmetik yerine M_q ile birlestir."""
    aileler = list(AGIRLIK)
    agir = [AGIRLIK[a] for a in aileler]
    print("\n  TOHUM EKSENI (aile harmani p=0 sabit, tohumlar M_q ile birlestiriliyor)")
    ic = {}
    for b in tm.BLOKLAR:
        v = veri[b.ad]
        ic[b.ad] = [kuvvet_ort([v[(t, a)] for a in aileler], agir, 0.0) for t in TOHUMLAR]
        sd = np.stack(ic[b.ad]).std(axis=0)
        print(
            f"    {b.ad}  tohum yayilmasi (log birimi): std ort {sd.mean():.5f}  "
            f"p90 {np.percentile(sd, 90):.5f}  maks {sd.max():.5f}"
        )
    taban = {b.ad: _skor(veri[b.ad], np.mean(ic[b.ad], axis=0)) for b in tm.BLOKLAR}
    print(f"\n  {'q':>7}{'fark(ort)':>12}{'SH':>9}{'t':>7}{'blok':>7}   bloklar")
    for q in (-4.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0):
        f = np.array(
            [
                taban[b.ad] - _skor(veri[b.ad], kuvvet_ort(ic[b.ad], [1.0] * 3, q))
                for b in tm.BLOKLAR
            ]
        )
        ort, sh, td, kaz = _t_tablo(f)
        blok = "  ".join(f"{x:+.5f}" for x in f)
        print(f"  {q:+7.2f}{ort:+12.5f}{sh:9.5f}{td:+7.2f}{kaz:5d}/3   {blok}")


def adim_soguk() -> None:
    """kis26 soguk: tohum ekseni, son islem sonrasi, kVA agirlikli."""
    egitim, test = d.cerceveleri_kur()
    _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, "kis26")
    dg = dogrulama[soguk]
    y = gercek[soguk]
    log_guc = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    tanim = dg["tanim"].astype(str).to_numpy()
    te_c = test[test["soguk_mu"] == 1]
    w, tani = ol.test_agirliklari(dg, te_c, ol.guc_kenarlari(te_c), eksenler=("guc",))
    print(
        f"  kis26 soguk {len(y):,} satir  trafo {pd.unique(tanim).size:,}  "
        f"ESS %{100 * tani['ess_orani']:.1f}"
    )
    tah = [np.load(SOG / f"kis26_{t}_taban.npy").astype("float64") for t in TOHUMLAR]
    sd = np.stack(tah).std(axis=0)
    print(
        f"  tohum yayilmasi (log birimi): std ort {sd.mean():.5f}  "
        f"p90 {np.percentile(sd, 90):.5f}  maks {sd.max():.5f}"
    )

    def buz(log_t):
        r = log_t - log_guc
        return np.clip(np.expm1(r.mean() + BETA * (r - r.mean()) + log_guc), 0.0, None)

    taban_l = kuvvet_ort(tah, [1.0] * 3, 0.0)
    taban = ol.agirlikli_rmsle(y, buz(taban_l), w)
    print(f"\n  URETIM (q=0) kis26 soguk = {taban:.5f}")
    print(f"  {'q':>7}{'skor':>11}{'fark':>11}{'genele':>10}")
    en = []
    for q in (-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0):
        s = ol.agirlikli_rmsle(y, buz(kuvvet_ort(tah, [1.0] * 3, q)), w)
        print(f"  {q:+7.2f}{s:11.5f}{taban - s:+11.5f}{-(taban - s) * SOGUK_KATSAYI:+10.5f}")
        en.append((taban - s, q))
    kaz, q = max(en)
    if kaz <= 0:
        print(f"\n  q kazanci YOK (en iyi q={q:+.2f}, {kaz:+.5f}) -- trafo ayrismasi gereksiz")
        return
    print(f"\n  --- KIRPILMIS (q={q:+.2f}) ---")
    g = np.log1p(np.clip(y, 0, None))
    b0 = buz(taban_l)
    b1 = buz(kuvvet_ort(tah, [1.0] * 3, q))
    dm = w * ((g - np.log1p(b0)) ** 2 - (g - np.log1p(b1)) ** 2)
    s_tr = pd.Series(dm).groupby(tanim).sum().sort_values(ascending=False)
    top = float(s_tr.sum())
    print(
        f"    toplam d(MSE) {top:+.1f}   en buyuk %{100 * s_tr.iloc[0] / top:.1f}   "
        f"ilk5 %{100 * s_tr.iloc[:5].sum() / top:.1f}"
    )
    for K in (0, 1, 5, 10, 25, 50):
        at = set(s_tr.index[:K])
        msk = ~pd.Series(tanim).isin(at).to_numpy()
        f = ol.agirlikli_rmsle(y[msk], b0[msk], w[msk]) - ol.agirlikli_rmsle(
            y[msk], b1[msk], w[msk]
        )
        print(f"    K={K:<4}{f:+11.5f}{-f * SOGUK_KATSAYI:+10.5f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adim", default="sicak")
    ar = ap.parse_args()
    if ar.adim == "soguk":
        adim_soguk()
        return 0
    veri = sicak_veri()
    if ar.adim in ("sicak", "hepsi"):
        adim_sicak(veri)
    if ar.adim in ("tohum", "hepsi"):
        adim_tohum(veri)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
