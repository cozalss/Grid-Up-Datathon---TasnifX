# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""CURUTME 2 -- iddianin ZAYIF NOKTALARINA saldiri.

A) TAVAN gercekten -0,00403 mu? KOSULLU (tabaka bazli) simpleks orakulu daha
   buyuk bir aile; bu aile de SEVIYE degil (her tabakada toplam=1).
B) Blok-disi KOSULLU simpleks pozitif mi? (iddianin "eksenin TAMAMI" hukmu)
C) Metrik dayanikliligi: agirliksiz RMSLE altinda hukum degisiyor mu?
D) AYRIK agirlik secimi (lgbm'siz vb.) blok-disi LOO ile kazaniyor mu?
E) NNLS = c x simpleks ayrisimi: c cikarilinca NNLS'te ne kaliyor?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import deney_nnls_curut as c1  # noqa: E402
import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402

TOHUMLAR = c1.TOHUMLAR
AILELER = c1.AILELER
URETIM_N = c1.URETIM_N
KAT = c1.SICAK_KATSAYI


def a_mse(e, w):  # noqa: ANN001, ANN202
    return float(np.dot(w, e * e) / w.sum())


def hukum(f, ad):  # noqa: ANN001, ANN202
    fa = np.asarray(f, dtype="float64")
    sh = float(fa.std(ddof=1) / np.sqrt(len(fa)))
    t = fa.mean() / sh if sh > 0 else 0.0
    print(
        f"  {ad:42}{fa.mean():+10.5f}{sh:10.5f}{t:+8.2f}"
        f"{int((fa > 0).sum()):>5}/{len(fa)}{-fa.mean() * KAT:+10.5f}"
    )
    return fa.mean(), sh, t


def main() -> int:
    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]

    V = {}
    for b in tm.BLOKLAR:
        _, dogrulama, _, soguk = di.blok_parcalari(egitim, b.ad)
        dg = dogrulama[~soguk]
        w, _ = ol.test_agirliklari(dg, te_s, guc_kenar, eksenler=("bayatlik",))
        y = np.log1p(np.clip(np.load(c1.DIZIN / f"{b.ad}_gercek.npy").astype("float64"), 0.0, None))
        X = {
            t: np.column_stack(
                [
                    np.load(c1.DIZIN / f"{b.ad}_{t}_{a}_uretim.npy").astype("float64")
                    for a in AILELER
                ]
            )
            for t in TOHUMLAR
        }
        tab = {
            "bayatlik": ol._kova(dg["t_son_kayit_yasi"].to_numpy("float64"), ol.BAYATLIK_KENARLARI),
            "ufuk": ol._kova(dg["ufuk_gun"].to_numpy("float64"), ol.UFUK_KENARLARI),
            "guc": ol._kova(np.log1p(dg["guc"].to_numpy("float64")), guc_kenar),
        }
        tab["bayatlik_x_guc"] = tab["bayatlik"] * 16 + tab["guc"]
        V[b.ad] = {"y": y, "w": w, "X": X, "trafo": dg["tanim"].to_numpy(), "tab": tab, "n": len(y)}

    KATLAR = {}
    for b in tm.BLOKLAR:
        u = pd.unique(V[b.ad]["trafo"])
        rng = np.random.default_rng(7)
        KATLAR[b.ad] = (
            pd.Series(rng.integers(0, 5, len(u)), index=u).reindex(V[b.ad]["trafo"]).to_numpy()
        )

    def kosullu(coz, eksen, kip, asgari=400):  # noqa: ANN001, ANN202
        """kip: 'orakul' | 'capraz' | 'blokdisi'. Tabaka basina simpleks."""

        def _p(bad, t, X, y, w):  # noqa: ANN001, ANN202
            s = V[bad]["tab"][eksen] if eksen else np.zeros(len(y), "int64")
            p = X @ URETIM_N
            for kod in np.unique(s):
                m = s == kod
                if kip == "orakul":
                    Xtr, ytr, wtr = X[m], y[m], w[m]
                elif kip == "capraz":
                    continue
                else:
                    par = [
                        (V[o.ad], V[o.ad]["tab"][eksen] == kod if eksen else slice(None))
                        for o in tm.BLOKLAR
                        if o.ad != bad
                    ]
                    Xtr = np.vstack([v["X"][t][q] for v, q in par])
                    ytr = np.concatenate([v["y"][q] for v, q in par])
                    wtr = np.concatenate([v["w"][q] for v, q in par])
                if len(ytr) < asgari:
                    continue
                p[m] = X[m] @ coz(Xtr, ytr, wtr)
            if kip == "capraz":
                for kk in range(5):
                    te = KATLAR[bad] == kk
                    tr = ~te
                    for kod in np.unique(s):
                        m = te & (s == kod)
                        mtr = tr & (s == kod)
                        if m.sum() == 0 or mtr.sum() < asgari:
                            continue
                        p[m] = X[m] @ coz(X[mtr], y[mtr], w[mtr])
            return p

        return _p

    def kos(protokol, agirlikli=True):  # noqa: ANN001, ANN202
        f = []
        for b in tm.BLOKLAR:
            v = V[b.ad]
            for t in TOHUMLAR:
                X, y = v["X"][t], v["y"]
                w = v["w"] if agirlikli else np.ones(v["n"])
                p0 = X @ URETIM_N
                p1 = protokol(b.ad, t, X, y, w)
                f.append(np.sqrt(a_mse(y - p0, w)) - np.sqrt(a_mse(y - p1, w)))
        return f

    print("\n" + "=" * 108)
    print("A/B) KOSULLU (tabaka bazli) SIMPLEKS -- toplam her tabakada 1, yani SEVIYE DEGIL")
    print("=" * 108)
    print(f"  {'PROTOKOL':42}{'fark':>10}{'SH':>10}{'t':>8}{'tohum':>8}{'genele':>10}")
    for eksen in (None, "bayatlik", "ufuk", "guc", "bayatlik_x_guc"):
        ad = eksen or "kuresel"
        for kip in ("orakul", "capraz", "blokdisi"):
            hukum(kos(kosullu(c1.cozum_simpleks, eksen, kip)), f"{kip:9} simpleks / {ad}")
        print()

    print("=" * 108)
    print("C) METRIK DAYANIKLILIGI -- AGIRLIKSIZ RMSLE (olcut kapali)")
    print("=" * 108)
    print(f"  {'PROTOKOL':42}{'fark':>10}{'SH':>10}{'t':>8}{'tohum':>8}{'genele':>10}")
    for ad, prot in (
        ("orakul kuresel simpleks", kosullu(c1.cozum_simpleks, None, "orakul")),
        ("blok-disi kuresel simpleks", kosullu(c1.cozum_simpleks, None, "blokdisi")),
        ("blok-disi bayatlik simpleks", kosullu(c1.cozum_simpleks, "bayatlik", "blokdisi")),
    ):
        hukum(kos(prot, agirlikli=False), ad)

    print("\n" + "=" * 108)
    print("D) AYRIK AGIRLIK ADAYLARI -- her aday SABIT, uydurma yok. LOO-blok gerekmez.")
    print("=" * 108)
    adaylar = {
        "uretim 3/1/1/1,4": (3, 1, 1, 1.4),
        "lgbm YOK 3/1/0/1,4": (3, 1, 0, 1.4),
        "ag YOK 3/1/1/0": (3, 1, 1, 0),
        "esit 1/1/1/1": (1, 1, 1, 1),
        "cat agir 5/1/1/1,4": (5, 1, 1, 1.4),
        "cat hafif 2/1/1/1,4": (2, 1, 1, 1.4),
        "ag agir 3/1/1/2,5": (3, 1, 1, 2.5),
        "xgb agir 3/2/1/1,4": (3, 2, 1, 1.4),
        "yalniz cat": (1, 0, 0, 0),
    }
    print(f"  {'ADAY':42}{'fark':>10}{'SH':>10}{'t':>8}{'tohum':>8}{'genele':>10}")
    for ad, vek in adaylar.items():
        bv = np.array(vek, dtype="float64")
        bv = bv / bv.sum()
        hukum(kos(lambda bad, t, X, y, w, bv=bv: X @ bv), ad)

    print("\n  BLOK BAZINDA (en iyi ayrik adayin blok tutarliligi)")
    print(f"  {'ADAY':28}" + "".join(f"{b.ad:>12}" for b in tm.BLOKLAR))
    for ad, vek in adaylar.items():
        bv = np.array(vek, dtype="float64")
        bv = bv / bv.sum()
        sat = f"  {ad:28}"
        for b in tm.BLOKLAR:
            v = V[b.ad]
            fk = []
            for t in TOHUMLAR:
                X, y, w = v["X"][t], v["y"], v["w"]
                fk.append(np.sqrt(a_mse(y - X @ URETIM_N, w)) - np.sqrt(a_mse(y - X @ bv, w)))
            sat += f"{np.mean(fk):+12.5f}"
        print(sat)

    print("\n" + "=" * 108)
    print("E) NNLS = c x SIMPLEKS AYRISIMI (orakul; c = NNLS agirlik toplami)")
    print("=" * 108)
    print(
        f"  {'blok':8}{'tohum':>7}{'NNLS kaz':>11}{'c-tek kaz':>11}{'simpleks kaz':>14}{'ARTIK':>10}"
    )
    for b in tm.BLOKLAR:
        v = V[b.ad]
        for t in TOHUMLAR:
            X, y, w = v["X"][t], v["y"], v["w"]
            p0 = X @ URETIM_N
            t0 = np.sqrt(a_mse(y - p0, w))
            bn = c1.cozum_nnls(X, y, w)
            bs = c1.cozum_simpleks(X, y, w)
            c = float(np.dot(w, p0 * y) / np.dot(w, p0 * p0))
            g_n = t0 - np.sqrt(a_mse(y - X @ bn, w))
            g_c = t0 - np.sqrt(a_mse(y - c * p0, w))
            g_s = t0 - np.sqrt(a_mse(y - X @ bs, w))
            print(f"  {b.ad:8}{t:>7}{g_n:+11.5f}{g_c:+11.5f}{g_s:+14.5f}{g_n - g_c - g_s:+10.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
