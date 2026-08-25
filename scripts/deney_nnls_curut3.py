# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""CURUTME 3 -- iki catlagi kovala.

CATLAK 1: 'xgb agir 3/2/1/1,4' SABIT vektoru +0,00196 t=+2,34, UC BLOKTA DA
          pozitif. Oysa blok-disi simpleks 0 vermisti. Hangisi dogru?
CATLAK 2: KOSULLU simpleks orakulu 3x buyuk tavan veriyor (-0,0112). Bu tavan
          'seviye' sinifina mi giriyor -- TABAKA BASINA skalar c ile yeniden
          uretilebiliyor mu?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import deney_nnls_curut as c1  # noqa: E402
import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402

TOHUMLAR, AILELER, URETIM_N, KAT = c1.TOHUMLAR, c1.AILELER, c1.URETIM_N, c1.SICAK_KATSAYI


def a_mse(e, w):  # noqa: ANN001, ANN202
    return float(np.dot(w, e * e) / w.sum())


def hukum(f, ad):  # noqa: ANN001, ANN202
    fa = np.asarray(f, "float64")
    sh = float(fa.std(ddof=1) / np.sqrt(len(fa)))
    t = fa.mean() / sh if sh > 0 else 0.0
    print(
        f"  {ad:40}{fa.mean():+10.5f}{sh:10.5f}{t:+8.2f}"
        f"{int((fa > 0).sum()):>5}/{len(fa)}{-fa.mean() * KAT:+10.5f}"
    )


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
        y = np.log1p(np.clip(np.load(c1.DIZIN / f"{b.ad}_gercek.npy").astype("float64"), 0, None))
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
            "guc": ol._kova(np.log1p(dg["guc"].to_numpy("float64")), guc_kenar),
        }
        V[b.ad] = {"y": y, "w": w, "X": X, "tab": tab, "n": len(y)}

    ORAKUL = {
        (b.ad, t): c1.cozum_simpleks(V[b.ad]["X"][t], V[b.ad]["y"], V[b.ad]["w"])
        for b in tm.BLOKLAR
        for t in TOHUMLAR
    }
    ort = np.mean([ORAKUL[k] for k in ORAKUL], axis=0)
    print("\n" + "=" * 104)
    print("1) ORAKUL SIMPLEKS ORTALAMASI vs URETIM")
    print("=" * 104)
    print(f"  {'':12}" + "".join(f"{a:>12}" for a in AILELER))
    print(f"  {'uretim':12}" + "".join(f"{x:12.4f}" for x in URETIM_N))
    print(f"  {'orakul ort':12}" + "".join(f"{x:12.4f}" for x in ort))

    print("\n  BLOK-DISI POOLED SIMPLEKS AGIRLIKLARI (fit = diger iki blok, ayni tohum)")
    print(f"  {'blok':8}{'tohum':>7}" + "".join(f"{a:>12}" for a in AILELER))
    BD = {}
    for b in tm.BLOKLAR:
        for t in TOHUMLAR:
            Xs = np.vstack([V[o.ad]["X"][t] for o in tm.BLOKLAR if o.ad != b.ad])
            ys = np.concatenate([V[o.ad]["y"] for o in tm.BLOKLAR if o.ad != b.ad])
            ws = np.concatenate([V[o.ad]["w"] for o in tm.BLOKLAR if o.ad != b.ad])
            BD[(b.ad, t)] = c1.cozum_simpleks(Xs, ys, ws)
            print(f"  {b.ad:8}{t:>7}" + "".join(f"{x:12.4f}" for x in BD[(b.ad, t)]))

    print("\n  BLOK-DISI ORAKUL-ORTALAMASI (diger iki blogun orakul agirliklarinin ortalamasi)")
    print(f"  {'blok':8}{'tohum':>7}" + "".join(f"{a:>12}" for a in AILELER))
    BDO = {}
    for b in tm.BLOKLAR:
        for t in TOHUMLAR:
            BDO[(b.ad, t)] = np.mean(
                [ORAKUL[(o.ad, tt)] for o in tm.BLOKLAR if o.ad != b.ad for tt in TOHUMLAR], axis=0
            )
            print(f"  {b.ad:8}{t:>7}" + "".join(f"{x:12.4f}" for x in BDO[(b.ad, t)]))

    def kos(fbeta):  # noqa: ANN001, ANN202
        f = []
        for b in tm.BLOKLAR:
            v = V[b.ad]
            for t in TOHUMLAR:
                X, y, w = v["X"][t], v["y"], v["w"]
                f.append(
                    np.sqrt(a_mse(y - X @ URETIM_N, w)) - np.sqrt(a_mse(y - X @ fbeta(b.ad, t), w))
                )
        return f

    print("\n" + "=" * 104)
    print("2) CATLAK 1: xgb yonu GERCEK mi? (hepsi 9 (blok,tohum) cifti, eslenik SH)")
    print("=" * 104)
    print(f"  {'PROTOKOL':40}{'fark':>10}{'SH':>10}{'t':>8}{'tohum':>8}{'genele':>10}")
    hukum(kos(lambda b, t: BD[(b, t)]), "blok-disi POOLED simpleks")
    hukum(kos(lambda b, t: BDO[(b, t)]), "blok-disi ORAKUL-ORTALAMASI")
    xa = np.array([3.0, 2.0, 1.0, 1.4])
    hukum(kos(lambda b, t, v=xa / xa.sum(): v), "SABIT xgb agir 3/2/1/1,4")
    hukum(kos(lambda b, t, v=ort: v), "SABIT orakul-ortalamasi (SIZINTILI)")

    print("\n  xgb agir 3/2/1/1,4 -- 9 ciftin TAMAMI")
    xn = xa / xa.sum()
    tum = []
    for b in tm.BLOKLAR:
        v = V[b.ad]
        for t in TOHUMLAR:
            X, y, w = v["X"][t], v["y"], v["w"]
            g = np.sqrt(a_mse(y - X @ URETIM_N, w)) - np.sqrt(a_mse(y - X @ xn, w))
            tum.append(g)
            print(f"    {b.ad:8}{t:>7}{g:+11.5f}")
    print(f"    en kotu {min(tum):+.5f}   en iyi {max(tum):+.5f}")

    print("\n  1-B TARAMA: xgb agirligi (cat 3, lgbm 1, ag 1,4 sabit) -- blok bazinda")
    print(f"  {'xgb':>6}" + "".join(f"{b.ad:>12}" for b in tm.BLOKLAR) + f"{'HAVUZ':>12}{'t':>8}")
    for xw in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        vek = np.array([3.0, xw, 1.0, 1.4])
        vek = vek / vek.sum()
        sat, hep = f"  {xw:>6.1f}", []
        for b in tm.BLOKLAR:
            v = V[b.ad]
            fk = [
                np.sqrt(a_mse(v["y"] - v["X"][t] @ URETIM_N, v["w"]))
                - np.sqrt(a_mse(v["y"] - v["X"][t] @ vek, v["w"]))
                for t in TOHUMLAR
            ]
            hep += fk
            sat += f"{np.mean(fk):+12.5f}"
        fa = np.array(hep)
        sh = fa.std(ddof=1) / np.sqrt(9)
        print(sat + f"{fa.mean():+12.5f}{fa.mean() / sh:+8.2f}")

    print("\n  LOO-BLOK MENU SECIMI (aday menusu iki blokta puanlanir, ucuncude sinanir)")
    menu = {
        "uretim": (3, 1, 1, 1.4),
        "lgbm yok": (3, 1, 0, 1.4),
        "ag yok": (3, 1, 1, 0),
        "esit": (1, 1, 1, 1),
        "cat agir": (5, 1, 1, 1.4),
        "cat hafif": (2, 1, 1, 1.4),
        "ag agir": (3, 1, 1, 2.5),
        "xgb agir": (3, 2, 1, 1.4),
        "xgb cok agir": (3, 3, 1, 1.4),
        "yalniz cat": (1, 0, 0, 0),
    }
    f = []
    for b in tm.BLOKLAR:
        puan = {}
        for ad, vek in menu.items():
            vv = np.array(vek, "float64")
            vv /= vv.sum()
            s = 0.0
            for o in tm.BLOKLAR:
                if o.ad == b.ad:
                    continue
                vo = V[o.ad]
                s += float(
                    np.mean([np.sqrt(a_mse(vo["y"] - vo["X"][t] @ vv, vo["w"])) for t in TOHUMLAR])
                )
            puan[ad] = s
        sec = min(puan, key=puan.get)
        sv = np.array(menu[sec], "float64")
        sv /= sv.sum()
        v = V[b.ad]
        for t in TOHUMLAR:
            f.append(
                np.sqrt(a_mse(v["y"] - v["X"][t] @ URETIM_N, v["w"]))
                - np.sqrt(a_mse(v["y"] - v["X"][t] @ sv, v["w"]))
            )
        print(f"    {b.ad:8} secilen: {sec}")
    hukum(f, "LOO-blok menu secimi")

    print("\n" + "=" * 104)
    print("3) CATLAK 2: KOSULLU orakul tavani SEVIYE SINIFI mi? (tabaka basina skalar c)")
    print("=" * 104)
    print(f"  {'PROTOKOL':40}{'fark':>10}{'SH':>10}{'t':>8}{'tohum':>8}{'genele':>10}")

    def tab_c(eksen):  # noqa: ANN001, ANN202
        f = []
        for b in tm.BLOKLAR:
            v = V[b.ad]
            s = v["tab"][eksen]
            for t in TOHUMLAR:
                X, y, w = v["X"][t], v["y"], v["w"]
                p0 = X @ URETIM_N
                p = p0.copy()
                for kod in np.unique(s):
                    m = s == kod
                    if m.sum() < 400:
                        continue
                    c = float(np.dot(w[m], p0[m] * y[m]) / np.dot(w[m], p0[m] * p0[m]))
                    p[m] = c * p0[m]
                f.append(np.sqrt(a_mse(y - p0, w)) - np.sqrt(a_mse(y - p, w)))
        return f

    for eksen in ("bayatlik", "guc"):
        hukum(tab_c(eksen), f"ORAKUL tabaka-basina SKALAR c / {eksen}")
    print("\n  (karsilastirma: orakul tabaka-basina SIMPLEKS bayatlik +0,02091  guc +0,02187)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
