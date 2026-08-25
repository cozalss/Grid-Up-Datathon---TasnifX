# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""CURUTME 4 -- KENDI catlagima dusman ol.

'xgb 1 -> 1,5/2,0' bulgusu ayakta mi?
  a) KIRPILMIS tablo (K trafo atilarak) -- yogunlasma var mi?
  b) TABAKA ayrismasi: kazanc bayatlik/ufuk/guc'te nerede?
  c) SKALAR c ile yeniden uretilebiliyor mu? (iddianin onerdigi kalici kural)
  d) METRIK: agirliksiz RMSLE'de de duruyor mu?
  e) TOHUM SAYISI: k=1 -> k=3'te optimum kayiyor mu? (uretim k=30)
  f) ag agirligi ekseni ayrica taranir.
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

TOHUMLAR, AILELER, URETIM_N, KAT = c1.TOHUMLAR, c1.AILELER, c1.URETIM_N, c1.SICAK_KATSAYI
XGB15 = np.array([3.0, 1.5, 1.0, 1.4]) / 6.9
XGB20 = np.array([3.0, 2.0, 1.0, 1.4]) / 7.4


def a_mse(e, w):  # noqa: ANN001, ANN202
    return float(np.dot(w, e * e) / w.sum())


def sat(f, ad, n=9):  # noqa: ANN001, ANN202
    fa = np.asarray(f, "float64")
    sh = float(fa.std(ddof=1) / np.sqrt(len(fa)))
    t = fa.mean() / sh if sh > 0 else 0.0
    print(
        f"  {ad:38}{fa.mean():+10.5f}{sh:10.5f}{t:+8.2f}"
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
        V[b.ad] = {
            "y": y,
            "w": w,
            "X": X,
            "trafo": dg["tanim"].to_numpy(),
            "guc": dg["guc"].to_numpy("float64"),
            "bay": ol._kova(dg["t_son_kayit_yasi"].to_numpy("float64"), ol.BAYATLIK_KENARLARI),
            "uf": ol._kova(dg["ufuk_gun"].to_numpy("float64"), ol.UFUK_KENARLARI),
            "gk": ol._kova(np.log1p(dg["guc"].to_numpy("float64")), guc_kenar),
            "n": len(y),
        }

    print("\n" + "=" * 104)
    print("a) KIRPILMIS TABLO -- SABIT xgb agirligi (K trafo, EN COK KAZANANLAR atilir)")
    print("=" * 104)
    for ad, vek in (("xgb=1,5", XGB15), ("xgb=2,0", XGB20)):
        print(f"\n  {ad}")
        KIRP = {}
        for b in tm.BLOKLAR:
            v = V[b.ad]
            dm = np.zeros(v["n"])
            for t in TOHUMLAR:
                e0, e1 = v["y"] - v["X"][t] @ URETIM_N, v["y"] - v["X"][t] @ vek
                dm += v["w"] * (e0 * e0 - e1 * e1)
            KIRP[b.ad] = pd.Series(dm / 3).groupby(pd.Series(v["trafo"])).sum()
        print(f"  {'K':>5}{'fark':>11}{'SH':>10}{'t':>8}{'tohum':>8}{'genele':>10}")
        for K in (0, 1, 5, 10, 25, 50):
            f = []
            for b in tm.BLOKLAR:
                v = V[b.ad]
                at = set(KIRP[b.ad].nlargest(K).index) if K else set()
                tut = ~pd.Series(v["trafo"]).isin(at).to_numpy()
                for t in TOHUMLAR:
                    f.append(
                        np.sqrt(a_mse((v["y"] - v["X"][t] @ URETIM_N)[tut], v["w"][tut]))
                        - np.sqrt(a_mse((v["y"] - v["X"][t] @ vek)[tut], v["w"][tut]))
                    )
            fa = np.array(f)
            sh = fa.std(ddof=1) / np.sqrt(9)
            print(
                f"  {K:>5}{fa.mean():+11.5f}{sh:10.5f}{fa.mean() / sh:+8.2f}"
                f"{int((fa > 0).sum()):>5}/9{-fa.mean() * KAT:+10.5f}"
            )
        print(f"  {'YOGUNLASMA':>12}{'trafo':>8}{'EN BUYUK':>11}{'ilk5':>9}{'ilk25':>9}")
        for b in tm.BLOKLAR:
            p = KIRP[b.ad].sort_values(ascending=False)
            top = p.sum()
            print(
                f"  {b.ad:>12}{p.size:>8}{100 * p.iloc[0] / top:10.1f}%"
                f"{100 * p.iloc[:5].sum() / top:8.1f}%{100 * p.iloc[:25].sum() / top:8.1f}%"
            )

    print("\n" + "=" * 104)
    print("b) TABAKA AYRISMASI (xgb=1,5; kazanc = agirlikli d(MSE) payi, %)")
    print("=" * 104)
    for eksen, adlar in (("bay", "bayatlik"), ("uf", "ufuk"), ("gk", "guc-kova")):
        print(f"\n  {adlar}")
        for b in tm.BLOKLAR:
            v = V[b.ad]
            dm = np.zeros(v["n"])
            for t in TOHUMLAR:
                e0, e1 = v["y"] - v["X"][t] @ URETIM_N, v["y"] - v["X"][t] @ XGB15
                dm += v["w"] * (e0 * e0 - e1 * e1)
            g = pd.Series(dm / 3).groupby(pd.Series(v[eksen])).agg(["sum", "size"])
            top = g["sum"].sum()
            print(
                f"    {b.ad:8}"
                + "  ".join(
                    f"k{int(i)}:{100 * r['sum'] / top:+6.1f}%(n{int(r['size']) // 1000}k)"
                    for i, r in g.iterrows()
                )
            )

    print("\n" + "=" * 104)
    print("c) SKALAR c TESTI -- xgb kazanci tek bir olcekle yeniden uretilebiliyor mu?")
    print("=" * 104)
    print(f"  {'PROTOKOL':38}{'fark':>10}{'SH':>10}{'t':>8}{'tohum':>8}{'genele':>10}")
    f_c, oranlar = [], []
    for b in tm.BLOKLAR:
        v = V[b.ad]
        for t in TOHUMLAR:
            X, y, w = v["X"][t], v["y"], v["w"]
            p0, p1 = X @ URETIM_N, X @ XGB15
            # xgb tahmininin uretim harmanina gore OLCEGI
            c = float(np.dot(w, p0 * p1) / np.dot(w, p0 * p0))
            oranlar.append(c)
            f_c.append(np.sqrt(a_mse(y - p0, w)) - np.sqrt(a_mse(y - c * p0, w)))
    sat(f_c, "xgb yonunun SKALAR c bileseni")
    print(f"    ima edilen olcek c araligi: {min(oranlar):.5f} - {max(oranlar):.5f}")
    # olcekten ARINDIRILMIS xgb yonu
    f_d = []
    for b in tm.BLOKLAR:
        v = V[b.ad]
        for t in TOHUMLAR:
            X, y, w = v["X"][t], v["y"], v["w"]
            p0, p1 = X @ URETIM_N, X @ XGB15
            c = float(np.dot(w, p0 * p1) / np.dot(w, p0 * p0))
            f_d.append(np.sqrt(a_mse(y - p0, w)) - np.sqrt(a_mse(y - p1 / c, w)))
    sat(f_d, "OLCEKTEN ARINDIRILMIS xgb yonu")

    print("\n" + "=" * 104)
    print("d) METRIK DAYANIKLILIGI -- AGIRLIKSIZ RMSLE")
    print("=" * 104)
    print(f"  {'PROTOKOL':38}{'fark':>10}{'SH':>10}{'t':>8}{'tohum':>8}{'genele':>10}")
    for ad, vek in (("xgb=1,5", XGB15), ("xgb=2,0", XGB20)):
        f = []
        for b in tm.BLOKLAR:
            v = V[b.ad]
            o = np.ones(v["n"])
            for t in TOHUMLAR:
                f.append(
                    np.sqrt(a_mse(v["y"] - v["X"][t] @ URETIM_N, o))
                    - np.sqrt(a_mse(v["y"] - v["X"][t] @ vek, o))
                )
        sat(f, f"agirliksiz {ad}")

    print("\n" + "=" * 104)
    print("e) TOHUM SAYISI: optimum xgb agirligi k=1 -> k=3'te kayiyor mu? (uretim k=30)")
    print("=" * 104)
    print(f"  {'xgb':>6}{'k=1 (9 fit ort)':>18}{'k=3 (torbalanmis)':>20}")
    for xw in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        vek = np.array([3.0, xw, 1.0, 1.4])
        vek /= vek.sum()
        g1, g3 = [], []
        for b in tm.BLOKLAR:
            v = V[b.ad]
            for t in TOHUMLAR:
                g1.append(
                    np.sqrt(a_mse(v["y"] - v["X"][t] @ URETIM_N, v["w"]))
                    - np.sqrt(a_mse(v["y"] - v["X"][t] @ vek, v["w"]))
                )
            p0 = np.mean([v["X"][t] @ URETIM_N for t in TOHUMLAR], axis=0)
            p1 = np.mean([v["X"][t] @ vek for t in TOHUMLAR], axis=0)
            g3.append(np.sqrt(a_mse(v["y"] - p0, v["w"])) - np.sqrt(a_mse(v["y"] - p1, v["w"])))
        print(f"  {xw:>6.1f}{np.mean(g1):+18.5f}{np.mean(g3):+20.5f}")

    print("\n" + "=" * 104)
    print("f) AG AGIRLIGI TARAMASI (cat 3, xgb 1, lgbm 1 sabit) ve BIRLESIK")
    print("=" * 104)
    print(f"  {'ag':>6}" + "".join(f"{b.ad:>12}" for b in tm.BLOKLAR) + f"{'HAVUZ':>12}{'t':>8}")
    for aw in (0.6, 1.0, 1.4, 1.8, 2.2):
        vek = np.array([3.0, 1.0, 1.0, aw])
        vek /= vek.sum()
        s, hep = f"  {aw:>6.1f}", []
        for b in tm.BLOKLAR:
            v = V[b.ad]
            fk = [
                np.sqrt(a_mse(v["y"] - v["X"][t] @ URETIM_N, v["w"]))
                - np.sqrt(a_mse(v["y"] - v["X"][t] @ vek, v["w"]))
                for t in TOHUMLAR
            ]
            hep += fk
            s += f"{np.mean(fk):+12.5f}"
        fa = np.array(hep)
        sh = fa.std(ddof=1) / np.sqrt(9)
        print(s + f"{fa.mean():+12.5f}{(fa.mean() / sh if sh > 0 else 0):+8.2f}")

    print("\n  BIRLESIK ADAYLAR")
    print(f"  {'ADAY':38}{'fark':>10}{'SH':>10}{'t':>8}{'tohum':>8}{'genele':>10}")
    for ad, vek in (
        ("3/1,5/1/1,4", (3, 1.5, 1, 1.4)),
        ("3/2/1/1,4", (3, 2, 1, 1.4)),
        ("3/1,5/1/1,2", (3, 1.5, 1, 1.2)),
        ("3/1,5/0,7/1,4", (3, 1.5, 0.7, 1.4)),
        ("3/2/1,5/1,4", (3, 2, 1.5, 1.4)),
        ("3,5/2/1/1,4", (3.5, 2, 1, 1.4)),
    ):
        vv = np.array(vek, "float64")
        vv /= vv.sum()
        f = []
        for b in tm.BLOKLAR:
            v = V[b.ad]
            for t in TOHUMLAR:
                f.append(
                    np.sqrt(a_mse(v["y"] - v["X"][t] @ URETIM_N, v["w"]))
                    - np.sqrt(a_mse(v["y"] - v["X"][t] @ vv, v["w"]))
                )
        sat(f, ad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
