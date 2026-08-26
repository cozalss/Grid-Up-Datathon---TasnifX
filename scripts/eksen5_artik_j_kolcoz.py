# ruff: noqa
"""EKSEN 5 -- EGITILMIS KOLLARIN cozumlemesi (mevcut olan bloklar icin).

A     hedef = ofs,          taban kolonlar
Aplus hedef = ofs,          taban + [sev, sev_n, sev_kaynak]
B     hedef = ofs - sev,    taban + [sev, sev_n, sev_kaynak]
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
import olcut  # noqa: E402
import tuketim_model as tm  # noqa: E402

CIK = KOK / "data" / "interim" / "eksen5"
TOHUMLAR = (1000, 1001, 1002)
KOLLAR = ("A", "Aplus", "B")
PENCERELER = (90, 180, 365, 9999)
MEVCUT_MSE = 1.03207


def wmean(x, w):
    return float(np.dot(w, np.asarray(x, "float64")) / w.sum())


def main() -> int:
    _egitim, test = d.cerceveleri_kur()
    tab = pd.read_parquet(CIK / "seviye_TEST.parquet")
    sev = np.full(len(test), np.nan)
    for W in PENCERELER:
        v = test["tanim"].map(tab[f"sev{W}"]).to_numpy("float64")
        y = ~np.isfinite(sev) & np.isfinite(v)
        sev[y] = v[y]
    uygun = ((test["soguk_mu"] == 0).to_numpy()) & np.isfinite(sev)
    p_test = float(uygun.mean())
    tsicak = test[uygun]
    gk = olcut.guc_kenarlari(test)
    print(f"  p (etkilenen test payi) = {p_test:.4f}")

    ESL = {}
    for b in tm.BLOKLAR:
        yol = CIK / f"kos_lgbm_{b.ad}.npz"
        if not yol.exists():
            print(f"  {b.ad}: HENUZ YOK, atlaniyor")
            continue
        z = np.load(yol, allow_pickle=False)
        p = {k: z[k] for k in z.files}
        mevcut = [t for t in TOHUMLAR if f"A_{t}" in p]
        cer = pd.DataFrame(
            {"guc": p["guc"], "ufuk_gun": p["ufuk_gun"], "t_son_kayit_yasi": p["t_son_kayit_yasi"]}
        )
        w, tani = olcut.test_agirliklari(cer, tsicak, gk)
        g = np.log1p(np.clip(p["gercek"], 0, None)) - p["lg"]
        tr = pd.Series(p["tanim"])
        print(
            f"\n=== {b.ad}  n {len(w):,}  trafo {tr.nunique():,}  tohum {mevcut}"
            f"  ESS {tani['ess_orani']:.3f}"
        )
        print(f"  {'kol':7}{'MSE_ag':>10}{'RMSLE':>10}{'b_i ort':>10}{'b_i std':>10}{'MSE_b':>9}")
        for kol in KOLLAR:
            r = np.mean([p[f"{kol}_{t}"] for t in mevcut], axis=0)
            e = g - r
            num = pd.Series(e * w).groupby(tr).sum()
            den = pd.Series(w).groupby(tr).sum()
            bi = (num / den.where(den > 0, 1.0)).to_numpy()
            wt = den.to_numpy()
            m = float(np.dot(wt, bi) / wt.sum())
            sd = float(np.sqrt(np.dot(wt, (bi - m) ** 2) / wt.sum()))
            mse = wmean(e**2, w)
            print(
                f"  {kol:7}{mse:10.5f}{np.sqrt(mse):10.5f}{m:+10.4f}{sd:10.4f}"
                f"{float(np.dot(wt, bi**2) / wt.sum()):9.4f}"
            )
        # eslenik farklar
        for a_, b_ in (("A", "Aplus"), ("A", "B"), ("Aplus", "B")):
            f = [
                wmean((g - p[f"{b_}_{t}"]) ** 2, w) - wmean((g - p[f"{a_}_{t}"]) ** 2, w)
                for t in mevcut
            ]
            ESL[(b.ad, a_, b_)] = f
            print(f"  {b_} - {a_}: dMSE {np.mean(f):+.5f}  (tohumlar {[round(x, 5) for x in f]})")
        # melez
        rA = np.mean([p[f"A_{t}"] for t in mevcut], axis=0)
        rB = np.mean([p[f"B_{t}"] for t in mevcut], axis=0)
        m0 = wmean((g - rA) ** 2, w)
        print("  MELEZ alfa*A + (1-alfa)*B:")
        for al in (0.0, 0.25, 0.5, 0.75, 1.0):
            mm = wmean((g - (al * rA + (1 - al) * rB)) ** 2, w)
            print(f"    alfa {al:.2f}  MSE {mm:.5f}  dMSE {mm - m0:+.5f}")
        # kirpma B-A
        dk = ((g - rB) ** 2 - (g - rA) ** 2) * w
        srt = pd.Series(dk).groupby(tr).sum().abs().sort_values(ascending=False).index.to_numpy()
        print(f"  KIRPMA (B - A):  {'K':>4}{'kalan_tr':>10}{'dMSE':>11}")
        for K in (0, 1, 5, 10, 25, 50):
            msk = ~tr.isin(set(srt[:K])).to_numpy()
            print(
                f"                  {K:4d}{int(tr[msk].nunique()):10,}"
                f"{wmean((g[msk] - rB[msk]) ** 2, w[msk]) - wmean((g[msk] - rA[msk]) ** 2, w[msk]):+11.5f}"
            )

    if ESL:
        print("\n" + "=" * 90)
        print("TOPLU ESLENIK HUKUM (mevcut bloklar)")
        for a_, b_ in (("A", "Aplus"), ("A", "B"), ("Aplus", "B")):
            v = np.concatenate(
                [np.array(x) for (k, aa, bb), x in ESL.items() if (aa, bb) == (a_, b_)]
            )
            sh = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
            print(
                f"  {b_ + ' - ' + a_:16}{v.mean():+10.5f}  SH {sh:.5f}  t {v.mean() / sh:+6.2f}"
                f"  neg {int((v < 0).sum())}/{len(v)}"
                f"  -> p*dMSE {p_test * v.mean():+.5f}  yeni RMSLE {np.sqrt(MEVCUT_MSE + p_test * v.mean()):.5f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
