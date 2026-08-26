# ruff: noqa
"""EKSEN 5 (b,c,d,e) -- kosunun cozumlemesi. FIT YOK, npz'den okur.

Olculenler:
  * agirlikli / duz RMSLE, ofs uzayinda MSE           (kol basina, blok basina)
  * ILERI SURUKLENME YANLILIGI b_i = ort(gercek ofs) - ort(tahmin ofs)
      agirlikli ortalama, std, %pozitif, ufuk kovalarina gore
  * trafo bazinda MSE ayrisimi + KIRPMA TABLOSU (K = 0,1,5,10,25,50)
  * MELEZ: alfa*A + (1-alfa)*B log uzayinda
  * dMSE cevirisi ve yeni RMSLE
  * eslenik SH: (blok, tohum) ciftleri uzerinde
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
import olcut  # noqa: E402
import tuketim_model as tm  # noqa: E402

CIK = KOK / "data" / "interim" / "eksen5"
TOHUMLAR = (1000, 1001, 1002)
KOLLAR = ("A", "Aplus", "B")
PENCERELER = (90, 180, 365, 9999)
MEVCUT_MSE = 1.03207  # v55 LB^2 tabani


def wmean(x, w):
    return float(np.dot(w, np.asarray(x, dtype="float64")) / w.sum())


def test_uygun(test: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Testin SICAK + seviyesi TANIMLI satirlari ve tum testteki payi."""
    tab = pd.read_parquet(CIK / "seviye_TEST.parquet")
    sev = np.full(len(test), np.nan)
    for W in PENCERELER:
        v = test["tanim"].map(tab[f"sev{W}"]).to_numpy(dtype="float64")
        yeni = ~np.isfinite(sev) & np.isfinite(v)
        sev[yeni] = v[yeni]
    msk = ((test["soguk_mu"] == 0).to_numpy()) & np.isfinite(sev)
    return test[msk], float(msk.mean())


def yukle(aile: str, blok: str) -> dict:
    z = np.load(CIK / f"kos_{aile}_{blok}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def kol_ofs(p: dict, kol: str, tohumlar) -> np.ndarray:
    """Tohum ortalamasi LOG (ofs) uzayinda -- uretimle ayni."""
    return np.mean([p[f"{kol}_{t}"] for t in tohumlar], axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aile", default="lgbm")
    ar = ap.parse_args()

    _egitim, test = d.cerceveleri_kur()
    tsicak, p_test = test_uygun(test)
    gk = olcut.guc_kenarlari(test)
    print(f"  TEST uygun (sicak + seviye tanimli): {len(tsicak):,}  tum testin %{p_test * 100:.2f}")

    V = {}
    for b in tm.BLOKLAR:
        p = yukle(ar.aile, b.ad)
        cer = pd.DataFrame(
            {
                "guc": p["guc"],
                "ufuk_gun": p["ufuk_gun"],
                "t_son_kayit_yasi": p["t_son_kayit_yasi"],
            }
        )
        w, tani = olcut.test_agirliklari(cer, tsicak, gk)
        g_ofs = np.log1p(np.clip(p["gercek"], 0, None)) - p["lg"]
        V[b.ad] = {
            "p": p,
            "w": w,
            "tani": tani,
            "g": g_ofs,
            "trafo": pd.Series(p["tanim"]),
            "ufuk": p["ufuk_gun"],
        }
        print(
            f"  {b.ad}: n {len(w):,}  ESS {tani['ess_orani']:.3f}"
            f"  kapsanmayan {tani['kapsanmayan']:.4f}  guvenilir {tani['guvenilir']}"
        )

    # ------------------------------------------------------------- 1 SKOR
    print("\n" + "=" * 100)
    print("1) SKOR -- ofs uzayinda MSE (= RMSLE^2), sicak+seviyeli satirlar")
    print(f"  {'blok':7}{'kol':7}{'MSE_ag':>10}{'RMSLE_ag':>10}{'MSE_duz':>10}{'RMSLE_duz':>11}")
    MSE = {}
    for b in tm.BLOKLAR:
        v = V[b.ad]
        for kol in KOLLAR:
            r = kol_ofs(v["p"], kol, TOHUMLAR)
            e = v["g"] - r
            ma = wmean(e**2, v["w"])
            md = float((e**2).mean())
            MSE[(b.ad, kol)] = (ma, md)
            print(f"  {b.ad:7}{kol:7}{ma:10.5f}{np.sqrt(ma):10.5f}{md:10.5f}{np.sqrt(md):11.5f}")
        print()

    # ------------------------------------------------- 2 ESLENIK FARKLAR
    print("=" * 100)
    print("2) ESLENIK FARKLAR -- (blok, tohum) ciftleri, dMSE_ag = kol - A  (negatif = IYI)")
    print(f"  {'karsilastirma':22}{'ort dMSE':>12}{'SH':>10}{'t':>8}{'neg':>9}")
    ciftler = {}
    for kol in ("Aplus", "B"):
        farklar = []
        for b in tm.BLOKLAR:
            v = V[b.ad]
            for t in TOHUMLAR:
                e0 = v["g"] - v["p"][f"A_{t}"]
                e1 = v["g"] - v["p"][f"{kol}_{t}"]
                farklar.append(wmean(e1**2, v["w"]) - wmean(e0**2, v["w"]))
        a = np.array(farklar)
        ciftler[kol] = a
        sh = a.std(ddof=1) / np.sqrt(len(a))
        print(
            f"  {kol + ' - A':22}{a.mean():+12.5f}{sh:10.5f}{a.mean() / sh:+8.2f}"
            f"{int((a < 0).sum()):>6}/{len(a)}"
        )
    # B - Aplus
    farklar = []
    for b in tm.BLOKLAR:
        v = V[b.ad]
        for t in TOHUMLAR:
            e0 = v["g"] - v["p"][f"Aplus_{t}"]
            e1 = v["g"] - v["p"][f"B_{t}"]
            farklar.append(wmean(e1**2, v["w"]) - wmean(e0**2, v["w"]))
    a = np.array(farklar)
    sh = a.std(ddof=1) / np.sqrt(len(a))
    print(
        f"  {'B - Aplus':22}{a.mean():+12.5f}{sh:10.5f}{a.mean() / sh:+8.2f}"
        f"{int((a < 0).sum()):>6}/{len(a)}"
    )
    print("\n  blok bazinda (tohum ortalamasi, dMSE_ag):")
    print(f"  {'blok':7}{'Aplus-A':>12}{'B-A':>12}{'B-Aplus':>12}")
    for b in tm.BLOKLAR:
        v = V[b.ad]
        s = {}
        for kol in KOLLAR:
            r = kol_ofs(v["p"], kol, TOHUMLAR)
            s[kol] = wmean((v["g"] - r) ** 2, v["w"])
        print(
            f"  {b.ad:7}{s['Aplus'] - s['A']:+12.5f}{s['B'] - s['A']:+12.5f}"
            f"{s['B'] - s['Aplus']:+12.5f}"
        )

    # ------------------------------------------------------ 3 YANLILIK b_i
    print("\n" + "=" * 100)
    print("3) ILERI SURUKLENME YANLILIGI  b_i = ort(gercek ofs) - ort(tahmin ofs), trafo bazinda")
    print(
        f"  {'blok':7}{'kol':7}{'ag.ort':>10}{'std':>9}{'medyan':>9}{'poz%':>8}"
        f"{'|ort|':>9}{'MSE_b':>9}"
    )
    for b in tm.BLOKLAR:
        v = V[b.ad]
        tr = v["trafo"]
        w = v["w"]
        for kol in KOLLAR:
            r = kol_ofs(v["p"], kol, TOHUMLAR)
            e = v["g"] - r
            num = pd.Series(e * w).groupby(tr).sum()
            den = pd.Series(w).groupby(tr).sum()
            bi = (num / den).to_numpy()
            wt = den.to_numpy()
            m = float(np.dot(wt, bi) / wt.sum())
            sd = float(np.sqrt(np.dot(wt, (bi - m) ** 2) / wt.sum()))
            med = float(np.median(bi))
            poz = float((bi > 0).mean() * 100)
            mse_b = float(np.dot(wt, bi**2) / wt.sum())
            print(
                f"  {b.ad:7}{kol:7}{m:+10.4f}{sd:9.4f}{med:+9.4f}{poz:8.1f}"
                f"{abs(m):9.4f}{mse_b:9.4f}"
            )
        print()

    # yanlilik x UFUK
    print("=" * 100)
    print("3b) YANLILIK x UFUK KOVASI (ag. ortalama e = gercek - tahmin)")
    kenar = [0, 31, 61, 91, 200]
    print(
        f"  {'blok':7}{'kol':7}" + "".join(f"{f'{kenar[i]}-{kenar[i + 1]}':>10}" for i in range(4))
    )
    for b in tm.BLOKLAR:
        v = V[b.ad]
        kv = np.digitize(v["ufuk"], kenar[1:-1])
        for kol in KOLLAR:
            r = kol_ofs(v["p"], kol, TOHUMLAR)
            e = v["g"] - r
            hu = []
            for k in range(4):
                m = kv == k
                hu.append(wmean(e[m], v["w"][m]) if m.any() else np.nan)
            print(f"  {b.ad:7}{kol:7}" + "".join(f"{x:+10.4f}" for x in hu))
        print()

    # ------------------------------------------------------ 4 KIRPMA TABLOSU
    print("=" * 100)
    print("4) KIRPMA TABLOSU -- en buyuk K trafo (B-A farkina katki) atilinca dMSE_ag")
    print(f"  {'blok':7}{'K':>4}{'kalan_tr':>10}{'MSE_A':>10}{'MSE_B':>10}{'B-A':>11}")
    for b in tm.BLOKLAR:
        v = V[b.ad]
        tr = v["trafo"]
        w = v["w"]
        rA = kol_ofs(v["p"], "A", TOHUMLAR)
        rB = kol_ofs(v["p"], "B", TOHUMLAR)
        dk = ((v["g"] - rB) ** 2 - (v["g"] - rA) ** 2) * w
        katki = pd.Series(dk).groupby(tr).sum()
        srt = katki.abs().sort_values(ascending=False).index.to_numpy()
        for K in (0, 1, 5, 10, 25, 50):
            m = ~tr.isin(set(srt[:K])).to_numpy()
            mA = wmean((v["g"][m] - rA[m]) ** 2, w[m])
            mB = wmean((v["g"][m] - rB[m]) ** 2, w[m])
            print(f"  {b.ad:7}{K:4d}{int(tr[m].nunique()):10,}{mA:10.5f}{mB:10.5f}{mB - mA:+11.5f}")
        print()

    # ------------------------------------------------------------- 5 MELEZ
    print("=" * 100)
    print("5) MELEZ -- ofs_melez = alfa*A + (1-alfa)*B   (log/ofs uzayinda)")
    alfalar = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
    print(f"  {'blok':7}" + "".join(f"{a:>10.1f}" for a in alfalar))
    genel = {a: [] for a in alfalar}
    for b in tm.BLOKLAR:
        v = V[b.ad]
        rA = kol_ofs(v["p"], "A", TOHUMLAR)
        rB = kol_ofs(v["p"], "B", TOHUMLAR)
        satir = []
        for a in alfalar:
            r = a * rA + (1 - a) * rB
            m = wmean((v["g"] - r) ** 2, v["w"])
            satir.append(m)
            genel[a].append(m)
        print(f"  {b.ad:7}" + "".join(f"{x:10.5f}" for x in satir))
    print(f"  {'ORT':7}" + "".join(f"{np.mean(genel[a]):10.5f}" for a in alfalar))
    print(
        f"  {'dMSE':7}"
        + "".join(f"{np.mean(genel[a]) - np.mean(genel[1.0]):+10.5f}" for a in alfalar)
    )
    # melez eslenik SH (en iyi alfa)
    eniyi = min(alfalar, key=lambda a: np.mean(genel[a]))
    farklar = []
    for b in tm.BLOKLAR:
        v = V[b.ad]
        for t in TOHUMLAR:
            rA = v["p"][f"A_{t}"]
            rB = v["p"][f"B_{t}"]
            e0 = v["g"] - rA
            e1 = v["g"] - (eniyi * rA + (1 - eniyi) * rB)
            farklar.append(wmean(e1**2, v["w"]) - wmean(e0**2, v["w"]))
    a_ = np.array(farklar)
    sh = a_.std(ddof=1) / np.sqrt(len(a_))
    print(
        f"\n  en iyi alfa {eniyi:.1f}: eslenik dMSE {a_.mean():+.5f}  SH {sh:.5f}"
        f"  t {a_.mean() / sh:+.2f}  neg {int((a_ < 0).sum())}/{len(a_)}"
    )

    # ------------------------------------------------------- 6 dMSE CEVIRI
    print("\n" + "=" * 100)
    print("6) dMSE CEVIRISI -- p = testin sicak+seviyeli payi")
    print(f"  p = {p_test:.4f}   mevcut MSE tabani {MEVCUT_MSE:.5f}  (RMSLE 1,01591)")
    print(f"  {'oneri':22}{'yerel dMSE':>12}{'p*dMSE':>10}{'yeni RMSLE':>12}{'gerekli?':>10}")
    for ad, dm in (
        ("Aplus - A", ciftler["Aplus"].mean()),
        ("B - A", ciftler["B"].mean()),
        (f"melez alfa={eniyi:.1f} - A", a_.mean()),
    ):
        g = p_test * dm
        yeni = np.sqrt(MEVCUT_MSE + g)
        print(
            f"  {ad:22}{dm:+12.5f}{g:+10.5f}{yeni:12.5f}{'EVET' if g <= -0.01933 else 'hayir':>10}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
