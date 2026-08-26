# ruff: noqa
"""EKSEN 5 -- SEVIYE PENCERESI ve BUZME: s_i, m_i'yi HIC yakalayabiliyor mu?

Artik hedefinin butun bahsi su: s_i, trafonun hedef penceresindeki GERCEK
seviyesinin (g_i) iyi bir kestiricisi olsun. Eger degilse, hedefi s_i ile
merkezlemek iyi bir kestiriciyi (m_i) kotusuyle degistirmektir.

Burada s_i'nin EN IYI halini ariyoruz:
  * pencere: 7/14/30/60/90/180/365/tum
  * afin kalibrasyon:  a + c*s_i     (blok ICI, YANLI ust sinir)
  * guvenilirlik buzmesi: n_i/(n_i+M) ile ortalamaya
ve hepsini m_i (modelin ima ettigi seviye) ile YAN YANA koyuyoruz.
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
import olcut  # noqa: E402
import tuketim_model as tm  # noqa: E402

ONB = KOK / "data" / "interim" / "aile_onbellek"
CIK = KOK / "data" / "interim" / "eksen5"
TOHUMLAR = (1000, 1001, 1002)
AGIRLIK = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 1.4}
PENC = (7, 14, 30, 60, 90, 180, 365, 9999)
KESME = {"yaz25": "2025-03-31", "guz25": "2025-07-31", "kis26": "2025-11-30", "TEST": "2026-03-31"}


def blok_verisi(egitim, blok):
    _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, blok)
    sicak = ~soguk
    dg = dogrulama[sicak].reset_index(drop=True)
    y = gercek[sicak]
    pay = sum(AGIRLIK.values())
    loglar = []
    for t in TOHUMLAR:
        s = np.zeros(len(dg), dtype="float64")
        for a, w in AGIRLIK.items():
            s += w * np.load(ONB / f"{blok}_{t}_{a}_uretim.npy").astype("float64")
        loglar.append(s / pay)
    lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    return dg, np.mean(loglar, axis=0) - lg, np.log1p(np.clip(y, 0, None)) - lg


def seviye_tablosu(ham, kesme):
    k = pd.Timestamp(kesme)
    alt = ham[(ham["tarih"] <= k) & (ham["tarih"] >= pd.Timestamp(tm.EGITIM_BASI))]
    poz = alt[alt["tuketim"] > 0].copy()
    poz["_ofs"] = np.log1p(poz["tuketim"].to_numpy(dtype="float64")) - np.log1p(
        poz["guc"].to_numpy(dtype="float64")
    )
    out = None
    for W in PENC:
        p = poz[poz["tarih"] >= k - pd.Timedelta(days=W - 1)]
        g = p.groupby("tanim")["_ofs"]
        t = pd.DataFrame({f"s{W}": g.mean(), f"n{W}": g.size()})
        out = t if out is None else out.join(t, how="outer")
    return out


def wrmse(x, w):
    return float(np.sqrt(np.dot(w, x**2) / w.sum()))


def main() -> int:
    ham, _ = tm.yukle()
    ham = ham[["tanim", "guc", "tarih", "tuketim"]].copy()
    ham["tarih"] = pd.to_datetime(ham["tarih"])
    egitim, test = d.cerceveleri_kur()
    gk = olcut.guc_kenarlari(test)
    tt = seviye_tablosu(ham, KESME["TEST"])
    sev_t = np.full(len(test), np.nan)
    for W in PENC:
        v = test["tanim"].map(tt[f"s{W}"]).to_numpy(dtype="float64")
        y = ~np.isfinite(sev_t) & np.isfinite(v)
        sev_t[y] = v[y]
    tsicak = test[((test["soguk_mu"] == 0).to_numpy()) & np.isfinite(sev_t)]

    print("=" * 104)
    print("SEVIYE KESTIRICI YARISI -- hedef: g_i (trafonun HEDEF penceresindeki gercek ort ofs)")
    print("  m_i = uretim modelinin ima ettigi seviye (ORTALAMA TAHMIN)")
    print("=" * 104)
    for b in tm.BLOKLAR:
        dg, r, g = blok_verisi(egitim, b.ad)
        tab = seviye_tablosu(ham, KESME[b.ad])
        tr = pd.Series(dg["tanim"].to_numpy())
        # merdivenli s (referans, 90'dan baslar)
        w_all, tani = olcut.test_agirliklari(dg, tsicak, gk)
        df = pd.DataFrame({"tr": tr.to_numpy(), "g": g, "r": r, "w": w_all})
        gb = df.groupby("tr")
        wt = gb["w"].sum().to_numpy()

        def wag(kol):
            num = gb.apply(lambda x: np.dot(x["w"], x[kol]), include_groups=False).to_numpy()
            return np.where(wt > 0, num / np.where(wt > 0, wt, 1.0), gb[kol].mean().to_numpy())

        gi, mi = wag("g"), wag("r")
        idx = gb.size().index
        print(f"\n=== {b.ad}  trafo {len(idx):,}  ESS {tani['ess_orani']:.3f}")
        print(
            f"  {'kestirici':16}{'kapsam%':>9}{'RMSE':>9}{'kor':>8}{'egim':>8}"
            f"{'afin RMSE':>11}{'buzme RMSE':>12}"
        )
        # m_i her zaman tanimli
        e = gi - mi
        c = np.polyfit(mi, gi, 1)[0]
        print(
            f"  {'m_i (MODEL)':16}{100.0:9.1f}{wrmse(e, wt):9.4f}"
            f"{np.corrcoef(gi, mi)[0, 1]:+8.3f}{c:+8.3f}{'-':>11}{'-':>12}"
        )
        for W in PENC:
            s = idx.map(tab[f"s{W}"]).to_numpy(dtype="float64")
            n = idx.map(tab[f"n{W}"]).to_numpy(dtype="float64")
            ok = np.isfinite(s)
            if ok.sum() < 50:
                continue
            wo = wt[ok]
            if wo.sum() <= 0:
                continue
            go, so, no = gi[ok], s[ok], n[ok]
            r_ = wrmse(go - so, wo)
            kor = np.corrcoef(go, so)[0, 1]
            a_, c_ = np.polyfit(so, go, 1)[::1][0], None
            cc, aa = np.polyfit(so, go, 1)
            afin = wrmse(go - (aa + cc * so), wo)
            # guvenilirlik buzmesi: ortalamaya, M taranarak (blok ICI = yanli ust sinir)
            enb = min(
                wrmse(go - ((no / (no + M)) * so + (M / (no + M)) * np.average(go, weights=wo)), wo)
                for M in (1, 3, 7, 15, 30, 60)
            )
            print(
                f"  {'s_' + str(W):16}{ok.mean() * 100:9.1f}{r_:9.4f}{kor:+8.3f}{cc:+8.3f}"
                f"{afin:11.4f}{enb:12.4f}"
            )
        # m_i ile s_90'in BIRLIKTE regresyonu (ust sinir)
        s90 = idx.map(tab["s90"]).to_numpy(dtype="float64")
        ok = np.isfinite(s90) & (wt > 0)
        X = np.column_stack([np.ones(ok.sum()), mi[ok], s90[ok]])
        W_ = wt[ok]
        beta = np.linalg.lstsq(X * np.sqrt(W_)[:, None], gi[ok] * np.sqrt(W_), rcond=None)[0]
        rr = wrmse(gi[ok] - X @ beta, W_)
        print(
            f"  BIRLIKTE g_i ~ 1 + m_i + s_90:  katsayilar"
            f" [{beta[0]:+.3f}, {beta[1]:+.3f}, {beta[2]:+.3f}]  RMSE {rr:.4f}"
            f"  (yalniz m_i {wrmse(gi[ok] - mi[ok], W_):.4f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
