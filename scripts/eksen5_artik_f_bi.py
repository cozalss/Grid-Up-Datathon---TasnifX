# ruff: noqa
"""EKSEN 5 (b) -- ASIL SORU: artik hedefi b_i'nin ORTALAMASINI ve STD'sini kucultuyor mu?

Artik hedefinin ZORLADIGI limitte modelin trafo seviyesi tam s_i olur, yani
    b_i(lam=1) = g_i - s_i          (gercek seviye eksi gecmis seviye)
    b_i(lam=0) = g_i - m_i          (gercek seviye eksi MODELIN seviyesi)

Ikisini yan yana koyuyoruz: ort, std, medyan, %pozitif, MSE_b.
Ayrica lam suruyoruz -- b_i'nin std'si lam'in KUADRATIK fonksiyonu.
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
PENCERELER = (90, 180, 365, 9999)


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


def sev_bagla(tanim, blok):
    tab = pd.read_parquet(CIK / f"seviye_{blok}.parquet")
    sev = np.full(len(tanim), np.nan)
    for W in PENCERELER:
        v = tanim.map(tab[f"sev{W}"]).to_numpy(dtype="float64")
        y = ~np.isfinite(sev) & np.isfinite(v)
        sev[y] = v[y]
    return sev


def ozet(bi, wt, ad):
    m = float(np.dot(wt, bi) / wt.sum())
    sd = float(np.sqrt(np.dot(wt, (bi - m) ** 2) / wt.sum()))
    return (
        f"  {ad:26}{m:+9.4f}{sd:9.4f}{float(np.median(bi)):+9.4f}"
        f"{float((bi > 0).mean() * 100):8.1f}{float(np.dot(wt, bi**2) / wt.sum()):9.4f}"
    )


def main() -> int:
    egitim, test = d.cerceveleri_kur()
    gk = olcut.guc_kenarlari(test)
    sev_t = sev_bagla(test["tanim"], "TEST")
    tsicak = test[((test["soguk_mu"] == 0).to_numpy()) & np.isfinite(sev_t)]

    for b in tm.BLOKLAR:
        dg, r, g = blok_verisi(egitim, b.ad)
        tr = pd.Series(dg["tanim"].to_numpy())
        s = sev_bagla(tr, b.ad)
        ok = np.isfinite(s)
        cer = dg[ok].reset_index(drop=True)
        w, tani = olcut.test_agirliklari(cer, tsicak, gk)
        trk = tr[ok].reset_index(drop=True)
        df = pd.DataFrame({"tr": trk.to_numpy(), "g": g[ok], "r": r[ok], "s": s[ok], "w": w})
        gb = df.groupby("tr")
        wt = gb["w"].sum().to_numpy()

        # agirlik toplami 0 olan trafolar: duz ortalamaya dus, sonra 0 agirlikla girsin
        def wag(kol):
            num = gb.apply(lambda x: np.dot(x["w"], x[kol]), include_groups=False).to_numpy()
            duz = gb[kol].mean().to_numpy()
            return np.where(wt > 0, num / np.where(wt > 0, wt, 1.0), duz)

        gi, mi = wag("g"), wag("r")
        si = gb["s"].first().to_numpy()
        wt2 = np.where(wt > 0, wt, 0.0)
        if wt2.sum() == 0:
            continue
        print(f"\n=== {b.ad}  trafo {len(wt):,}  ESS {tani['ess_orani']:.3f}")
        print(f"  {'b_i tanimi':26}{'ag.ort':>9}{'std':>9}{'medyan':>9}{'poz%':>8}{'MSE_b':>9}")
        print(ozet(gi - mi, wt2, "lam=0  g_i - m_i (MEVCUT)"))
        print(ozet(gi - si, wt2, "lam=1  g_i - s_i (ARTIK)"))
        print("  --- lam taramasi: b_i = g_i - (m_i + lam*(s_i-m_i))")
        print(f"  {'lam':>6}{'ag.ort':>10}{'std':>9}{'MSE_b':>9}")
        for la in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
            bi = gi - (mi + la * (si - mi))
            m = float(np.dot(wt2, bi) / wt2.sum())
            sd = float(np.sqrt(np.dot(wt2, (bi - m) ** 2) / wt2.sum()))
            print(f"  {la:>6.2f}{m:+10.4f}{sd:9.4f}{float(np.dot(wt2, bi**2) / wt2.sum()):9.4f}")
        # seviye tahmincisi olarak s_i mi m_i mi daha iyi?
        print(
            f"  SEVIYE KESTIRIMI RMSE:  m_i {np.sqrt(np.dot(wt2, (gi - mi) ** 2) / wt2.sum()):.4f}"
            f"   s_i {np.sqrt(np.dot(wt2, (gi - si) ** 2) / wt2.sum()):.4f}"
            f"   kor(g_i,m_i) {np.corrcoef(gi, mi)[0, 1]:+.3f}"
            f"   kor(g_i,s_i) {np.corrcoef(gi, si)[0, 1]:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
