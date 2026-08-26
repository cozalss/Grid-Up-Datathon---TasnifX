# ruff: noqa
"""EKSEN 5 -- ayakta kalan TEK adayin (kucuk lam) kirpma tablosu ve rejim raporu.

Kalici kural 1: her kazanc trafo bazinda ayristirilir, K = 0,1,5,10,25,50.
Kalici kural 4: mevsimsel ikiz (yaz25) ayri.
Sart (d): soguk rejim TANIMSIZ -- dokunulmayan pay acikca yazilir.
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
MEVCUT_MSE = 1.03207


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
    return dg, [x - lg for x in loglar], np.log1p(np.clip(y, 0, None)) - lg


def sev_bagla(tanim, blok):
    tab = pd.read_parquet(CIK / f"seviye_{blok}.parquet")
    s = np.full(len(tanim), np.nan)
    for W in PENCERELER:
        v = tanim.map(tab[f"s{W}" if f"s{W}" in tab.columns else f"sev{W}"]).to_numpy("float64")
        y = ~np.isfinite(s) & np.isfinite(v)
        s[y] = v[y]
    return s


def wmean(x, w):
    return float(np.dot(w, np.asarray(x, dtype="float64")) / w.sum())


def grup_ort(x, w, tr):
    xs = pd.Series(np.asarray(x, dtype="float64"))
    s = (xs * w).groupby(tr).transform("sum")
    t = pd.Series(w).groupby(tr).transform("sum")
    duz = xs.groupby(tr).transform("mean")
    return np.where(t.to_numpy() > 0, (s / t.where(t > 0, 1.0)).to_numpy(), duz.to_numpy())


def main() -> int:
    egitim, test = d.cerceveleri_kur()
    gk = olcut.guc_kenarlari(test)
    sev_t = sev_bagla(test["tanim"], "TEST")
    sicak_t = (test["soguk_mu"] == 0).to_numpy()
    uygun_t = sicak_t & np.isfinite(sev_t)
    tsicak = test[uygun_t]
    print("REJIM PAYLARI (TEST, 714.688 satir)")
    print(f"  soguk (artik hedefi TANIMSIZ)          %{(~sicak_t).mean() * 100:.2f}")
    print(
        f"  sicak ama POZITIF GECMIS YOK (olu)     %{(sicak_t & ~np.isfinite(sev_t)).mean() * 100:.2f}"
    )
    print(f"  sicak + seviye tanimli (ETKILENEN p)   %{uygun_t.mean() * 100:.2f}")
    p_test = float(uygun_t.mean())

    V = {}
    for b in tm.BLOKLAR:
        dg, ofsl, g = blok_verisi(egitim, b.ad)
        tr = pd.Series(dg["tanim"].to_numpy())
        s = sev_bagla(tr, b.ad)
        ok = np.isfinite(s)
        cer = dg[ok].reset_index(drop=True)
        w, tani = olcut.test_agirliklari(cer, tsicak, gk)
        V[b.ad] = {
            "w": w,
            "tani": tani,
            "tr": tr[ok].reset_index(drop=True),
            "s": s[ok],
            "g": g[ok],
            "ofsl": [x[ok] for x in ofsl],
        }

    for la in (0.10, 0.20):
        print("\n" + "=" * 96)
        print(f"KIRPMA TABLOSU  lam = {la:.2f}   (r + lam*(s_i - m_i))")
        print(f"  {'blok':7}{'K':>4}{'kalan_tr':>10}{'MSE_0':>10}{'MSE_lam':>10}{'dMSE':>11}")
        for b in tm.BLOKLAR:
            v = V[b.ad]
            r = np.mean(v["ofsl"], axis=0)
            m = grup_ort(r, v["w"], v["tr"])
            duz = la * (v["s"] - m)
            tr = v["tr"]
            dk = ((v["g"] - (r + duz)) ** 2 - (v["g"] - r) ** 2) * v["w"]
            srt = (
                pd.Series(dk).groupby(tr).sum().abs().sort_values(ascending=False).index.to_numpy()
            )
            for K in (0, 1, 5, 10, 25, 50):
                msk = ~tr.isin(set(srt[:K])).to_numpy()
                m0 = wmean((v["g"][msk] - r[msk]) ** 2, v["w"][msk])
                m1 = wmean((v["g"][msk] - (r[msk] + duz[msk])) ** 2, v["w"][msk])
                print(
                    f"  {b.ad:7}{K:4d}{int(tr[msk].nunique()):10,}{m0:10.5f}{m1:10.5f}{m1 - m0:+11.5f}"
                )
            print()

    # ------- ADIL PROTOKOL: lam TEK TEMIZ FOLD'da (kis26) uydurulur, IKIZDE sinanir
    print("=" * 96)
    print("ADIL PROTOKOL -- lam yalniz-gecmis fold'da (kis26) uydurulur, MEVSIMSEL IKIZDE sinanir")
    lam = {}
    for b in tm.BLOKLAR:
        v = V[b.ad]
        r = np.mean(v["ofsl"], axis=0)
        m = grup_ort(r, v["w"], v["tr"])
        duz = v["s"] - m
        lam[b.ad] = wmean((v["g"] - r) * duz, v["w"]) / wmean(duz**2, v["w"])
    print("  blok ici lam*: " + "  ".join(f"{k} {x:+.3f}" for k, x in lam.items()))
    for kaynak in ("kis26", "guz25"):
        v = V["yaz25"]
        r = np.mean(v["ofsl"], axis=0)
        m = grup_ort(r, v["w"], v["tr"])
        duz = v["s"] - m
        m0 = wmean((v["g"] - r) ** 2, v["w"])
        m1 = wmean((v["g"] - (r + lam[kaynak] * duz)) ** 2, v["w"])
        print(
            f"  {kaynak} lam*={lam[kaynak]:+.3f} -> yaz25 (IKIZ):  dMSE {m1 - m0:+.5f}"
            f"   p*dMSE {p_test * (m1 - m0):+.5f}"
            f"   yeni RMSLE {np.sqrt(MEVCUT_MSE + p_test * (m1 - m0)):.5f}"
        )
    # sabit lam=0.10 icin ikiz sonucu
    v = V["yaz25"]
    r = np.mean(v["ofsl"], axis=0)
    m = grup_ort(r, v["w"], v["tr"])
    duz = v["s"] - m
    m0 = wmean((v["g"] - r) ** 2, v["w"])
    for la in (0.05, 0.10, 0.15, 0.20):
        m1 = wmean((v["g"] - (r + la * duz)) ** 2, v["w"])
        print(f"  sabit lam={la:.2f} -> yaz25 (IKIZ): dMSE {m1 - m0:+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
