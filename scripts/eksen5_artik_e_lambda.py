# ruff: noqa
"""EKSEN 5 -- SEVIYE GECIRGENLIGI lambda taramasi (onbellekten, FIT YOK).

Artik hedefi (u = ofs - s_i) sunu ZORLAR: modelin ima ettigi trafo seviyesi
tam olarak s_i olsun, yani gecirgenlik katsayisi 1. Bunu fit etmeden, DAGITILABILIR
bir bicimde tarayabiliriz:

    r_i(lam) = r_ham + lam * (s_i - m_i)

    m_i = modelin o trafo icin ORTALAMA tahmini (tahmin aninda bilinir)
    s_i = kesme oncesi son-90g pozitif ort ofs (tahmin aninda bilinir)

    lam = 0  -> mevcut model (ofs hedefi)
    lam = 1  -> modelin seviyesi s_i'ye tam oturur (artik hedefinin ZORLADIGI sey)

Kalici kural 3: hukum (blok, tohum) ciftleri uzerinde eslenik SH ile.
Kalici kural 4: yaz25 = testin mevsimsel ikizi, ayri raporlanir.
Kalici kural 1: kirpma tablosu.
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
LAMBDALAR = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
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
    return {
        "cerceve": dg,
        "tohum_ofs": [x - lg for x in loglar],
        "g": np.log1p(np.clip(y, 0, None)) - lg,
        "lg": lg,
    }


def sev_bagla(tanim: pd.Series, blok: str) -> np.ndarray:
    tab = pd.read_parquet(CIK / f"seviye_{blok}.parquet")
    sev = np.full(len(tanim), np.nan)
    for W in PENCERELER:
        v = tanim.map(tab[f"sev{W}"]).to_numpy(dtype="float64")
        y = ~np.isfinite(sev) & np.isfinite(v)
        sev[y] = v[y]
    return sev


def wmean(x, w):
    return float(np.dot(w, np.asarray(x, dtype="float64")) / w.sum())


def grup_ort(x, w, tr):
    """Trafo bazinda AGIRLIKLI ortalama. Agirlik toplami 0 olan trafolarda
    (olcut o tabakayi kapsamiyor) duz ortalamaya duser -- yoksa 0/0 = NaN
    butun toplami zehirler."""
    xs = pd.Series(np.asarray(x, dtype="float64"))
    s = (xs * w).groupby(tr).transform("sum")
    t = pd.Series(w).groupby(tr).transform("sum")
    duz = xs.groupby(tr).transform("mean")
    return np.where(t.to_numpy() > 0, (s / t.where(t > 0, 1.0)).to_numpy(), duz.to_numpy())


def main() -> int:
    egitim, test = d.cerceveleri_kur()
    gk = olcut.guc_kenarlari(test)

    # TEST'te uygun pay
    sev_t = sev_bagla(test["tanim"], "TEST")
    uygun_t = ((test["soguk_mu"] == 0).to_numpy()) & np.isfinite(sev_t)
    p_test = float(uygun_t.mean())
    tsicak = test[uygun_t]
    print(
        f"  TEST uygun (sicak + seviye tanimli) pay = {p_test:.4f}  ({int(uygun_t.sum()):,} satir)"
    )

    V = {}
    for b in tm.BLOKLAR:
        v = blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        tr = pd.Series(dg["tanim"].to_numpy())
        sev = sev_bagla(tr, b.ad)
        ok = np.isfinite(sev)
        cer = dg[ok].reset_index(drop=True)
        w, tani = olcut.test_agirliklari(cer, tsicak, gk)
        V[b.ad] = {
            "w": w,
            "tani": tani,
            "tr": tr[ok].reset_index(drop=True),
            "s": sev[ok],
            "g": v["g"][ok],
            "tohum_ofs": [x[ok] for x in v["tohum_ofs"]],
            "ufuk": cer["ufuk_gun"].to_numpy(dtype="float64"),
        }
        print(
            f"  {b.ad}: {int(ok.sum()):,} satir  trafo {tr[ok].nunique():,}"
            f"  ESS {tani['ess_orani']:.3f}  guvenilir {tani['guvenilir']}"
        )

    # ------------------------------------------------------- 1 LAMBDA TARAMA
    print("\n" + "=" * 104)
    print("1) LAMBDA TARAMASI  r(lam) = r + lam*(s_i - m_i)   [MSE_agirlikli, tohum ort.]")
    print(
        f"  {'blok':7}" + "".join(f"{la:>11.2f}" for la in LAMBDALAR) + f"{'lam*':>9}{'MSE*':>10}"
    )
    ORT = {la: [] for la in LAMBDALAR}
    for b in tm.BLOKLAR:
        v = V[b.ad]
        r = np.mean(v["tohum_ofs"], axis=0)
        m = grup_ort(r, v["w"], v["tr"])
        duz = v["s"] - m
        satir = []
        for la in LAMBDALAR:
            mse = wmean((v["g"] - (r + la * duz)) ** 2, v["w"])
            satir.append(mse)
            ORT[la].append(mse)
        # analitik en iyi lambda
        e = v["g"] - r
        lam_yildiz = wmean(e * duz, v["w"]) / wmean(duz**2, v["w"])
        mse_yildiz = wmean((e - lam_yildiz * duz) ** 2, v["w"])
        print(
            f"  {b.ad:7}"
            + "".join(f"{x:11.5f}" for x in satir)
            + f"{lam_yildiz:+9.3f}{mse_yildiz:10.5f}"
        )
    print(f"  {'ORT':7}" + "".join(f"{np.mean(ORT[la]):11.5f}" for la in LAMBDALAR))
    print(
        f"  {'dMSE':7}"
        + "".join(f"{np.mean(ORT[la]) - np.mean(ORT[0.0]):+11.5f}" for la in LAMBDALAR)
    )

    # ------------------------------------------------- 2 ESLENIK SH (blok,tohum)
    print("\n" + "=" * 104)
    print("2) ESLENIK dMSE -- (blok, tohum) ciftleri, lam'a gore  (negatif = IYI)")
    print(
        f"  {'lam':>6}{'ort dMSE':>12}{'SH':>10}{'t':>8}{'neg':>9}{'yaz25':>10}{'guz25':>10}{'kis26':>10}"
    )
    for la in LAMBDALAR[1:]:
        fark = []
        blokca = {b.ad: [] for b in tm.BLOKLAR}
        for b in tm.BLOKLAR:
            v = V[b.ad]
            for i, _t in enumerate(TOHUMLAR):
                r = v["tohum_ofs"][i]
                m = grup_ort(r, v["w"], v["tr"])
                duz = v["s"] - m
                m0 = wmean((v["g"] - r) ** 2, v["w"])
                m1 = wmean((v["g"] - (r + la * duz)) ** 2, v["w"])
                fark.append(m1 - m0)
                blokca[b.ad].append(m1 - m0)
        a = np.array(fark)
        sh = a.std(ddof=1) / np.sqrt(len(a))
        print(
            f"  {la:>6.2f}{a.mean():+12.5f}{sh:10.5f}{a.mean() / sh:+8.2f}"
            f"{int((a < 0).sum()):>6}/{len(a)}"
            + "".join(f"{np.mean(blokca[b.ad]):+10.5f}" for b in tm.BLOKLAR)
        )

    # ------------------------------------------- 3 BLOKLARARASI lam* TASINIRLIGI
    print("\n" + "=" * 104)
    print("3) lam* TASINIRLIGI -- bir blokta uydur, DIGERINDE uygula (dMSE)")
    lam_b = {}
    for b in tm.BLOKLAR:
        v = V[b.ad]
        r = np.mean(v["tohum_ofs"], axis=0)
        m = grup_ort(r, v["w"], v["tr"])
        duz = v["s"] - m
        e = v["g"] - r
        lam_b[b.ad] = wmean(e * duz, v["w"]) / wmean(duz**2, v["w"])
    print("  uydurulan lam*: " + "  ".join(f"{b.ad} {lam_b[b.ad]:+.3f}" for b in tm.BLOKLAR))
    print(f"  {'kaynak->hedef':22}{'lam':>8}{'dMSE':>11}")
    for kb in tm.BLOKLAR:
        for hb in tm.BLOKLAR:
            if kb.ad == hb.ad:
                continue
            v = V[hb.ad]
            r = np.mean(v["tohum_ofs"], axis=0)
            m = grup_ort(r, v["w"], v["tr"])
            duz = v["s"] - m
            m0 = wmean((v["g"] - r) ** 2, v["w"])
            m1 = wmean((v["g"] - (r + lam_b[kb.ad] * duz)) ** 2, v["w"])
            print(f"  {kb.ad + ' -> ' + hb.ad:22}{lam_b[kb.ad]:+8.3f}{m1 - m0:+11.5f}")

    # ---------------------------------------------------- 4 KIRPMA (lam = 1)
    print("\n" + "=" * 104)
    print("4) KIRPMA TABLOSU (lam = 1.0, artik hedefinin ZORLADIGI nokta)")
    print(f"  {'blok':7}{'K':>4}{'kalan_tr':>10}{'MSE_lam0':>11}{'MSE_lam1':>11}{'dMSE':>11}")
    for b in tm.BLOKLAR:
        v = V[b.ad]
        r = np.mean(v["tohum_ofs"], axis=0)
        m = grup_ort(r, v["w"], v["tr"])
        duz = v["s"] - m
        tr = v["tr"]
        dk = ((v["g"] - (r + duz)) ** 2 - (v["g"] - r) ** 2) * v["w"]
        katki = pd.Series(dk).groupby(tr).sum()
        srt = katki.abs().sort_values(ascending=False).index.to_numpy()
        for K in (0, 1, 5, 10, 25, 50):
            msk = ~tr.isin(set(srt[:K])).to_numpy()
            m0 = wmean((v["g"][msk] - r[msk]) ** 2, v["w"][msk])
            m1 = wmean((v["g"][msk] - (r[msk] + duz[msk])) ** 2, v["w"][msk])
            print(
                f"  {b.ad:7}{K:4d}{int(tr[msk].nunique()):10,}{m0:11.5f}{m1:11.5f}{m1 - m0:+11.5f}"
            )
        print()

    # --------------------------------------------------------- 5 dMSE CEVIRI
    print("=" * 104)
    print(f"5) dMSE CEVIRISI   p = {p_test:.4f}   taban MSE {MEVCUT_MSE:.5f} (RMSLE 1,01591)")
    print(f"  {'lam':>6}{'yerel dMSE':>13}{'p*dMSE':>11}{'yeni RMSLE':>13}")
    for la in LAMBDALAR[1:]:
        fark = []
        for b in tm.BLOKLAR:
            v = V[b.ad]
            for i in range(len(TOHUMLAR)):
                r = v["tohum_ofs"][i]
                m = grup_ort(r, v["w"], v["tr"])
                duz = v["s"] - m
                fark.append(
                    wmean((v["g"] - (r + la * duz)) ** 2, v["w"]) - wmean((v["g"] - r) ** 2, v["w"])
                )
        dm = float(np.mean(fark))
        print(
            f"  {la:>6.2f}{dm:+13.5f}{p_test * dm:+11.5f}{np.sqrt(MEVCUT_MSE + p_test * dm):13.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
