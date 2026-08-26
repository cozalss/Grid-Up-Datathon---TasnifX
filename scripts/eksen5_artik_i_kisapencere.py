# ruff: noqa
"""EKSEN 5 -- EKSENIN EN GUCLU HALI: kisa pencereli seviye ile lam taramasi.

g_pencere olcumu s_7/s_14/s_30'un kis26'da m_i'yi ACIK ARA yendigini soyledi
(0,498 vs 0,728). Eger artik hedefi bir yerde kazanacaksa ORASIDIR. Burada
lam taramasini pencere x blok olarak yapiyoruz; AYNI trafo altkumesinde
(s_W tanimli) m_i ile karsilastirarak elma-elma tutuyoruz.
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
TOHUMLAR = (1000, 1001, 1002)
AGIRLIK = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 1.4}
PENC = (7, 14, 30, 90)
LAM = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
KESME = {"yaz25": "2025-03-31", "guz25": "2025-07-31", "kis26": "2025-11-30", "TEST": "2026-03-31"}
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


def seviye_tablosu(ham, kesme):
    k = pd.Timestamp(kesme)
    alt = ham[(ham["tarih"] <= k) & (ham["tarih"] >= pd.Timestamp(tm.EGITIM_BASI))]
    poz = alt[alt["tuketim"] > 0].copy()
    poz["_ofs"] = np.log1p(poz["tuketim"].to_numpy("float64")) - np.log1p(
        poz["guc"].to_numpy("float64")
    )
    out = None
    for W in PENC:
        p = poz[poz["tarih"] >= k - pd.Timedelta(days=W - 1)]
        t = p.groupby("tanim")["_ofs"].mean().rename(f"s{W}").to_frame()
        out = t if out is None else out.join(t, how="outer")
    return out


def wmean(x, w):
    return float(np.dot(w, np.asarray(x, "float64")) / w.sum())


def grup_ort(x, w, tr):
    xs = pd.Series(np.asarray(x, "float64"))
    s = (xs * w).groupby(tr).transform("sum")
    t = pd.Series(w).groupby(tr).transform("sum")
    duz = xs.groupby(tr).transform("mean")
    return np.where(t.to_numpy() > 0, (s / t.where(t > 0, 1.0)).to_numpy(), duz.to_numpy())


def main() -> int:
    ham, _ = tm.yukle()
    ham = ham[["tanim", "guc", "tarih", "tuketim"]].copy()
    ham["tarih"] = pd.to_datetime(ham["tarih"])
    egitim, test = d.cerceveleri_kur()
    gk = olcut.guc_kenarlari(test)
    tt = seviye_tablosu(ham, KESME["TEST"])
    sv = test["tanim"].map(tt["s90"]).to_numpy("float64")
    tsicak = test[((test["soguk_mu"] == 0).to_numpy()) & np.isfinite(sv)]
    p_test = {}
    for W in PENC:
        v = test["tanim"].map(tt[f"s{W}"]).to_numpy("float64")
        p_test[W] = float((((test["soguk_mu"] == 0).to_numpy()) & np.isfinite(v)).mean())
    print("  TEST etkilenen pay: " + "  ".join(f"s{W} {p_test[W]:.4f}" for W in PENC))

    print("\n" + "=" * 110)
    print("LAM TARAMASI x SEVIYE PENCERESI  (dMSE_agirlikli, kendi altkumesinde, tohum ort.)")
    print(
        f"  {'blok':7}{'pen':>5}{'satir':>9}"
        + "".join(f"{la:>9.2f}" for la in LAM[1:])
        + f"{'lam*':>8}{'dMSE*':>10}"
    )
    ESL = {}
    for b in tm.BLOKLAR:
        dg, ofsl, g = blok_verisi(egitim, b.ad)
        tab = seviye_tablosu(ham, KESME[b.ad])
        tr_all = pd.Series(dg["tanim"].to_numpy())
        for W in PENC:
            s_all = tr_all.map(tab[f"s{W}"]).to_numpy("float64")
            ok = np.isfinite(s_all)
            cer = dg[ok].reset_index(drop=True)
            w, _ = olcut.test_agirliklari(cer, tsicak, gk)
            tr = tr_all[ok].reset_index(drop=True)
            s = s_all[ok]
            gg = g[ok]
            satir, esl = [], []
            for i in range(len(TOHUMLAR)):
                r = ofsl[i][ok]
                m = grup_ort(r, w, tr)
                duz = s - m
                m0 = wmean((gg - r) ** 2, w)
                esl.append([wmean((gg - (r + la * duz)) ** 2, w) - m0 for la in LAM])
            esl = np.array(esl)  # tohum x lam
            ESL[(b.ad, W)] = esl
            r = np.mean([ofsl[i][ok] for i in range(len(TOHUMLAR))], axis=0)
            m = grup_ort(r, w, tr)
            duz = s - m
            e = gg - r
            ls = wmean(e * duz, w) / wmean(duz**2, w)
            dstar = wmean((e - ls * duz) ** 2, w) - wmean(e**2, w)
            print(
                f"  {b.ad:7}{W:>5}{int(ok.sum()):>9,}"
                + "".join(f"{x:+9.4f}" for x in esl.mean(axis=0)[1:])
                + f"{ls:+8.3f}{dstar:+10.4f}"
            )
        print()

    print("=" * 110)
    print("ESLENIK HUKUM -- (blok, tohum) 9 cift, pencere basina  (negatif = IYI)")
    print(
        f"  {'pen':>5}{'lam':>6}{'ort dMSE':>12}{'SH':>9}{'t':>7}{'neg':>8}{'yaz25(IKIZ)':>13}{'kis26(TEMIZ)':>14}{'yeni RMSLE':>12}"
    )
    for W in PENC:
        for j, la in enumerate(LAM):
            if la == 0.0:
                continue
            a = np.concatenate([ESL[(b.ad, W)][:, j] for b in tm.BLOKLAR])
            sh = a.std(ddof=1) / np.sqrt(len(a))
            iz = ESL[("yaz25", W)][:, j].mean()
            kz = ESL[("kis26", W)][:, j].mean()
            print(
                f"  {W:>5}{la:>6.2f}{a.mean():+12.5f}{sh:9.5f}{a.mean() / sh:+7.2f}"
                f"{int((a < 0).sum()):>5}/{len(a)}{iz:+13.5f}{kz:+14.5f}"
                f"{np.sqrt(MEVCUT_MSE + p_test[W] * a.mean()):12.5f}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
