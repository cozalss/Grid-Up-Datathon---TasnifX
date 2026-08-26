# ruff: noqa
"""EKSEN 5 -- HIPOTEZIN MEKANIZMA SINAMASI (onbellekten, FIT YOK).

Hipotez: model trafo SEVIYESINI merdivenle yaklastiriyor, bu yuzden
  (i)  seviyeyi ORTALAMAYA DOGRU SIKISTIRIYOR  -> b_i, sev_i ile TERS iliskili
  (ii) o sikismanin bedeli hatanin buyuk bir payi

Olcum: uretim harmani (aile_onbellek, 3 tohum, 5 uye) tahminleri uzerinde
  b_i  = ort(gercek ofs) - ort(tahmin ofs)      trafo bazinda
  s_i  = kesme oncesi son-90g pozitif ort ofs   (artik hedefin cikaracagi seviye)
  m_i  = ort(tahmin ofs)                        modelin ima ettigi seviye

  regresyon  b_i ~ a + c*(s_i - s_ort)      c<0 => model sikistiriyor
  regresyon  m_i ~ a + c*s_i                c<1 => merdiven sikismasi
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


def blok_verisi(egitim, blok):
    """deney_sicak_artik.blok_verisi ile birebir ayni (o modul import edilemiyor)."""
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
    log_t = np.mean(loglar, axis=0)
    lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    return {
        "cerceve": dg,
        "r": log_t - lg,
        "g": np.log1p(np.clip(y, 0, None)) - lg,
        "lg": lg,
        "y": y,
    }


CIK = KOK / "data" / "interim" / "eksen5"
PENCERELER = (90, 180, 365, 9999)


def sev_bagla(tanim: pd.Series, blok: str) -> tuple[np.ndarray, np.ndarray]:
    tab = pd.read_parquet(CIK / f"seviye_{blok}.parquet")
    sev = np.full(len(tanim), np.nan)
    kay = np.full(len(tanim), -1.0)
    for i, W in enumerate(PENCERELER):
        v = tanim.map(tab[f"sev{W}"]).to_numpy(dtype="float64")
        y = ~np.isfinite(sev) & np.isfinite(v)
        sev[y] = v[y]
        kay[y] = i
    return sev, kay


def wls(x, y, w):
    """Agirlikli dogrusal regresyon -> (kesme, egim, R2)."""
    W = w / w.sum()
    mx, my = np.dot(W, x), np.dot(W, y)
    sxx = np.dot(W, (x - mx) ** 2)
    sxy = np.dot(W, (x - mx) * (y - my))
    c = sxy / sxx
    a = my - c * mx
    yh = a + c * x
    r2 = 1 - np.dot(W, (y - yh) ** 2) / np.dot(W, (y - my) ** 2)
    return float(a), float(c), float(r2)


def main() -> int:
    egitim, test = d.cerceveleri_kur()
    gk = olcut.guc_kenarlari(test)
    tsicak = test[test["soguk_mu"] == 0]

    print("=" * 100)
    print("MEKANIZMA: uretim harmani, trafo bazinda yanlilik b_i ve seviye s_i")
    print("=" * 100)
    for b in tm.BLOKLAR:
        v = blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        tr = pd.Series(dg["tanim"].to_numpy())
        w, tani = olcut.test_agirliklari(dg, tsicak, gk)
        e = np.asarray(v["g"] - v["r"], dtype="float64")
        sev, kay = sev_bagla(tr, b.ad)
        ok = np.isfinite(sev)

        # trafo bazinda topla
        df = pd.DataFrame(
            {
                "tr": tr[ok].to_numpy(),
                "e": e[ok],
                "r": v["r"][ok],
                "g": v["g"][ok],
                "s": sev[ok],
                "w": w[ok],
            }
        )
        gb = df.groupby("tr")
        wt = gb["w"].sum()
        bi = gb.apply(lambda x: np.dot(x["w"], x["e"]) / x["w"].sum(), include_groups=False)
        mi = gb.apply(lambda x: np.dot(x["w"], x["r"]) / x["w"].sum(), include_groups=False)
        gi = gb.apply(lambda x: np.dot(x["w"], x["g"]) / x["w"].sum(), include_groups=False)
        si = gb["s"].first()
        W = wt.to_numpy()
        bi_, mi_, gi_, si_ = bi.to_numpy(), mi.to_numpy(), gi.to_numpy(), si.to_numpy()

        mse = float(np.dot(w[ok], e[ok] ** 2) / w[ok].sum())
        pay_b = float(np.dot(W, bi_**2) / W.sum()) / mse * 100

        a1, c1, r1 = wls(si_, bi_, W)
        a2, c2, r2 = wls(si_, mi_, W)
        a3, c3, r3 = wls(si_, gi_, W)
        print(
            f"\n--- {b.ad}  trafo {len(W):,}  satir {int(ok.sum()):,}  ESS {tani['ess_orani']:.3f}"
        )
        print(f"  MSE(sicak+sev) {mse:.5f}   TRAFO yanliliginin payi %{pay_b:.1f}")
        print(f"  b_i ~ s_i    egim {c1:+.4f}  R2 {r1:.4f}   (negatif = model SIKISTIRIYOR)")
        print(f"  m_i ~ s_i    egim {c2:+.4f}  R2 {r2:.4f}   (1'den kucuk = merdiven sikismasi)")
        print(f"  g_i ~ s_i    egim {c3:+.4f}  R2 {r3:.4f}   (GERCEGIN seviyeye duyarliligi)")
        print(
            f"  ort b_i {float(np.dot(W, bi_) / W.sum()):+.4f}"
            f"   std {float(np.sqrt(np.dot(W, (bi_ - np.dot(W, bi_) / W.sum()) ** 2) / W.sum())):.4f}"
        )
        # NE KADARI SEVIYE HATASI: b_i'yi s_i dogrusuyla acikla
        kalan = bi_ - (a1 + c1 * si_)
        print(
            f"  b_i'nin s_i ile aciklanan karesi: %{(1 - np.dot(W, kalan**2) / np.dot(W, bi_**2)) * 100:.1f}"
        )
        # TAVAN: b_i mukemmel bilinseydi
        tav = mse - float(np.dot(W, bi_**2) / W.sum())
        print(f"  TAVAN (mukemmel b_i): MSE {mse:.5f} -> {tav:.5f}  (dMSE {tav - mse:+.5f})")
        # SEV DOGRUSU ile duzeltme (uygulanabilir)
        duz = a1 + c1 * pd.Series(si_, index=bi.index)
        dsat = tr[ok].map(duz).to_numpy(dtype="float64")
        yeni = float(np.dot(w[ok], (e[ok] - dsat) ** 2) / w[ok].sum())
        print(f"  s_i DOGRUSU ile duzeltme: MSE -> {yeni:.5f}  (dMSE {yeni - mse:+.5f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
