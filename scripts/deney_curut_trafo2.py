# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""IDDIA SINAMASI-2: trafo yanliligi KIMLIK olarak degil, FONKSIYON olarak tasinir mi?

Iddia yalnizca "trafo -> a_i" ARAMA TABLOSUNU blok-disi sinadi. Duzeltme bir
arama tablosu olmak zorunda degil: a_i tahmin aninda GOZLENEBILIR degiskenlerin
bir fonksiyonuysa, o fonksiyon bloklar arasi tasinabilir.

Protokol: artik e = log1p(y) - log1p(tahmin), IKI blokta ogrenilir (test
karisimi agirligiyla), UCUNCU blokta uygulanir. lambda izgarasi + eslenik SH.
Referans olarak: (0) yalnizca genel ortalama, (1) bayatlik kovasi ortalamasi.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import deney as d
import deney_sicak_artik as sa  # noqa: E402
import lightgbm as lgb  # noqa: E402
import olcut  # noqa: E402
import tuketim_model as tm

OZ = [
    "t_son_kayit_yasi",
    "ufuk_gun",
    "yas",
    "t_gun_sayisi",
    "t_doluluk",
    "t_log_ort",
    "t_log_std",
    "t_trend",
    "t_sifir_orani",
    "t_yuk_faktoru",
    "t_olu_mu",
    "t_kuyruk_sifir",
    "t_yayilma",
    "t_kayma",
    "t_son7_gun",
    "t_son30_gun",
    "t_son90_gun",
    "p_doluluk",
    "p_yayilma",
    "p_son_ofset",
    "guc",
    "guc_yuzdelik",
    "t_mevsim_genlik",
    "t_hg_genligi",
]


def main() -> int:
    egitim, test = d.cerceveleri_kur()
    gk = olcut.guc_kenarlari(test)
    tsicak = test[test["soguk_mu"] == 0]
    adlar = [b.ad for b in tm.BLOKLAR]
    D = {}
    for b in tm.BLOKLAR:
        v = sa.blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        w, tani = olcut.test_agirliklari(dg, tsicak, gk)
        X = dg[OZ].apply(pd.to_numeric, errors="coerce").astype("float32")
        bay = olcut._kova(
            dg["t_son_kayit_yasi"].to_numpy(dtype="float64"), olcut.BAYATLIK_KENARLARI
        )
        D[b.ad] = dict(v=v, dg=dg, w=w, X=X, bay=bay, ess=tani["ess_orani"])
        print(f"  {b.ad}  n={len(dg):,}  ESS={tani['ess_orani']:.2f}")

    print("\n" + "=" * 92)
    print("BLOK-DISI DUZELTME (iki blokta ogren, ucuncude uygula) -- agirlikli sicak RMSLE farki")
    print(
        f"  {'blok':7}{'yontem':12}{'lam':>6}{'d(RMSLE)':>11}{'SH':>9}{'t':>7}{'tohum':>7}"
        f"{'ic-blok orakul':>16}"
    )
    for b in tm.BLOKLAR:
        ad = b.ad
        S = D[ad]
        w = S["w"]
        y = S["v"]["y"]
        lg = S["v"]["lg"]
        kaynak = [o for o in adlar if o != ad]
        for yontem in ("sabit", "bayatlik", "lgbm"):
            farklar = {lam: [] for lam in (0.25, 0.50, 1.00)}
            orakul = []
            for k in range(len(sa.TOHUMLAR)):
                # kaynak bloklardan duzeltme ogren
                if yontem == "lgbm":
                    Xk = pd.concat([D[o]["X"] for o in kaynak], ignore_index=True)
                    ek = np.concatenate(
                        [
                            D[o]["v"]["g"] - (D[o]["v"]["tohum_loglari"][k] - D[o]["v"]["lg"])
                            for o in kaynak
                        ]
                    )
                    wk = np.concatenate([D[o]["w"] for o in kaynak])
                    m = lgb.LGBMRegressor(
                        n_estimators=200,
                        num_leaves=8,
                        learning_rate=0.05,
                        min_child_samples=500,
                        verbose=-1,
                        random_state=7,
                    )
                    m.fit(Xk, ek, sample_weight=wk)
                    dz = m.predict(S["X"])
                else:
                    parc = []
                    for o in kaynak:
                        eo = D[o]["v"]["g"] - (D[o]["v"]["tohum_loglari"][k] - D[o]["v"]["lg"])
                        parc.append(pd.DataFrame({"e": eo, "w": D[o]["w"], "b": D[o]["bay"]}))
                    P = pd.concat(parc, ignore_index=True)
                    if yontem == "sabit":
                        s = float(np.average(P["e"], weights=P["w"]))
                        dz = np.full(len(S["dg"]), s)
                    else:
                        g = P.groupby("b").apply(lambda z: np.average(z["e"], weights=z["w"]))
                        dz = (
                            pd.Series(S["bay"])
                            .map(g)
                            .fillna(float(np.average(P["e"], weights=P["w"])))
                            .to_numpy()
                        )
                r = S["v"]["tohum_loglari"][k] - lg
                taban = olcut.agirlikli_rmsle(y, np.expm1(lg + r), w)
                for lam in farklar:
                    farklar[lam].append(
                        taban - olcut.agirlikli_rmsle(y, np.expm1(lg + r + lam * dz), w)
                    )
                en = min(
                    olcut.agirlikli_rmsle(y, np.expm1(lg + r + l2 * dz), w)
                    for l2 in np.arange(-1.0, 2.01, 0.1)
                )
                orakul.append(taban - en)
            for lam in (0.25, 0.50, 1.00):
                v_ = np.array(farklar[lam])
                sh = v_.std(ddof=1) / np.sqrt(len(v_))
                print(
                    f"  {ad:7}{yontem:12}{lam:6.2f}{v_.mean():+11.5f}{sh:9.5f}"
                    f"{v_.mean() / max(sh, 1e-12):7.2f}{int((v_ > 0).sum()):5}/{len(v_)}"
                    f"{np.mean(orakul):+16.5f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
