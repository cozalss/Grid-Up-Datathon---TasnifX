# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""SICAK ARTIK -- ADIM M: TEK KAYNAKLI transfer + zaman komsulugu.

Iddia yalnizca 'DIGER IKI BLOGUN ORTALAMASI' ile lambda>=0,25 denemis.
Burada: her (hedef,kaynak) cifti AYRI, kucuk lambda, TOHUM BAZINDA, ve
zaman-komsulugunun korelasyonu nasil degistirdigi.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import deney as d
import deney_ileri as di  # noqa: E402
import deney_sicak_artik as sa  # noqa: E402
import olcut  # noqa: E402
import tuketim_model as tm

TOHUMLAR = (1000, 1001, 1002)
ONB = KOK / "data" / "interim" / "aile_onbellek"
AGIRLIK = sa.AGIRLIK


def tohum_verisi(egitim: pd.DataFrame, blok: str, tohum: int) -> dict:
    _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, blok)
    sicak = ~soguk
    dg = dogrulama[sicak].reset_index(drop=True)
    y = gercek[sicak]
    pay = sum(AGIRLIK.values())
    s = np.zeros(len(dg), dtype="float64")
    for a, w in AGIRLIK.items():
        s += w * np.load(ONB / f"{blok}_{tohum}_{a}_uretim.npy").astype("float64")
    log_t = s / pay
    lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    return {
        "cerceve": dg,
        "r": log_t - lg,
        "g": np.log1p(np.clip(y, 0, None)) - lg,
        "lg": lg,
        "y": y,
    }


def main() -> int:
    egitim, test = d.cerceveleri_kur()
    gk = olcut.guc_kenarlari(test)
    tsicak = test[test["soguk_mu"] == 0]
    adlar = [b.ad for b in tm.BLOKLAR]

    D = {}
    for b in adlar:
        for s in TOHUMLAR:
            v = tohum_verisi(egitim, b, s)
            dg = v["cerceve"]
            w, _ = olcut.test_agirliklari(dg, tsicak, gk)
            t = pd.Series(dg["tanim"].to_numpy())
            e = pd.Series(v["g"] - v["r"])
            D[(b, s)] = {
                "v": v,
                "w": w,
                "t": t,
                "e": e,
                "a": e.groupby(t).mean(),
                "n": e.groupby(t).size(),
            }

    print("C1) TEK KAYNAKLI a_i, TOHUM BAZINDA korelasyon (ortak trafolar)")
    for i in range(3):
        for j in range(i + 1, 3):
            rs = []
            for s in TOHUMLAR:
                x = pd.concat(
                    [D[(adlar[i], s)]["a"], D[(adlar[j], s)]["a"]], axis=1, join="inner"
                ).dropna()
                x.columns = ["a", "b"]
                rs.append(
                    (float(x["a"].corr(x["b"])), float(np.polyfit(x["a"], x["b"], 1)[0]), len(x))
                )
            print(
                f"    {adlar[i]:6} x {adlar[j]:6}  n={rs[0][2]:,}  "
                f"kor {' '.join(f'{r[0]:+.3f}' for r in rs)}   "
                f"OLS {' '.join(f'{r[1]:+.3f}' for r in rs)}"
            )

    print(
        "\nC2) TEK KAYNAKLI UYGULAMA  tahmin' = tahmin + lambda * a_i^kaynak  (agirlikli RMSLE farki)"
    )
    print(
        f"  {'hedef':8}{'kaynak':8}{'lam':>6}{'t1000':>10}{'t1001':>10}{'t1002':>10}{'ort':>10}{'SH':>9}{'t':>7}{'kapsam%':>9}"
    )
    ozet = {}
    for hedef in adlar:
        for kaynak in adlar:
            if kaynak == hedef:
                continue
            for lam in (0.10, 0.15, 0.20, 0.30):
                farklar = []
                for s in TOHUMLAR:
                    dh = D[(hedef, s)]
                    ai = D[(kaynak, s)]["a"]
                    d_ai = dh["t"].map(ai).fillna(0.0).to_numpy()
                    v = dh["v"]
                    taban = olcut.agirlikli_rmsle(v["y"], np.expm1(v["lg"] + v["r"]), dh["w"])
                    yeni = olcut.agirlikli_rmsle(
                        v["y"], np.expm1(v["lg"] + v["r"] + lam * d_ai), dh["w"]
                    )
                    farklar.append(yeni - taban)
                kapsam = float(D[(hedef, 1000)]["t"].map(D[(kaynak, 1000)]["a"]).notna().mean())
                f = np.array(farklar)
                sh = float(f.std(ddof=1) / np.sqrt(len(f)))
                tst = float(f.mean() / sh) if sh > 0 else np.nan
                print(
                    f"  {hedef:8}{kaynak:8}{lam:6.2f}"
                    f"{f[0]:+10.5f}{f[1]:+10.5f}{f[2]:+10.5f}{f.mean():+10.5f}{sh:9.5f}{tst:+7.2f}{kapsam * 100:9.1f}"
                )
                ozet[(hedef, kaynak, lam)] = f
            print()

    print("C3) ZAMAN KOMSULUGU: blok yarilari arasi a_i korelasyonu (tohum ort a_i)")
    A2 = {}
    for b in adlar:
        dh = D[(b, 1000)]
        e = sum(D[(b, s)]["e"] for s in TOHUMLAR) / 3.0
        gn = pd.to_datetime(dh["v"]["cerceve"]["tarih"]).values.astype("datetime64[D]")
        gunler = np.sort(pd.unique(gn))
        srt = pd.Series(gn).map({g: i for i, g in enumerate(gunler)}).to_numpy()
        on = srt < len(gunler) / 2
        A2[(b, "on")] = e[on].groupby(dh["t"][on]).mean()
        A2[(b, "arka")] = e[~on].groupby(dh["t"][~on]).mean()
    ciftler = [
        ("guz25", "arka", "kis26", "on"),
        ("guz25", "on", "kis26", "arka"),
        ("guz25", "arka", "kis26", "arka"),
        ("guz25", "on", "kis26", "on"),
        ("yaz25", "arka", "guz25", "on"),
        ("yaz25", "on", "guz25", "arka"),
    ]
    for b1, y1, b2, y2 in ciftler:
        x = pd.concat([A2[(b1, y1)], A2[(b2, y2)]], axis=1, join="inner").dropna()
        x.columns = ["a", "b"]
        print(f"    {b1}/{y1:5} x {b2}/{y2:5}  n={len(x):,}  kor={float(x['a'].corr(x['b'])):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
