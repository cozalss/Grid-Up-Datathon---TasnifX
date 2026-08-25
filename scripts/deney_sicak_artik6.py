# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""SICAK ARTIK -- ADIM E: hatanin AYRISIMI (gun / trafo / etkilesim) x UFUK ve
model TAZE SAPMAYI ufka gore dogru soneltiyor mu?"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import deney as d
import deney_sicak_artik as sa  # noqa: E402
import tuketim_model as tm  # noqa: E402


def main() -> int:
    egitim, _ = d.cerceveleri_kur()
    for b in tm.BLOKLAR:
        v = sa.blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        trafo = pd.Series(dg["tanim"].to_numpy())
        gun = pd.to_datetime(dg["tarih"])
        gund = gun.values.astype("datetime64[D]")
        e = pd.Series(v["g"] - v["r"])
        mu = e.mean()
        e0 = e - mu
        a = e0.groupby(trafo).transform("mean")  # trafo etkisi (kaba)
        bd = (e0 - a).groupby(pd.Series(gund)).transform("mean")
        eps = e0 - a - bd
        mse = float((e**2).mean())
        print(f"\n=== {b.ad}  sicak MSE {mse:.5f}")
        print(
            f"  ayrisim: sabit {mu**2 / mse * 100:5.1f}%   TRAFO {float((a**2).mean()) / mse * 100:5.1f}%"
            f"   GUN {float((bd**2).mean()) / mse * 100:5.1f}%   ETKILESIM {float((eps**2).mean()) / mse * 100:5.1f}%"
        )

        uf = pd.cut(dg["ufuk_gun"], [0, 15, 31, 61, 91, 200])
        t = pd.DataFrame(
            {
                "uf": uf,
                "e": e.to_numpy(),
                "a": a.to_numpy(),
                "bd": bd.to_numpy(),
                "eps": eps.to_numpy(),
            }
        )
        g = t.groupby("uf", observed=True)
        o = g.agg(
            n=("e", "size"),
            mse=("e", lambda x: float((x**2).mean())),
            yanlilik=("e", "mean"),
            trafo2=("a", lambda x: float((x**2).mean())),
            gun2=("bd", lambda x: float((x**2).mean())),
            etk2=("eps", lambda x: float((x**2).mean())),
        )
        print(o.to_string(float_format=lambda x: f"{x:9.4f}"))

        # TAZE SAPMA: model ufka gore dogru sonelttiyor mu?
        ort = dg["t_log_ort"].to_numpy(dtype="float64")
        print("  TAZE SAPMA ile HATA iliskisi (ufka gore OLS egimi; 0 = model dogru kullanmis)")
        satirlar = {}
        for ad, k in (("son7", "t_log_son7"), ("son30", "t_log_son30"), ("son90", "t_log_son90")):
            z = dg[k].to_numpy(dtype="float64") - ort
            ok = np.isfinite(z)
            satirlar[ad] = (z, ok)
        satirlar["trend"] = (
            dg["t_trend"].to_numpy(dtype="float64"),
            np.isfinite(dg["t_trend"].to_numpy(dtype="float64")),
        )
        bas = f"  {'z':8}{'dolu%':>7}"
        kova = list(o.index)
        for kv in kova:
            bas += f"{str(kv):>14}"
        print(bas)
        for ad, (z, ok) in satirlar.items():
            sat = f"  {ad:8}{ok.mean() * 100:6.1f}%"
            for kv in kova:
                m = ok & (uf == kv).to_numpy()
                if m.sum() < 200 or np.std(z[m]) < 1e-9:
                    sat += f"{'-':>14}"
                    continue
                eg = float(np.polyfit(z[m], e.to_numpy()[m], 1)[0])
                c = float(np.corrcoef(z[m], e.to_numpy()[m])[0, 1])
                sat += f"{eg:+7.3f}/{c:+.2f}"
            print(sat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
