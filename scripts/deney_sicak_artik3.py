# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""SICAK ARTIK -- ADIM B: artik hangi (grup x gun) yapisinda? Bolunmus-yari GUVENILIRLIK.

Her bilesen icin cov(yariA, yariB) hesaplanir -- gurultu bu kovaryansa girmez,
yani TEKRARLANABILIR varyans dogrudan olculur ve MSE dusus tavani odur.
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
import deney_sicak_artik as sa  # noqa: E402
import tuketim_model as tm  # noqa: E402

RNG = np.random.default_rng(7)


def yari_kov(x: np.ndarray, hucre: pd.Series, bolen: np.ndarray) -> tuple[float, int]:
    """hucre = bilesen kimligi; bolen = 0/1 yari etiketi (hucre ICINDE bagimsiz)."""
    df = pd.DataFrame({"h": hucre.to_numpy(), "b": bolen, "x": x})
    m = df.groupby(["h", "b"])["x"].agg(["mean", "size"]).unstack("b")
    ort = m["mean"]
    n = m["size"]
    ok = ort.notna().all(axis=1) & (n >= 2).all(axis=1)
    if ok.sum() < 5:
        return 0.0, int(ok.sum())
    a = ort.loc[ok, 0].to_numpy()
    bb = ort.loc[ok, 1].to_numpy()
    w = n.loc[ok].sum(axis=1).to_numpy()
    a = a - np.average(a, weights=w)
    bb = bb - np.average(bb, weights=w)
    return float(np.average(a * bb, weights=w)), int(ok.sum())


def main() -> int:
    egitim, _ = d.cerceveleri_kur()
    for b in tm.BLOKLAR:
        v = sa.blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        trafo = dg["tanim"].to_numpy()
        gun = pd.to_datetime(dg["tarih"])
        gund = gun.values.astype("datetime64[D]")
        ga = sa.iki_yonlu_arindir(v["g"], trafo, gund)
        ra = sa.iki_yonlu_arindir(v["r"], trafo, gund)
        ea = ga - ra
        tv = float(ga.var())
        print(f"\n=== {b.ad}  artik var(GERCEK)={tv:.4f}   var(HATA)={float(ea.var()):.4f}")

        # trafo bazli rastgele yari (trafo x gun bilesenleri icin)
        tk = pd.factorize(trafo)[1]
        tmap = pd.Series(RNG.integers(0, 2, len(tk)), index=tk)
        yari_trafo = pd.Series(trafo).map(tmap).to_numpy()
        # gun bazli tek/cift (trafo x zaman bilesenleri icin)
        yari_gun = gun.dt.day.to_numpy() % 2

        gruplar = {
            "ilce x gun": (
                dg["ilce_key"].astype(str) + "|" + gun.dt.strftime("%Y%m%d"),
                yari_trafo,
            ),
            "kva_kova x gun": (
                pd.qcut(dg["guc"], 8, duplicates="drop").astype(str)
                + "|"
                + gun.dt.strftime("%Y%m%d"),
                yari_trafo,
            ),
            "on3 x gun": (
                dg["tanim_on3"].astype(str) + "|" + gun.dt.strftime("%Y%m%d"),
                yari_trafo,
            ),
            "yuk_fak x gun": (
                pd.qcut(dg["t_yuk_faktoru"].rank(method="first"), 8).astype(str)
                + "|"
                + gun.dt.strftime("%Y%m%d"),
                yari_trafo,
            ),
            "log_std x gun": (
                pd.qcut(dg["t_log_std"].rank(method="first"), 8).astype(str)
                + "|"
                + gun.dt.strftime("%Y%m%d"),
                yari_trafo,
            ),
            "TRAFO x ay": (pd.Series(trafo).astype(str) + "|" + gun.dt.strftime("%Y%m"), yari_gun),
            "TRAFO x hafta": (
                pd.Series(trafo).astype(str) + "|" + gun.dt.strftime("%Y%V"),
                yari_gun,
            ),
            "TRAFO x haftagunu": (
                pd.Series(trafo).astype(str) + "|hg" + gun.dt.dayofweek.astype(str),
                yari_gun,
            ),
        }
        print(f"  {'bilesen':22}{'kov(A,B)':>11}{'artik payi':>12}{'hucre':>9}")
        for ad, (hucre, bol) in gruplar.items():
            k, nh = yari_kov(ga, hucre, bol)
            print(f"  {ad:22}{k:>11.5f}{k / tv * 100:>11.1f}%{nh:>9,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
