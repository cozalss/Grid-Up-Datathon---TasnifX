# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""IDDIA SINAMASI-3: TRAFO ekseninin GENLIGI (v55'in tarifi, gun yerine trafo).

Iddia "trafo seviyesi tasinmiyor" derken a_i'yi BASKA bloktan almayi sinadi.
Ama v55'in tarifi baska: modelin KENDI urettigi trafo ofsetini c ile olcekle.
Bu etiketsiz uygulanabilir. Once c* bloklar arasi KARARLI mi?
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
import olcut  # noqa: E402
import tuketim_model as tm


def ayristir(x, w, t, g, tur=6):
    """Agirlikli iki yonlu ayrisim: mu, a_trafo, b_gun, eps."""
    ws = pd.Series(w)
    mu = float(np.average(x, weights=w))
    s = pd.Series(x - mu)
    a = pd.Series(0.0, index=s.index)
    b = pd.Series(0.0, index=s.index)
    for _ in range(tur):
        rr = s - b
        da = (rr * ws).groupby(t).transform("sum") / ws.groupby(t).transform("sum")
        a = da
        rr = s - a
        db = (rr * ws).groupby(g).transform("sum") / ws.groupby(g).transform("sum")
        b = db
    return mu, a.to_numpy(), b.to_numpy(), (s - a - b).to_numpy()


def main() -> int:
    egitim, test = d.cerceveleri_kur()
    gk = olcut.guc_kenarlari(test)
    tsicak = test[test["soguk_mu"] == 0]
    print(
        f"  {'blok':7}{'eksen':9}{'model std':>10}{'gercek std':>11}{'kor':>7}"
        f"{'oran':>7}{'c*=kor*oran':>12}{'olculen c*':>11}{'d(RMSLE)@c*':>13}{'tohum':>7}"
    )
    for b in tm.BLOKLAR:
        v = sa.blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        w, _ = olcut.test_agirliklari(dg, tsicak, gk)
        t = pd.Series(dg["tanim"].to_numpy())
        g = pd.Series(pd.to_datetime(dg["tarih"]).values.astype("datetime64[D]"))
        y = v["y"]
        lg = v["lg"]
        for eksen in ("TRAFO", "GUN"):
            kaz = {}
            istat = []
            for k in range(len(sa.TOHUMLAR)):
                r = v["tohum_loglari"][k] - lg
                mr, ar, br, er = ayristir(r, w, t, g)
                mg, ag, bg, eg = ayristir(v["g"], w, t, g)
                mr_, gr_ = (ar, ag) if eksen == "TRAFO" else (br, bg)
                sm = np.sqrt(np.average(mr_**2, weights=w))
                sg = np.sqrt(np.average(gr_**2, weights=w))
                kor = np.average(mr_ * gr_, weights=w) / (sm * sg)
                istat.append((sm, sg, kor))
                taban = olcut.agirlikli_rmsle(y, np.expm1(lg + r), w)
                for c in np.round(np.arange(0.4, 3.01, 0.05), 2):
                    yeni = lg + r + (c - 1.0) * mr_
                    kaz.setdefault(c, []).append(
                        taban - olcut.agirlikli_rmsle(y, np.expm1(yeni), w)
                    )
            sm, sg, kor = np.array(istat).mean(axis=0)
            en_c = max(kaz, key=lambda c: np.mean(kaz[c]))
            vv = np.array(kaz[en_c])
            print(
                f"  {b.ad:7}{eksen:9}{sm:10.4f}{sg:11.4f}{kor:+7.3f}{sg / sm:7.3f}"
                f"{kor * sg / sm:12.3f}{en_c:11.2f}{vv.mean():+13.5f}{int((vv > 0).sum()):5}/{len(vv)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
