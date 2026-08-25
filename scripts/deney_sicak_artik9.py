# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""SICAK ARTIK -- ADIM H: blok ici kazanc NEREDEN geliyor ve ZAMANDA ILERI tasiniyor mu?

A) oznitelik altkumesi kirilimi (trafo yarisi capraz, ayni gunler)
B) ZAMANDA ILERI: blogun ILK yarisinda uydur, IKINCI yarisinda olc
   (+ trafo yarisi da ayrilarak: hem gorulmemis trafo hem gorulmemis gun)
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
import olcut  # noqa: E402
import tuketim_model as tm
from deney_sicak_artik7 import z_kur  # noqa: E402

RNG = np.random.default_rng(11)
KUMELER = {
    "hepsi": None,
    "statik trafo": [
        "t_log_ort",
        "t_log_std",
        "t_yuk_faktoru",
        "t_hg_genligi",
        "log_guc",
        "t_yayilma",
        "t_sifir_orani",
        "t_gun_sayisi",
        "t_doluluk",
    ],
    "yalniz ufuk": ["ufuk"],
    "statik+ufuk": [
        "t_log_ort",
        "t_log_std",
        "t_yuk_faktoru",
        "t_hg_genligi",
        "log_guc",
        "t_yayilma",
        "t_sifir_orani",
        "t_gun_sayisi",
        "t_doluluk",
        "ufuk",
    ],
    "yalniz taze sapma": ["sapma_s7", "sapma_s14", "sapma_s30", "sapma_s60", "sapma_s90"],
    "yalniz hava": ["sicaklik_ort", "cdd22", "t_egim_sicaklik_ort", "egim_x_sic"],
}


def uy(X, y, Xt):  # noqa: ANN001
    import lightgbm as lgb

    m = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        verbose=-1,
        random_state=0,
    )
    m.fit(X, y)
    return m.predict(Xt)


def main() -> int:
    egitim, test = d.cerceveleri_kur()
    gk = olcut.guc_kenarlari(test)
    tsicak = test[test["soguk_mu"] == 0]
    for b in tm.BLOKLAR:
        v = sa.blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        z = z_kur(dg)
        e = v["g"] - v["r"]
        trafo = dg["tanim"].to_numpy()
        gun = pd.to_datetime(dg["tarih"])
        orta = gun.min() + (gun.max() - gun.min()) / 2
        ikinci = (gun > orta).to_numpy()
        tk = pd.unique(trafo)
        h = pd.Series(trafo).map(pd.Series(RNG.integers(0, 2, len(tk)), index=tk)).to_numpy()
        w, _ = olcut.test_agirliklari(dg, tsicak, gk)
        taban = olcut.agirlikli_rmsle(v["y"], np.expm1(v["lg"] + v["r"]), w)
        print(f"\n=== {b.ad}   agirlikli taban {taban:.5f}   ikinci yari n={int(ikinci.sum()):,}")
        print(
            f"  {'kume':20}{'a=0,25':>10}{'a=0,50':>10}{'a=1,00':>10}{'R2%':>8}  (trafo yarisi, ayni gunler)"
        )
        for ad, ks in KUMELER.items():
            zz = z if ks is None else z[ks]
            eh = np.zeros(len(dg))
            for kat in (0, 1):
                eh[h != kat] = uy(zz[h == kat], e[h == kat], zz[h != kat])
            c = float(np.corrcoef(eh, e)[0, 1])
            fk = [
                olcut.agirlikli_rmsle(v["y"], np.expm1(v["lg"] + v["r"] + a * eh), w) - taban
                for a in (0.25, 0.5, 1.0)
            ]
            print(f"  {ad:20}{fk[0]:+10.5f}{fk[1]:+10.5f}{fk[2]:+10.5f}{c * c * 100:8.2f}")
        # ZAMANDA ILERI
        print("  ZAMANDA ILERI (ilk yaride uydur -> ikinci yaride olc)")
        wi, _ = olcut.test_agirliklari(dg[ikinci], tsicak, gk)
        tab_i = olcut.agirlikli_rmsle(
            v["y"][ikinci], np.expm1(v["lg"][ikinci] + v["r"][ikinci]), wi
        )
        for ad, kosul in (
            ("ayni trafolar", np.ones(len(dg), bool)),
            ("gorulmemis trafolar", h == 1),
        ):
            egit = (~ikinci) & (h == 0) if ad == "gorulmemis trafolar" else (~ikinci)
            olc = ikinci & kosul
            eh = np.zeros(len(dg))
            eh[olc] = uy(z[egit], e[egit], z[olc])
            fk = []
            for a in (0.25, 0.5, 1.0):
                tah = np.expm1(v["lg"] + v["r"] + a * eh)
                fk.append(olcut.agirlikli_rmsle(v["y"][ikinci], tah[ikinci], wi) - tab_i)
            c = float(np.corrcoef(eh[olc], e[olc])[0, 1])
            print(
                f"    {ad:22} taban {tab_i:.5f}  a025 {fk[0]:+.5f}  a050 {fk[1]:+.5f}"
                f"  a100 {fk[2]:+.5f}   R2={c * c * 100:.2f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
