# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""SICAK ARTIK -- ADIM F: KARAR OLCUMU. Sicak HATA ikinci asamada ogrenilebilir mi?

Blok-disi protokol: iki blogun sicak hatasi uzerinde kucuk bir model uydurulur,
UCUNCU blokta uygulanir. Uc blokta da kazanmiyorsa REDDEDILIR.
Olcut testin karisimina agirliklandirilmis (scripts/olcut.py).
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

ALFALAR = (0.0, 0.25, 0.50, 1.00)


def z_kur(dg: pd.DataFrame) -> pd.DataFrame:
    ort = dg["t_log_ort"].to_numpy(dtype="float64")
    uf = dg["ufuk_gun"].to_numpy(dtype="float64")
    z = pd.DataFrame(index=dg.index)
    for ad, k in (
        ("s7", "t_log_son7"),
        ("s14", "t_log_son14"),
        ("s30", "t_log_son30"),
        ("s60", "t_log_son60"),
        ("s90", "t_log_son90"),
    ):
        z[f"sapma_{ad}"] = dg[k].to_numpy(dtype="float64") - ort
    for tau in (20.0, 45.0, 90.0):
        z[f"sonuk30_{int(tau)}"] = z["sapma_s30"] * np.exp(-uf / tau)
        z[f"sonuk7_{int(tau)}"] = z["sapma_s7"] * np.exp(-uf / tau)
    z["ufuk"] = uf
    z["log_guc"] = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    for k in (
        "t_log_ort",
        "t_log_std",
        "t_yuk_faktoru",
        "t_trend",
        "t_yayilma",
        "t_sifir_orani",
        "t_gun_sayisi",
        "t_hg_genligi",
        "t_son_kayit_yasi",
        "t_kayma",
        "sicaklik_ort",
        "cdd22",
        "t_egim_sicaklik_ort",
        "t_doluluk",
    ):
        z[k] = dg[k].to_numpy(dtype="float64")
    z["egim_x_sic"] = z["t_egim_sicaklik_ort"] * z["sicaklik_ort"]
    return z


def main() -> int:
    import lightgbm as lgb

    egitim, test = d.cerceveleri_kur()
    gk = olcut.guc_kenarlari(test)
    veri = {}
    for b in tm.BLOKLAR:
        v = sa.blok_verisi(egitim, b.ad)
        veri[b.ad] = v
        v["z"] = z_kur(v["cerceve"])
        v["e"] = v["g"] - v["r"]
    tsicak = test[test["soguk_mu"] == 0]
    print(f"  test sicak {len(tsicak):,}")

    print(f"\n  {'blok':8}{'alfa':>6}{'duz RMSLE':>12}{'agirlikli':>12}{'ESS':>8}{'fark(ag)':>11}")
    toplam = {a: 0.0 for a in ALFALAR}
    for b in tm.BLOKLAR:
        kaynak = [o.ad for o in tm.BLOKLAR if o.ad != b.ad]
        X = pd.concat([veri[k]["z"] for k in kaynak], ignore_index=True)
        y = np.concatenate([veri[k]["e"] for k in kaynak])
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
        v = veri[b.ad]
        e_hat = m.predict(v["z"])
        dg = v["cerceve"]
        w, tani = olcut.test_agirliklari(dg, tsicak, gk)
        gercek = v["y"]
        taban_ag = None
        for a in ALFALAR:
            tah = np.expm1(v["lg"] + v["r"] + a * e_hat)
            duz = olcut.agirlikli_rmsle(gercek, tah)
            ag = olcut.agirlikli_rmsle(gercek, tah, w)
            if a == 0.0:
                taban_ag = ag
            toplam[a] += ag
            print(
                f"  {b.ad:8}{a:6.2f}{duz:12.5f}{ag:12.5f}{tani['ess_orani']:8.2f}"
                f"{ag - taban_ag:+11.5f}"
            )
        onem = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(f"    en onemli 6: {', '.join(onem.index[:6])}")
    print("\n  ORTALAMA (uc blok, agirlikli)")
    for a in ALFALAR:
        print(
            f"    alfa {a:4.2f}  {toplam[a] / 3:.5f}  fark {toplam[a] / 3 - toplam[0.0] / 3:+.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
