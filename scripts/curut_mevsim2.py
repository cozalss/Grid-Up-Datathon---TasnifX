# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""CURUTME 2: mevsim agirliginin KOKEN / OZET-PENCERE / BAYATLIK kompozisyonu."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import deney as d
import tuketim_model as tm  # noqa: E402


def dairesel(tarih, hedef_doy):
    doy = pd.to_datetime(tarih).dt.dayofyear.to_numpy()
    h = np.unique(np.asarray(hedef_doy))
    f = np.abs(doy[:, None] - h[None, :])
    return np.minimum(f, 365 - f).min(axis=1).astype("float64")


egitim, test = d.cerceveleri_kur()
ek = d._ek_kokenler_kur(False)
ek = ek[ek["_blok"].isin([a for a, _, _ in tm.EK_KOKENLER])]
ortak = [k for k in egitim.columns if k in ek.columns]
genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)

TAU = 45.0
BAY = "t_son_kayit_yasi"


def rapor(ad, parca, hedef_df, hedef_ozet):
    hd = pd.to_datetime(hedef_df["tarih"])
    u = dairesel(parca["tarih"], hd.dt.dayofyear.unique())
    w = np.exp(-u / TAU)
    w = w / w.mean()
    print(f"\n--- {ad}  egitim {len(parca):,}  hedef ozet_pencere_gun={hedef_ozet}")
    print(
        f"    agirlik p10={np.quantile(w, 0.1):.3f} med={np.median(w):.3f} p90={np.quantile(w, 0.9):.3f}"
        f"  ESS={w.sum() ** 2 / (w**2).sum():,.0f} (%{100 * (w.sum() ** 2 / (w**2).sum()) / len(w):.1f})"
    )
    df = pd.DataFrame(
        {
            "koken": parca["_blok"].to_numpy(),
            "w": w,
            "ozet": parca["ozet_pencere_gun"].to_numpy(),
            "bay": parca[BAY].to_numpy() if BAY in parca.columns else np.nan,
        }
    )
    g = df.groupby("koken").agg(n=("w", "size"), wsum=("w", "sum"), ozet=("ozet", "first"))
    g["duz_pay"] = 100 * g["n"] / len(df)
    g["agr_pay"] = 100 * g["wsum"] / df["w"].sum()
    g["kat"] = g["agr_pay"] / g["duz_pay"]
    print(
        g[["n", "ozet", "duz_pay", "agr_pay", "kat"]]
        .sort_values("ozet")
        .to_string(float_format=lambda x: f"{x:8.2f}")
    )
    om_d = df["ozet"].mean()
    om_a = (df["ozet"] * df["w"]).sum() / df["w"].sum()
    print(
        f"    ozet_pencere_gun  duz ort {om_d:7.1f}   AGIRLIKLI {om_a:7.1f}   HEDEF {hedef_ozet}"
        f"   -> agirlik hedeften {'UZAKLASTIRIYOR' if abs(om_a - hedef_ozet) > abs(om_d - hedef_ozet) else 'YAKINLASTIRIYOR'}"
        f" ({abs(om_d - hedef_ozet):.0f} -> {abs(om_a - hedef_ozet):.0f} gun)"
    )
    if BAY in parca.columns:
        m = np.isfinite(df["bay"].to_numpy())
        bd = df.loc[m, "bay"]
        bw = df.loc[m, "w"]
        p_d = 100 * (bd >= 1).mean()
        p_a = 100 * (bw * (bd >= 1)).sum() / bw.sum()
        print(f"    bayat (t_son_kayit_yasi>=1) pay  duz %{p_d:.1f}  AGIRLIKLI %{p_a:.1f}")


for b in tm.BLOKLAR:
    dog = egitim[egitim["_blok"] == b.ad]
    parca = tm.kokenleri_ayikla(genis, b.ad)
    rapor(b.ad, parca, dog, float(dog["ozet_pencere_gun"].iloc[0]))

rapor("URETIM->TEST", genis, test, float(test["ozet_pencere_gun"].iloc[0]))

# test tarafinda bayatlik referansi
if BAY in test.columns:
    m = np.isfinite(test[BAY].to_numpy())
    print(f"\nTEST bayat pay %{100 * (test.loc[m, BAY] >= 1).mean():.1f}")
