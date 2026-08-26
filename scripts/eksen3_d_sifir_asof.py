# ruff: noqa
"""EKSEN 3 (d) DUZELTME: sifir kovasi KESME ONCESI (as-of) tanimlanir -- kural 7.

Onceki kosuda kova hedef penceresinin sifir oraniyla tanimlanmisti; bu SIZINTI.
Burada kova, kesme aninda BITEN pencereden gelir; bloklar arasi isaret
tutarliligi ancak boyle sinanabilir.

    python scripts/eksen3_d_sifir_asof.py
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
import tuketim_model as tm  # noqa: E402

AILELER = ("cat", "xgb", "lgbm")
AGIRLIK = (3.0, 1.0, 1.0)
ONBELLEK = KOK / "data" / "interim" / "deney" / "sicak_tahmin.npz"
KOVALAR = [-0.01, 1e-9, 0.05, 0.25, 0.75, 0.95, 1.01]
ETIKET = ["kesme oncesi hic sifir yok", "<=5%", "5-25%", "25-75%", "75-95%", ">95%"]


def main() -> int:
    z = np.load(ONBELLEK)
    egitim, _ = d.cerceveleri_kur()
    ham, _t = tm.yukle()
    ham["tarih"] = pd.to_datetime(ham["tarih"])
    ham["sifir"] = (ham["tuketim"] <= 0).astype("float64")

    hepsi = {}
    for b in tm.BLOKLAR:
        kesme = pd.Timestamp(b.etiket_basi)
        _, dv, gr, sk = di.blok_parcalari(egitim, b.ad)
        dg = dv[~sk].reset_index(drop=True)
        lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
        pay = sum(AGIRLIK)
        r = (
            np.mean(
                [
                    sum(AGIRLIK[i] * z[f"{b.ad}_{t}_{a}"] for i, a in enumerate(AILELER)) / pay
                    for t in di.TOHUMLAR
                ],
                axis=0,
            )
            - lg
        )
        g = np.log1p(np.clip(gr[~sk], 0, None)) - lg
        T = (
            pd.DataFrame({"tanim": dg["tanim"].to_numpy(), "e": g - r})
            .groupby("tanim")
            .agg(b=("e", "mean"), n=("e", "size"))
        )
        gec = ham[ham["tarih"] < kesme]
        p90 = gec[gec["tarih"] >= kesme - pd.Timedelta(days=90)]
        T["sifir_asof90"] = p90.groupby("tanim")["sifir"].mean()
        T["sifir_asof_tum"] = gec.groupby("tanim")["sifir"].mean()
        T["kuyruk"] = (kesme - gec[gec["tuketim"] > 0].groupby("tanim")["tarih"].max()).dt.days
        hepsi[b.ad] = T.dropna(subset=["sifir_asof90"])

    for anahtar in ("sifir_asof90", "sifir_asof_tum"):
        print("=" * 100)
        print(f"(d) AS-OF SIFIR KOVASI: {anahtar}  -- b (satir agirlikli), uc blok")
        print("=" * 100)
        print(f"  {'kova':<28}" + "".join(f"{ad:>22}" for ad in ("yaz25", "guz25", "kis26")))
        print(f"  {'':<28}" + "".join(f"{'pay':>8}{'b':>8}{'trafo':>6}" for _ in range(3)))
        for lo, hi, et in zip(KOVALAR[:-1], KOVALAR[1:], ETIKET):
            satir = f"  {et:<28}"
            for ad in ("yaz25", "guz25", "kis26"):
                T = hepsi[ad]
                s = T[(T[anahtar] > lo) & (T[anahtar] <= hi)]
                if not len(s):
                    satir += f"{'-':>8}{'-':>8}{0:>6}"
                    continue
                pay = s["n"].sum() / T["n"].sum()
                bb = float((s["b"] * s["n"]).sum() / s["n"].sum())
                satir += f"{pay:>8.4f}{bb:>+8.3f}{len(s):>6}"
            print(satir)
        satir = f"  {'GENEL':<28}"
        for ad in ("yaz25", "guz25", "kis26"):
            T = hepsi[ad]
            satir += f"{1.0:>8.4f}{float((T['b'] * T['n']).sum() / T['n'].sum()):>+8.3f}{len(T):>6}"
        print(satir + "\n")

    print("=" * 100)
    print("(d) AS-OF OLU KUYRUK (kesme oncesi son POZITIF kayittan bu yana gun)")
    print("=" * 100)
    print(f"  {'kuyruk (gun)':<20}" + "".join(f"{ad:>22}" for ad in ("yaz25", "guz25", "kis26")))
    for lo, hi in ((1, 7), (8, 30), (31, 60), (61, 120), (121, 9999)):
        satir = f"  {f'{lo}-{hi}':<20}"
        for ad in ("yaz25", "guz25", "kis26"):
            T = hepsi[ad]
            s = T[T["kuyruk"].between(lo, hi)]
            if not len(s):
                satir += f"{'-':>8}{'-':>8}{0:>6}"
                continue
            pay = s["n"].sum() / T["n"].sum()
            bb = float((s["b"] * s["n"]).sum() / s["n"].sum())
            satir += f"{pay:>8.4f}{bb:>+8.3f}{len(s):>6}"
        print(satir)

    print()
    print("=" * 100)
    print("SABIT DELTA'nin >75% AS-OF SIFIR KOVASINDA dMSE'si (uc blok, capraz)")
    print("=" * 100)
    print(
        f"  {'blok':<8}{'pay':>9}{'b':>9}{'m0':>10}{'m1(delta=b)':>13}{'dMSE(hot)':>12}{'dMSE(tum)':>12}"
    )
    for ad in ("yaz25", "guz25", "kis26"):
        T = hepsi[ad]
        s = T[T["sifir_asof90"] > 0.75]
        if not len(s):
            print(f"  {ad:<8} kova bos")
            continue
        pay = float(s["n"].sum() / T["n"].sum())
        bb = float((s["b"] * s["n"]).sum() / s["n"].sum())
        print(
            f"  {ad:<8}{pay:>9.4f}{bb:>+9.3f}{'':>10}{'':>13}{-(bb**2) * pay:>+12.5f}"
            f"{-(bb**2) * pay * 0.7784:>+12.5f}"
        )
    print("\n  NOT: her blok KENDI b'sini kullanirsa bu bir TAVAN. Capraz blok sinamasi:")
    print("  bir blogun b'sini digerine uygula.")
    B = {}
    for ad in ("yaz25", "guz25", "kis26"):
        T = hepsi[ad]
        s = T[T["sifir_asof90"] > 0.75]
        B[ad] = (float((s["b"] * s["n"]).sum() / s["n"].sum()), float(s["n"].sum() / T["n"].sum()))
    print(
        f"\n  {'delta kaynagi':<14}"
        + "".join(f"{'-> ' + ad:>14}" for ad in ("yaz25", "guz25", "kis26"))
    )
    for kay in ("yaz25", "guz25", "kis26"):
        dlt = B[kay][0]
        satir = f"  {kay:<14}"
        for hed in ("yaz25", "guz25", "kis26"):
            bb, pay = B[hed]
            satir += f"{pay * (dlt**2 - 2 * dlt * bb):>+14.5f}"
        print(satir)
    print("\n  (negatif = KAZANC, sicak satirlarda dMSE)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
