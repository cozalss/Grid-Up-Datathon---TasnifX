# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""SICAK ARTIK -- ADIM C: ufka gore buyume, hava x trafo-egimi etkilesimi,
trafo nitelikleriyle iliski."""

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


def guvenli(x):
    x = np.asarray(x, dtype="float64")
    m = np.nanmedian(x)
    return np.nan_to_num(x, nan=0.0 if not np.isfinite(m) else m)


def main() -> int:
    egitim, test = d.cerceveleri_kur()
    for b in tm.BLOKLAR:
        v = sa.blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        trafo = dg["tanim"].to_numpy()
        gun = pd.to_datetime(dg["tarih"])
        gund = gun.values.astype("datetime64[D]")
        ga = sa.iki_yonlu_arindir(v["g"], trafo, gund)
        ra = sa.iki_yonlu_arindir(v["r"], trafo, gund)
        e_ham = v["g"] - v["r"]
        print(f"\n=== {b.ad}")

        uf = pd.cut(dg["ufuk_gun"], [0, 15, 31, 61, 91, 200])
        t = pd.DataFrame({"uf": uf, "ga2": ga**2, "e2": e_ham**2, "kg": ga, "kr": ra}).groupby(
            "uf", observed=True
        )
        ozet = t.agg(n=("ga2", "size"), artik_var=("ga2", "mean"), ham_mse=("e2", "mean"))
        ozet["kor(g,r)"] = t.apply(
            lambda x: np.corrcoef(x["kg"], x["kr"])[0, 1], include_groups=False
        )
        print("  UFUK KIRILIMI")
        print(ozet.to_string(float_format=lambda x: f"{x:9.4f}"))

        print("  HAVA x TRAFO-EGIMI (iki yonlu arindirilmis, artik ile)")
        for hk, ek in (("sicaklik_ort", "t_egim_sicaklik_ort"), ("cdd22", "t_egim_cdd22")):
            if dg[ek].notna().mean() < 0.05:
                print(
                    f"    {ek:26} dogrulamada %{dg[ek].notna().mean() * 100:.1f} dolu -- OLCULEMEZ"
                )
                continue
            hava = guvenli(dg[hk])
            eg = guvenli(dg[ek])
            hs = hava - pd.Series(hava).groupby(pd.Series(trafo)).transform("mean").to_numpy()
            for ad, z in (
                (f"{ek} * {hk}", eg * hava),
                (f"{ek} * {hk}_sapma", eg * hs),
                (f"{hk}_sapma (duz)", hs),
            ):
                za = sa.iki_yonlu_arindir(guvenli(z), trafo, gund)
                if za.std() < 1e-9:
                    continue
                c = float(np.corrcoef(za, ga)[0, 1])
                cr = float(np.corrcoef(za, ra)[0, 1])
                print(
                    f"    {ad:34} kor(artik)={c:+.4f}  R2={c * c * 100:5.2f}%   kor(model)={cr:+.4f}"
                )

        tf = (
            pd.DataFrame(
                {
                    "trafo": trafo,
                    "ga2": ga**2,
                    "e2": e_ham**2,
                    "guc": dg["guc"].to_numpy(),
                    "yuk": dg["t_yuk_faktoru"].to_numpy(),
                    "std": dg["t_log_std"].to_numpy(),
                    "gun_s": dg["t_gun_sayisi"].to_numpy(),
                    "sifir": dg["t_sifir_orani"].to_numpy(),
                    "trend": dg["t_trend"].to_numpy(),
                    "ort": dg["t_log_ort"].to_numpy(),
                    "yayilma": dg["t_yayilma"].to_numpy(),
                }
            )
            .groupby("trafo")
            .mean()
        )
        tf["artik_std"] = np.sqrt(tf["ga2"])
        print("  TRAFO DUZEYI (spearman, artik_std ile)   n=%d" % len(tf))
        cs = []
        for k in ("guc", "yuk", "std", "gun_s", "sifir", "trend", "ort", "yayilma"):
            s = tf[[k, "artik_std"]].dropna()
            cs.append(f"{k}={s[k].corr(s['artik_std'], method='spearman'):+.3f}")
        print("    " + "  ".join(cs))
        # MSE payi: artik_std ust yuzdelik dilimler
        tf["e2n"] = tf["e2"] * pd.Series(trafo).value_counts().reindex(tf.index).to_numpy()
        srt = tf.sort_values("artik_std", ascending=False)
        pay = srt["e2n"].cumsum() / srt["e2n"].sum()
        for q in (0.01, 0.05, 0.10, 0.25):
            k = max(1, int(len(srt) * q))
            print(
                f"    en artikli %{q * 100:>4.0f} trafo ({k:5,}) -> ham HATA^2'nin %{pay.iloc[k - 1] * 100:.1f}'i"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
