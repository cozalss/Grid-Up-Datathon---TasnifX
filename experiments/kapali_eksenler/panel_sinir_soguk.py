"""PANEL SINIR -- SOGUK TARAF. Testte 2026-05-11'de 1.326 SOGUK trafo
panele giriyor; giris gunu sistematik olarak farkli mi?

Soguk harmani ``experiments/soguk_kaldirac/ortak.py``'den gelir; oradaki
``mse`` KIRPMASIZ oldugu icin burada URETIM KIRPMASI ile yeniden tanimlandi
(np.clip(np.expm1(.),0,None) <=> max(r+lgc, 0)).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "experiments" / "soguk_kaldirac"))

from ortak import BLOKLAR, SOGUK_PAY, taban_r, tum_bloklar  # noqa: E402

GUN = pd.Timedelta(days=1)
pd.set_option("display.width", 240)


def mse_k(b, r: np.ndarray) -> float:
    """URETIM KIRPMASI ile MSE."""
    e = b.lgy - np.maximum(r + b.lgc, 0.0)
    return float((e * e).mean())


def kdelta(b, r: np.ndarray) -> float:
    en, enm = 0.0, mse_k(b, r)
    adim = 0.08
    for _ in range(5):
        for d in np.arange(en - 4 * adim, en + 4.001 * adim, adim):
            m = mse_k(b, r + float(d))
            if m < enm:
                en, enm = float(d), m
        adim /= 4.0
    return en


def main() -> int:
    bloklar = tum_bloklar()
    print("=" * 100)
    print("SOGUK: GIRIS/CIKIS GUNU YANLILIGI (blok ici varlik deseninden)")
    print("=" * 100)
    maskeler = {}
    for ad in BLOKLAR:
        b = bloklar[ad]
        d = pd.DataFrame({"tanim": b.tanim, "tarih": pd.to_datetime(b.tarih)})
        d["_i"] = np.arange(len(d))
        d = d.sort_values(["tanim", "tarih"], kind="mergesort")
        onc = d.groupby("tanim", observed=True)["tarih"].shift(1)
        son = d.groupby("tanim", observed=True)["tarih"].shift(-1)
        d["giris"] = onc.isna() | ((d["tarih"] - onc) > GUN)
        d["cikis"] = son.isna() | ((son - d["tarih"]) > GUN)
        # blok kenarlarini disla
        d.loc[d["tarih"] == d["tarih"].min(), "giris"] = False
        d.loc[d["tarih"] == d["tarih"].max(), "cikis"] = False
        d = d.sort_values("_i")
        g = d["giris"].to_numpy()
        c = d["cikis"].to_numpy()
        maskeler[ad] = (g.astype("float64"), c.astype("float64"))

        r0 = taban_r(b)
        e = b.lgy - np.maximum(r0 + b.lgc, 0.0)
        print(f"\n-- {ad} (soguk n={b.n:,}, taban MSE {mse_k(b, r0):.5f}) --")
        for nm, m in (("GIRIS", g), ("CIKIS", c), ("ic", ~(g | c))):
            if m.sum() == 0:
                print(f"   {nm}: YOK")
                continue
            print(
                f"   {nm:6}: n={int(m.sum()):6,}  yanlilik {e[m].mean():+.4f}  "
                f"t={e[m].mean() / (e[m].std() / np.sqrt(m.sum())):+8.2f}  "
                f"mse {float((e[m] ** 2).mean()):.4f}  "
                f"y0 payi {float((b.y[m] <= 0).mean()):.4f}"
            )

    print("\n" + "=" * 100)
    print("BLOK BASINA OPTIMUM (d_giris, d_cikis) -- seviye-notr, kirpmali")
    print("=" * 100)
    izgara = np.arange(-1.20, 0.401, 0.10)
    for ad in BLOKLAR:
        b = bloklar[ad]
        r0 = taban_r(b)
        m0 = mse_k(b, r0 + kdelta(b, r0))
        en = None
        for dg in izgara:
            for dc in izgara:
                rr = r0 + dg * maskeler[ad][0] + dc * maskeler[ad][1]
                v = mse_k(b, rr + kdelta(b, rr))
                if en is None or v < en[2]:
                    en = (float(dg), float(dc), v)
        print(
            f"  {ad:6} d_giris {en[0]:+.2f}  d_cikis {en[1]:+.2f}  "
            f"kazanc {en[2] - m0:+.6f} soguk MSE  "
            f"({(en[2] - m0) * SOGUK_PAY:+.6f} test MSE)"
        )

    print("\n" + "=" * 100)
    print("SABIT ADAYLAR")
    print("=" * 100)
    print(
        f"{'d_giris/d_cikis':20}{'yaz25':>11}{'guz25':>11}{'kis26':>11}"
        f"{'GENEL':>11}{'testdMSE':>11}  karar"
    )
    print("-" * 88)
    for dg, dc in ((-0.30, -0.30), (-0.50, -0.50), (-0.70, -0.70), (-0.50, -0.20)):
        s = {}
        tn = td = 0.0
        for ad in BLOKLAR:
            b = bloklar[ad]
            r0 = taban_r(b)
            m0 = mse_k(b, r0 + kdelta(b, r0))
            rr = r0 + dg * maskeler[ad][0] + dc * maskeler[ad][1]
            d = mse_k(b, rr + kdelta(b, rr)) - m0
            s[ad] = d
            tn += b.n
            td += d * b.n
        gen = td / tn
        tm = gen * SOGUK_PAY
        uc = all(s[k] < 0 for k in BLOKLAR) or all(s[k] > 0 for k in BLOKLAR)
        karar = (
            "KABUL"
            if uc and tm <= -0.002
            else "red(kucuk)"
            if uc and tm < 0
            else "ters isaret"
            if tm < 0
            else "RED(zararli)"
        )
        print(
            f"{f'{dg:+.2f} / {dc:+.2f}':20}{s['yaz25']:>+11.5f}{s['guz25']:>+11.5f}"
            f"{s['kis26']:>+11.5f}{gen:>+11.5f}{tm:>+11.5f}  {karar}"
        )
    print(f"\nNOT: soguk dMSE -> test dMSE carpani {SOGUK_PAY:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
