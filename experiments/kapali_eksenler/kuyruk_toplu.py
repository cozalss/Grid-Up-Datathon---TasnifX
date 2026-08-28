"""KUYRUK/GENC ETKISI: TOPLU DOGUM mu, GECMIS UZUNLUGU mu?

dogum_dalgasi.py'nin bulgusu: bloklarin genc kohortlari KOMPOZISYON
bakimindan cok farkli.

    7-30g bandinda TOPLU dogumlu pay:  guz25 %0,0 | kis26 %69,6 | TEST %0,0
    0-6g  bandinda:                    guz25 %89,4 | kis26 %72,3 | TEST %92,4
    31-90g bandinda:                   guz25 %38,5 | kis26 %43,7 | TEST %24,7

kuyruk_adaylar.py'de pencereyi genisletmek kis26'da BUYUK kazandirip
guz25'te kaybettirdi. Bu betik hangi eksenin gercek oldugunu ayirir:
yanlilik TOPLU DOGUM'a mi bagli, yoksa GECMIS UZUNLUGU'na mi?

Hukum test kompozisyonu ile verilir.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "experiments" / "sicak_kaldirac"))

from ortak import (  # noqa: E402
    BLOKLAR,
    KUYRUK_DELTA,
    SICAK_PAY,
    bloklari_kur,
    kuresel_delta,
    mse,
    taban_r,
)

CIK = Path(__file__).resolve().parent
TOPLU_ESIK = 20
pd.set_option("display.width", 240)


def toplu_gunler() -> set:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
        encoding="utf-8",
    )
    ilk = tr.groupby("tanim")["tarih"].min()
    say = ilk.value_counts()
    return set(say[say >= TOPLU_ESIK].index), ilk


def main() -> int:
    tg, ilk = toplu_gunler()
    bloklar = bloklari_kur()
    taban = {k: taban_r(bloklar[k]) for k in BLOKLAR}

    kenar = [-1e9, 6, 30, 90, 180, 1e9]
    et = ["<=6g", "7-30g", "31-90g", "91-180g", ">180g"]

    print("=" * 106)
    print("1) YANLILIK = f(gecmis bandi, dogum turu)   [uretim tabani, kirpmali]")
    print("   yanlilik>0 -> MODEL DUSUK TAHMIN EDIYOR (yukari duzeltme gerekir)")
    print("=" * 106)
    kayit = []
    for ad in BLOKLAR:
        b = bloklar[ad]
        r0 = taban[ad]
        e = b.lgy - np.maximum(r0 + b.lgc, 0.0)
        dg = b.cerceve["tanim"].map(ilk)
        toplu = dg.isin(tg).to_numpy()
        band = pd.cut(b.cerceve["gecmis_gun"].to_numpy(), kenar, labels=et)
        d = pd.DataFrame(
            {
                "band": band,
                "dogum": np.where(toplu, "TOPLU", "tekil"),
                "e": e,
                "e2": e * e,
                "t": b.cerceve["tanim"].to_numpy(),
            }
        )
        g = d.groupby(["band", "dogum"], observed=True).agg(
            n=("e", "size"),
            trafo=("t", "nunique"),
            yanlilik=("e", "mean"),
            std=("e", "std"),
            mse=("e2", "mean"),
        )
        g["t_ist"] = g["yanlilik"] / (g["std"] / np.sqrt(g["n"]))
        print(f"\n-- {ad} --")
        print(g.round(4).to_string())
        for (bd, dt), row in g.iterrows():
            kayit.append(
                {
                    "blok": ad,
                    "band": str(bd),
                    "dogum": dt,
                    "n": int(row["n"]),
                    "yanlilik": float(row["yanlilik"]),
                    "t": float(row["t_ist"]),
                }
            )

    print("\n" + "=" * 106)
    print("2) BANT ICINDE TOPLU - TEKIL FARKI (ayni bantta, kompozisyondan arindirilmis)")
    print("=" * 106)
    df = pd.DataFrame(kayit)
    p = df.pivot_table(index=["band"], columns=["blok", "dogum"], values="yanlilik")
    print(p.round(4).to_string())

    print("\n" + "=" * 106)
    print("3) ADAYLAR -- TEST KOMPOZISYONUNA gore hedeflenmis")
    print("=" * 106)

    def maske(b, band_alt, band_ust, dogum: str | None):
        g = b.cerceve["gecmis_gun"].to_numpy()
        m = (g >= band_alt) & (g <= band_ust)
        if dogum is not None:
            dg = b.cerceve["tanim"].map(ilk)
            t = dg.isin(tg).to_numpy()
            m = m & (t if dogum == "TOPLU" else ~t)
        return m.astype("float64")

    def yap(band_alt, band_ust, dogum, d, kuyrugu_kaldir=False):
        def f(b, r0):
            r = r0 - (
                KUYRUK_DELTA * b.cerceve["kuyruk"].to_numpy(dtype="float64")
                if kuyrugu_kaldir
                else 0.0
            )
            return r + d * maske(b, band_alt, band_ust, dogum)

        return f

    adaylar = [
        # kuyruk bandini TOPLU/tekil ayirarak yeniden seviyele (uretim 0,16640 kaldirilir)
        ("T1  <=6g TOPLU +0,17 (tekil 0)", yap(-1e9, 6, "TOPLU", 0.17, True)),
        ("T2  <=6g TOPLU +0,30 (tekil 0)", yap(-1e9, 6, "TOPLU", 0.30, True)),
        ("T3  <=6g tekil +0,17 (TOPLU 0)", yap(-1e9, 6, "tekil", 0.17, True)),
        # genisletme YALNIZ tekil dogumlulara (test 7-30g %100 tekil)
        ("T4  7-30g tekil +0,17", yap(7, 30, "tekil", 0.17)),
        ("T5  7-30g tekil -0,17", yap(7, 30, "tekil", -0.17)),
        ("T6  7-90g tekil +0,17", yap(7, 90, "tekil", 0.17)),
        ("T7  7-90g tekil -0,17", yap(7, 90, "tekil", -0.17)),
        # genisletme YALNIZ toplu dogumlulara (kis26'nin kazandigi yer mi?)
        ("T8  7-90g TOPLU +0,17", yap(7, 90, "TOPLU", 0.17)),
        ("T9  7-90g TOPLU +0,30", yap(7, 90, "TOPLU", 0.30)),
        ("T10 7-30g TOPLU +0,30", yap(7, 30, "TOPLU", 0.30)),
    ]

    satirlar = []
    for ad_a, fn in adaylar:
        s: dict = {"aday": ad_a}
        tn = td = 0.0
        for k in BLOKLAR:
            b = bloklar[k]
            r0 = taban[k]
            r1 = fn(b, r0)
            r1 = r1 + kuresel_delta(b, r1)
            dd = mse(b, r1) - mse(b, r0)
            s[k] = dd
            tn += b.n
            td += dd * b.n
        s["GENEL"] = td / tn
        s["testMSE"] = s["GENEL"] * SICAK_PAY
        s["uc_ayni"] = all(s[k] < 0 for k in BLOKLAR) or all(s[k] > 0 for k in BLOKLAR)
        s["iki_ayni"] = (s["guz25"] < 0) == (s["kis26"] < 0)
        satirlar.append(s)

    print(
        f"\n{'aday':34}{'yaz25':>11}{'guz25':>11}{'kis26':>11}{'GENEL':>11}{'testdMSE':>11}  karar"
    )
    print("-" * 102)
    for s in satirlar:
        if s["testMSE"] >= 0:
            k = "RED(zararli)"
        elif s["uc_ayni"] and s["testMSE"] <= -0.002:
            k = "KABUL"
        elif s["uc_ayni"]:
            k = "red(kucuk)"
        elif s["iki_ayni"]:
            k = "2/2 blok ayni -> ADAY"
        else:
            k = "ters isaret -> PROB"
        print(
            f"{s['aday'][:34]:34}{s['yaz25']:>+11.5f}{s['guz25']:>+11.5f}"
            f"{s['kis26']:>+11.5f}{s['GENEL']:>+11.5f}{s['testMSE']:>+11.5f}  {k}"
        )

    with (CIK / "kuyruk_toplu.jsonl").open("w", encoding="utf-8") as f:
        for s in satirlar:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
        for r in kayit:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
