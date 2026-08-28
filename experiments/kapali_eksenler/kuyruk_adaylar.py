"""KUYRUK REJIMI ADAYLARI -- uc blokta, uretim olcutuyle (kirpmali).

TABAN = v83 uretim sicak zinciri (gun ekseni c=1,3301 + KUYRUK +0,16640),
her blokta kuresel seviye yeniden kalibre edilmis. Yani bir aday ancak
YAPI ekleyerek kazanabilir; kuresel seviye LB'de cozulmus sayilir.

Her aday da kendi kuresel seviyesine kalibre edilir -- boylece "kuyruk
satirlarini yukari it" ile "her seyi yukari it" karisimaz.

Kullanim:  uv run python experiments/kapali_eksenler/kuyruk_adaylar.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

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


def _k(b) -> np.ndarray:
    return b.cerceve["kuyruk"].to_numpy(dtype="float64")


def _g(b) -> np.ndarray:
    return b.cerceve["gecmis_gun"].to_numpy(dtype="float64")


def sabit(yeni: float):
    """Kuyruk sabitini ``yeni`` ile DEGISTIR (uretimdeki 0,16640 yerine)."""

    def f(b, r0):
        return r0 + (yeni - KUYRUK_DELTA) * _k(b)

    return f


def us_sonum(d: float, tau: float):
    """Sert <=6g kesiminin yerine SUREKLI sonum: delta * exp(-gecmis/tau).

    Gerekce: doz-tepki kis26'da 6 gunde bitmiyor (7-30g +0,161, 31-90g
    +0,170). Sert kesim etkinin coguna dokunmuyor.
    """

    def f(b, r0):
        return r0 - KUYRUK_DELTA * _k(b) + d * np.exp(-_g(b) / tau)

    return f


def genis_pencere(gun: float, d: float):
    """Kuyruk penceresini ``gun``e genislet, sabit ``d``."""

    def f(b, r0):
        return r0 - KUYRUK_DELTA * _k(b) + d * (_g(b) <= gun).astype("float64")

    return f


def merdiven(d1: float, d2: float, d3: float):
    """Uc basamak: <=6g / 7-90g / 91-180g. Genlikler DISARIDAN verilir."""

    def f(b, r0):
        g = _g(b)
        v = np.where(g <= 6, d1, np.where(g <= 90, d2, np.where(g <= 180, d3, 0.0)))
        return r0 - KUYRUK_DELTA * _k(b) + v

    return f


def buzme(s: float):
    """Kuyruk satirlarini kendi ortalamalarina dogru BUZ (varyans dusur).

    Kuyruk satirlarinin artik std'si 0,82-0,86, digerleri 0,71-0,73.
    Model orada daha gurultulu; buzme kareli kayipta kazandirabilir.
    """

    def f(b, r0):
        k = _k(b).astype(bool)
        if not k.any():
            return r0
        r = r0.copy()
        m = r[k].mean()
        r[k] = m + s * (r[k] - m)
        return r

    return f


def sifir_egilimi(d: float):
    """Kuyrukta y=0 payi yuksek (kis26 %8,3 vs %4,0). En DUSUK tahminli
    kuyruk satirlarini asagi cek -- kirpmayla birlikte sifir cezasini keser.
    """

    def f(b, r0):
        k = _k(b).astype(bool)
        if not k.any():
            return r0
        r = r0.copy()
        alt = k & (r0 + b.lgc < np.quantile((r0 + b.lgc)[k], 0.25))
        r[alt] = r[alt] + d
        return r

    return f


def olc(bloklar, aday, ad: str, taban: dict) -> dict:
    s: dict = {"aday": ad}
    tn = td = 0.0
    for k in BLOKLAR:
        b = bloklar[k]
        r0 = taban[k]
        r1 = aday(b, r0)
        r1 = r1 + kuresel_delta(b, r1)  # SEVIYE-NOTR
        d = mse(b, r1) - mse(b, r0)
        s[k] = d
        tn += b.n
        td += d * b.n
    s["GENEL"] = td / tn
    s["testMSE"] = s["GENEL"] * SICAK_PAY
    s["uc_blok_ayni"] = all(s[k] < 0 for k in BLOKLAR) or all(s[k] > 0 for k in BLOKLAR)
    s["iki_blok_ayni"] = (s["guz25"] < 0) == (s["kis26"] < 0)
    return s


def main() -> int:
    bloklar = bloklari_kur()
    taban = {k: taban_r(bloklar[k]) for k in BLOKLAR}  # URETIM (kuyruk 0,16640)
    for k in BLOKLAR:
        print(f"taban {k}: sicak MSE {mse(bloklar[k], taban[k]):.6f}")

    adaylar = [
        ("K1  kuyruk sabiti 0,00 (KALDIR)", sabit(0.00)),
        ("K2  kuyruk sabiti 0,25", sabit(0.25)),
        ("K3  kuyruk sabiti 0,31", sabit(0.31)),
        ("K4  kuyruk sabiti 0,40", sabit(0.40)),
        ("K5  us sonum d=0,35 tau=30g", us_sonum(0.35, 30.0)),
        ("K6  us sonum d=0,30 tau=60g", us_sonum(0.30, 60.0)),
        ("K7  us sonum d=0,20 tau=90g", us_sonum(0.20, 90.0)),
        ("K8  pencere<=30g d=0,20", genis_pencere(30.0, 0.20)),
        ("K9  pencere<=90g d=0,17", genis_pencere(90.0, 0.17)),
        ("K10 pencere<=90g d=0,25", genis_pencere(90.0, 0.25)),
        ("K11 merdiven 0,29/0,17/0,00", merdiven(0.29, 0.17, 0.00)),
        ("K12 merdiven 0,29/0,17/-0,14", merdiven(0.29, 0.17, -0.14)),
        ("K13 kuyruk buzme s=0,80", buzme(0.80)),
        ("K14 kuyruk buzme s=0,60", buzme(0.60)),
        ("K15 kuyruk alt-ceyrek -0,30", sifir_egilimi(-0.30)),
        ("K16 kuyruk alt-ceyrek +0,30", sifir_egilimi(+0.30)),
    ]

    satirlar = [olc(bloklar, f, ad, taban) for ad, f in adaylar]

    print(
        f"\n{'aday':34}{'yaz25':>11}{'guz25':>11}{'kis26':>11}{'GENEL':>11}{'testdMSE':>11}  karar"
    )
    print("-" * 102)
    for s in satirlar:
        if s["testMSE"] >= 0:
            karar = "RED(zararli)"
        elif not s["uc_blok_ayni"]:
            karar = "PROB ADAYI" if s["iki_blok_ayni"] else "ters isaret"
        elif s["testMSE"] <= -0.002:
            karar = "KABUL"
        else:
            karar = "red(kucuk)"
        print(
            f"{s['aday'][:34]:34}{s['yaz25']:>+11.5f}{s['guz25']:>+11.5f}"
            f"{s['kis26']:>+11.5f}{s['GENEL']:>+11.5f}{s['testMSE']:>+11.5f}  {karar}"
        )

    print(
        "\nNOT: yaz25'te YALNIZ 4 kuyruk trafosu (447 satir) var -- o blok bu ekseni"
        "\n     GOREMEZ. Kural 9 (iki ortusmeyen kesme) guz25+kis26 ile saglaniyor."
    )
    with (CIK / "kuyruk_adaylar.jsonl").open("w", encoding="utf-8") as f:
        for s in satirlar:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
