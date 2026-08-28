"""B+C: bootstrap kapisi ile aday tablosu (onarilmis olcut)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BURA = Path(__file__).resolve().parent
sys.path.insert(0, str(BURA))
import tezgah as tz, kapi as kp  # noqa

BETA, C, DELTA = 0.60, 1.3301, 0.1046
P_SOGUK = 0.22159


def var(setup, blok, ayar):
    if setup == "kirli" and blok == "kis26":
        setup = "onarilmis"
    return (tz.ONBELLEK / f"{setup}_{blok}_{ayar}.npy").exists()


def yukle(setup, blok, ayar):
    if setup == "kirli" and blok == "kis26":
        setup = "onarilmis"
    return np.load(tz.ONBELLEK / f"{setup}_{blok}_{ayar}.npy").mean(axis=0)


def harman(setup, blok, w):
    pay = sum(w.values())
    return sum(yukle(setup, blok, a) * wi for a, wi in w.items() if wi) / pay


ADAYLAR = {
    "URETIM cat d7": {"cat_d7": 1.0},
    "cat d5": {"cat_d5": 1.0},
    "cat d6": {"cat_d6": 1.0},
    "cat d8": {"cat_d8": 1.0},
    "xgb": {"xgb": 1.0},
    "lgbm": {"lgbm": 1.0},
    "uclu 1/1/1": {"cat_d7": 1.0, "xgb": 1.0, "lgbm": 1.0},
    "uclu 2/1/1": {"cat_d7": 2.0, "xgb": 1.0, "lgbm": 1.0},
    "uclu 3/1/1": {"cat_d7": 3.0, "xgb": 1.0, "lgbm": 1.0},
    "cat/lgbm 1/1": {"cat_d7": 1.0, "lgbm": 1.0},
    "cat/lgbm 2/1": {"cat_d7": 2.0, "lgbm": 1.0},
    "cat/xgb 1/1": {"cat_d7": 1.0, "xgb": 1.0},
    "cat/xgb 2/1": {"cat_d7": 2.0, "xgb": 1.0},
    "cat d5+d7 1/1": {"cat_d5": 1.0, "cat_d7": 1.0},
    "cat d6+d7 1/1": {"cat_d6": 1.0, "cat_d7": 1.0},
    "cat d7 rs1": {"cat_d7_rs1": 1.0},
    "cat d7 rs4": {"cat_d7_rs4": 1.0},
    "cat d7 lr.03": {"cat_d7_lr03": 1.0},
    "cat d7 l2=1": {"cat_d7_l2_1": 1.0},
    "cat d6 rs4": {"cat_d6_rs4": 1.0},
    "cat d5 rs4": {"cat_d5_rs4": 1.0},
    "cat d7 lr.03+rs4": {"cat_d7_lr03_rs4": 1.0},
    "cat d7 -kimlik": {"cat_d7_nokimlik": 1.0},
    "cat d7 -tnum": {"cat_d7_notanimnum": 1.0},
    "cat d7 ekkoken": {"cat_d7_ekkoken": 1.0},
}


def main(setup="onarilmis"):
    veri = {b: (tz.meta(b),) for b in tz.BLOKLAR}
    for b in tz.BLOKLAR:
        m = veri[b][0]
        veri[b] = (m, m["y"].to_numpy(float), np.log1p(m["guc"].to_numpy(float)))

    kullan = {
        k: v for k, v in ADAYLAR.items() if all(var(setup, b, a) for b in tz.BLOKLAR for a in v)
    }
    e = {}
    for ad, w in kullan.items():
        for b in tz.BLOKLAR:
            m, y, lgc = veri[b]
            lg = kp.zincir(harman(setup, b, w), lgc, m, BETA, C, DELTA)
            e[(ad, b)] = kp.kare_hatalar(y, lg)

    print("=" * 122)
    print(
        f"ADAY TABLOSU -- setup={setup} | son islem beta={BETA} c={C} delta={DELTA} | kirpma DAHIL"
    )
    print("=" * 122)
    hdr = f"{'aday':17}"
    for b in tz.BLOKLAR:
        hdr += f"{b + ' MSE':>12}{'dMSE':>10}"
    print(hdr + f"{'isaret':>9}{'kis26 CI':>22}{'kaz.trafo':>11}{'KAPI':>7}")
    taban = {b: e[("URETIM cat d7", b)] for b in tz.BLOKLAR}
    kayit = {}
    for ad in kullan:
        satir, dler = f"{ad:17}", []
        for b in tz.BLOKLAR:
            m = e[(ad, b)].mean()
            d = m - taban[b].mean()
            dler.append(d)
            satir += f"{m:>12.5f}{d:>+10.5f}"
        ayni = "AYNI" if (all(v < 0 for v in dler) or all(v > 0 for v in dler)) else "-"
        if ad == "URETIM cat d7":
            print(satir + f"{'(taban)':>9}")
            continue
        r = kp.bootstrap(taban["kis26"], e[(ad, "kis26")], veri["kis26"][0])
        ci = f"[{r['ci_lo']:+.5f},{r['ci_hi']:+.5f}]"
        print(
            satir + f"{ayni:>9}{ci:>22}{100 * r['kazanan_trafo']:>10.1f}%"
            f"{('GECTI' if r['gecti'] else '-'):>7}"
        )
        kayit[ad] = {"dmse": dler, "isaret": ayni, **r}
    (BURA / f"aday_{setup}.json").write_text(json.dumps(kayit, indent=2), encoding="utf-8")

    # uc-blok BIRLESIK bootstrap (onarilmis olcutte uc blok da temiz)
    if setup == "onarilmis":
        print("\n" + "=" * 122)
        print(
            "UC BLOK BIRLESIK bootstrap (trafolar bloklar arasi ayrik degil -> blok icinde kumeleme,"
        )
        print("blok agirligi = TEST soguk kohortuna esit pay)")
        print("=" * 122)
        print(f"{'aday':17}{'dMSE':>11}{'CI':>24}{'kaz.trafo':>11}{'KAPI':>7}")
        metalar = [veri[b][0] for b in tz.BLOKLAR]
        for ad in kullan:
            if ad == "URETIM cat d7":
                continue
            et = np.concatenate([taban[b] for b in tz.BLOKLAR])
            ea = np.concatenate([e[(ad, b)] for b in tz.BLOKLAR])
            mm = pd.concat(
                [
                    m.assign(tanim=b + "_" + m["tanim"].astype(str))
                    for b, m in zip(tz.BLOKLAR, metalar)
                ],
                ignore_index=True,
            )
            r = kp.bootstrap(et, ea, mm)
            ci = f"[{r['ci_lo']:+.5f},{r['ci_hi']:+.5f}]"
            print(
                f"{ad:17}{r['dmse']:>+11.5f}{ci:>24}{100 * r['kazanan_trafo']:>10.1f}%"
                f"{('GECTI' if r['gecti'] else '-'):>7}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
