"""A3 + B + C: isaret tutarliligi sinavi, bootstrap kapisi, aday tablosu."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BURA = Path(__file__).resolve().parent
sys.path.insert(0, str(BURA))
import tezgah as tz  # noqa
import kapi as kp  # noqa

BLOKLAR = tz.BLOKLAR
ONB = tz.ONBELLEK

# uretim soguk son islem sabitleri (v83 zinciri)
BETA, C, DELTA = 0.60, 1.3301, 0.1046


def var(setup, blok, ayar) -> bool:
    if setup == "kirli" and blok == "kis26":
        setup = "onarilmis"  # onarim kis26'da hicbir sey degistirmiyor
    return (ONB / f"{setup}_{blok}_{ayar}.npy").exists()


def yukle(setup, blok, ayar) -> np.ndarray:
    if setup == "kirli" and blok == "kis26":
        setup = "onarilmis"
    return np.load(ONB / f"{setup}_{blok}_{ayar}.npy")


def harman(setup, blok, w: dict[str, float]) -> np.ndarray:
    """Tohum-ortalamali LOG harman. w: ayar adi -> agirlik."""
    pay = sum(w.values())
    out = None
    for ayar, wi in w.items():
        if wi == 0:
            continue
        a = yukle(setup, blok, ayar).mean(axis=0)
        out = a * wi if out is None else out + a * wi
    return out / pay


def blok_verisi(blok):
    m = tz.meta(blok)
    return m, m["y"].to_numpy(dtype="float64"), np.log1p(m["guc"].to_numpy(dtype="float64"))


def skorla(lg, y, lgc, meta, son_islem=True):
    if son_islem:
        lg = kp.zincir(lg, lgc, meta, BETA, C, DELTA)
    return kp.kare_hatalar(y, lg)


ADAYLAR = {
    "URETIM cat d7": {"cat_d7": 1.0},
    "cat d5": {"cat_d5": 1.0},
    "cat d6": {"cat_d6": 1.0},
    "cat d8": {"cat_d8": 1.0},
    "xgb": {"xgb": 1.0},
    "lgbm": {"lgbm": 1.0},
    "uclu 1/1/1": {"cat_d7": 1.0, "xgb": 1.0, "lgbm": 1.0},
    "uclu 3/1/1": {"cat_d7": 3.0, "xgb": 1.0, "lgbm": 1.0},
    "uclu 2/1/1": {"cat_d7": 2.0, "xgb": 1.0, "lgbm": 1.0},
    "cat/lgbm 1/1": {"cat_d7": 1.0, "lgbm": 1.0},
    "cat/lgbm 2/1": {"cat_d7": 2.0, "lgbm": 1.0},
    "cat/xgb 1/1": {"cat_d7": 1.0, "xgb": 1.0},
    "d5 uclu 1/1/1": {"cat_d5": 1.0, "xgb": 1.0, "lgbm": 1.0},
    "cat d5/d7 1/1": {"cat_d5": 1.0, "cat_d7": 1.0},
}
EK = {
    "cat d7 rs1": {"cat_d7_rs1": 1.0},
    "cat d7 rs4": {"cat_d7_rs4": 1.0},
    "cat d7 lr.03": {"cat_d7_lr03": 1.0},
    "cat d7 l2=1": {"cat_d7_l2_1": 1.0},
    "cat d6 rs4": {"cat_d6_rs4": 1.0},
    "cat d7 -kimlik": {"cat_d7_nokimlik": 1.0},
    "cat d7 -tnum": {"cat_d7_notanimnum": 1.0},
}


def main():
    hepsi = dict(ADAYLAR)
    for k, v in EK.items():
        if all(var("onarilmis", b, a) for b in BLOKLAR for a in v):
            hepsi[k] = v

    veri = {b: blok_verisi(b) for b in BLOKLAR}

    print("=" * 118)
    print("A3  ISARET TUTARLILIGI SINAVI -- ayni aday, uc blok, KIRLI vs ONARILMIS")
    print("     (soguk MSE, uretim son islemi beta=0.60 c=1.3301 delta=0.1046, kirpma DAHIL)")
    print("=" * 118)
    for setup in ("kirli", "onarilmis"):
        kullan = {
            k: v for k, v in hepsi.items() if all(var(setup, b, a) for b in BLOKLAR for a in v)
        }
        if not kullan:
            continue
        print(f"\n--- {setup.upper()} ---")
        hdr = f"{'aday':18}" + "".join(f"{b + ' MSE':>13}{'dMSE':>10}" for b in BLOKLAR)
        print(hdr + f"{'uc-blok isaret':>16}")
        taban_mse, taban_e = {}, {}
        for ad, w in kullan.items():
            satir, dler = f"{ad:18}", []
            for b in BLOKLAR:
                m, y, lgc = veri[b]
                e = skorla(harman(setup, b, w), y, lgc, m)
                if ad == "URETIM cat d7":
                    taban_mse[b] = e.mean()
                    taban_e[(setup, b)] = e
                d = float(e.mean() - taban_mse[b])
                dler.append(d)
                satir += f"{e.mean():>13.5f}{d:>+10.5f}"
            ayni = "AYNI" if (all(v < 0 for v in dler) or all(v > 0 for v in dler)) else "-"
            if ad == "URETIM cat d7":
                ayni = "(taban)"
            print(satir + f"{ayni:>16}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
