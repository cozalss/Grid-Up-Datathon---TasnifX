"""PROB TASARIMI -- adim 4: HANGI DESEN? (kaynak blok x hedef blok tablosu)

teshis_seviye.py sicak seviye deseninin kis26'da TERS dondugunu gosterdi
(yaz25|kis26 rho = -0.787). Yani "hangi blogun deseni" sorusu onemli.

Test penceresi 2026-04..07 -> MEVSIM IKIZI yaz25. Bu betik her yon icin:
  * blok ciftleri arasi ISARETLI agirlikli korelasyon
  * kaynak blok deseni -> hedef blokta gerceklesen prob kazanci
  * "hile" tavani (ayni bloktan ogrenilen TAM 10-parametreli ofset)
tablosunu basar. Prob kazanci = rho^2 * tavan oldugu icin ISARET onemsiz;
oldurucu olan tek sey |rho| ~ 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
BURA = Path(__file__).resolve().parent
sys.path.insert(0, str(BURA))

from tavan import BLOKLAR, SIC, SICAK_PAY, SOGUK_PAY, _soguk_modul, desil, kova_kva  # noqa: E402


def analiz(bl, tab, ad, grup_fn, artik_fn, pay, rejim, cikti):
    desen, boyut, artik, gd = {}, {}, {}, {}
    for k in BLOKLAR:
        b = bl[k]
        r0 = tab[k]
        e = artik_fn(b, r0)
        e = e - e.mean()
        g = pd.Series(np.asarray(grup_fn(b, r0)).astype(str))
        artik[k], gd[k] = e, g
        s = pd.Series(e).groupby(g)
        desen[k], boyut[k] = s.mean(), s.size()

    # isaretli agirlikli korelasyon (ortak gruplar, hedef-notr agirlik)
    rho = {}
    for i, a in enumerate(BLOKLAR):
        for b2 in BLOKLAR[i + 1 :]:
            ort = desen[a].index.intersection(desen[b2].index)
            if len(ort) < 3:
                rho[f"{a}|{b2}"] = float("nan")
                continue
            w = (boyut[a].reindex(ort).fillna(0) + boyut[b2].reindex(ort).fillna(0)).to_numpy()
            w = w / w.sum()
            x = desen[a].reindex(ort).to_numpy()
            y = desen[b2].reindex(ort).to_numpy()
            xm, ym = w @ x, w @ y
            vx, vy = w @ (x - xm) ** 2, w @ (y - ym) ** 2
            rho[f"{a}|{b2}"] = (
                float((w @ ((x - xm) * (y - ym))) / np.sqrt(vx * vy))
                if vx > 0 and vy > 0
                else float("nan")
            )

    # tavan (hile): ayni bloktan tam ofset
    tavan = {}
    for k in BLOKLAR:
        d = gd[k].map(desen[k]).fillna(0.0).to_numpy()
        yeni = artik[k] - d
        tavan[k] = float((yeni**2).mean() - (artik[k] ** 2).mean())
    n_top = sum(bl[k].n for k in BLOKLAR)
    tavan_ort = sum(tavan[k] * bl[k].n for k in BLOKLAR) / n_top

    # kaynak -> hedef kazanc matrisi
    mat = {}
    for kay in BLOKLAR:
        for hed in BLOKLAR:
            if kay == hed:
                continue
            d = gd[hed].map(desen[kay]).fillna(0.0).to_numpy()
            d = d - d.mean()
            q = float(d @ d) / len(d)
            if q <= 0:
                mat[f"{kay}->{hed}"] = {"kazanc": 0.0, "kappa": 0.0}
                continue
            L = float(d @ artik[hed]) / len(d)
            mat[f"{kay}->{hed}"] = {"kazanc": -(L * L) / q, "kappa": L / q}

    cikti.append(
        {
            "rejim": rejim,
            "yon": ad,
            "grup": int(len(desen[BLOKLAR[0]])),
            "tavan_rejim": tavan_ort,
            "tavan_toplam": tavan_ort * pay,
            "tavan_blok": tavan,
            "rho": rho,
            "kaynak_hedef": {k: v["kazanc"] * pay for k, v in mat.items()},
            "kappa": {k: v["kappa"] for k, v in mat.items()},
            "desen": {k: {str(i): float(x) for i, x in desen[k].items()} for k in BLOKLAR},
        }
    )
    rs = "  ".join(f"{k}={v:+.3f}" for k, v in rho.items())
    print(
        f"\n{rejim.upper()} {ad}  G={len(desen[BLOKLAR[0]])}  tavan_toplam={tavan_ort * pay:+.5f}"
    )
    print(f"   isaretli rho: {rs}")
    kh = "  ".join(f"{k}={v:+.5f}" for k, v in sorted(cikti[-1]["kaynak_hedef"].items()))
    print(f"   kaynak->hedef kazanc (toplam MSE): {kh}")


def main() -> None:
    cikti: list[dict] = []

    bl = SIC.bloklari_kur()
    tab = {k: SIC.taban_r(bl[k]) for k in BLOKLAR}
    sic_art = lambda b, r: b.lgy - np.maximum(r + b.lgc, 0.0)  # noqa: E731
    for ad, fn in {
        "ilce": lambda b, r: b.cerceve["ilce"].to_numpy(),
        "seviye_desili10": lambda b, r: desil(np.maximum(r + b.lgc, 0.0), 10),
        "kva_kovasi": lambda b, r: b.cerceve["kova"].to_numpy(),
        "ilce_x_kova": lambda b, r: (
            b.cerceve["ilce"].astype(str) + "|" + b.cerceve["kova"].astype(str)
        ).to_numpy(),
    }.items():
        analiz(bl, tab, ad, fn, sic_art, SICAK_PAY, "sicak", cikti)
    del bl, tab

    SOG = _soguk_modul()
    sb = SOG.tum_bloklar()
    stab = {k: SOG.taban_r(sb[k]) for k in BLOKLAR}
    sog_art = lambda b, r: b.lgy - (r + b.lgc)  # noqa: E731
    for ad, fn in {
        "ilce": lambda b, r: b.ilce,
        "seviye_desili10": lambda b, r: desil(r + b.lgc, 10),
        "kva_kovasi": lambda b, r: kova_kva(b.guc),
        "ilce_x_kova": lambda b, r: np.char.add(
            np.char.add(b.ilce.astype(str), "|"), kova_kva(b.guc).astype(str)
        ),
    }.items():
        analiz(sb, stab, ad, fn, sog_art, SOGUK_PAY, "soguk", cikti)

    (BURA / "karar.json").write_text(
        json.dumps(cikti, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nyazildi: karar.json")


if __name__ == "__main__":
    main()
