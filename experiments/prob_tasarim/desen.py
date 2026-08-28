"""PROB TASARIMI -- adim 2: PROB SIMULASYONU + desen uretimi.

Adim 1 (tavan.py) "ofset ayni bloktan ogrenilirse" tavanini verdi. Bu tavan
IKI sebeple oldugundan buyuk gorunur:

  (a) grup ortalamalari gurultuyu de yakalar -> yanlilik = G*sigma^2/n
  (b) tavan "her grubun kendi optimum ofseti" demek; PROB TEK SKALER olcer.

Bu betik asil sayiyi olcer: **PROB SIMULASYONU**.

  1. Desen d, hedef blogun DISINDAKI iki bloktan ogrenilir (grup ofsetleri).
  2. Hedef blokta o yonun optimum olcegi cozulur:  kappa* = L/Q,
     L = <d, e>/n,  Q = <d, d>/n.
  3. Gerceklesen kazanc = -L^2/Q.

Adim 2, LB probunun yaptigi seyin BIREBIR AYNISIDIR: prob L'yi test kumesinde
tam olarak olcer, kappa* ondan cozulur. Yani buradaki sayi, tasima sorunu
OLMAYAN, dogrudan beklenen prob kazancidir.

Ciktilar:
  * ``prob_simulasyon.json`` -- yon basina blok blok gerceklesen kazanc
  * ``desenler.json``        -- test kumesinde kullanilacak agirlik desenleri
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

from tavan import (  # noqa: E402
    BLOKLAR,
    SIC,
    SICAK_PAY,
    SOGUK_PAY,
    _soguk_modul,
    desil,
    kova_kva,
    test_cercevesi,
)


def blok_ofset(e: np.ndarray, g: np.ndarray) -> tuple[pd.Series, pd.Series]:
    """Grup ortalamasi artik ve grup buyuklukleri."""
    s = pd.Series(e - e.mean())
    key = pd.Series(np.asarray(g))
    return s.groupby(key).mean(), key.value_counts()


def havuzla(
    ofsetler: dict[str, pd.Series],
    boyutlar: dict[str, pd.Series],
    haric: str | None,
    n0: float = 200.0,
) -> dict:
    """Blok(lar)dan James-Stein buzmeli havuzlanmis desen."""
    pay: dict[object, list[float]] = {}
    for k, o in ofsetler.items():
        if k == haric:
            continue
        b = boyutlar[k]
        for idx, val in o.items():
            n = float(b.get(idx, 0.0))
            if n <= 0:
                continue
            pay.setdefault(idx, [0.0, 0.0])
            pay[idx][0] += val * n
            pay[idx][1] += n
    return {i: (s / n) * (n / (n + n0)) for i, (s, n) in pay.items() if n > 0}


def prob_kazanci(e: np.ndarray, g: np.ndarray, harita: dict) -> dict:
    """Verilen desenin hedef blokta gerceklestirdigi kazanc (optimum kappa ile)."""
    d = pd.Series(np.asarray(g)).map(harita).fillna(0.0).to_numpy(dtype="float64")
    d = d - d.mean()  # kuresel seviye zaten LB ile cozulmus
    q = float(d @ d) / len(d)
    if q <= 0:
        return {"Q": 0.0, "L": 0.0, "kappa": 0.0, "kazanc": 0.0, "kapsam": 0.0}
    ec = e - e.mean()
    L = float(d @ ec) / len(d)
    return {
        "Q": q,
        "L": L,
        "kappa": L / q,
        "kazanc": -(L * L) / q,
        "kapsam": float((np.abs(d) > 1e-12).mean()),
    }


def rejim_kos(bl, taban, gruplar, pay: float, rejim: str, artik_fn):
    art, gd, boyut = {}, {}, {}
    for k in BLOKLAR:
        b = bl[k]
        art[k] = artik_fn(b, taban[k])
        for ad, fn in gruplar.items():
            gd.setdefault(ad, {})[k] = np.asarray(fn(b, taban[k]))

    satirlar, desenler = [], {}
    for ad in gruplar:
        ofs, boy = {}, {}
        for k in BLOKLAR:
            o, bz = blok_ofset(art[k], gd[ad][k])
            ofs[k], boy[k] = o, bz
        blok_kazanc, blok_kappa, blok_q = {}, {}, {}
        for k in BLOKLAR:
            harita = havuzla(ofs, boy, haric=k)
            r = prob_kazanci(art[k], gd[ad][k], harita)
            blok_kazanc[k] = r["kazanc"]
            blok_kappa[k] = r["kappa"]
            blok_q[k] = r["Q"]
        n_top = sum(bl[k].n for k in BLOKLAR)
        kaz = sum(blok_kazanc[k] * bl[k].n for k in BLOKLAR) / n_top
        # tum bloklardan havuzlanmis desen -> TEST icin kullanilacak
        desenler[ad] = {str(i): float(v) for i, v in havuzla(ofs, boy, haric=None).items()}
        satirlar.append(
            {
                "rejim": rejim,
                "yon": ad,
                "grup": int(len(desenler[ad])),
                "kazanc_rejim": kaz,
                "kazanc_toplam": kaz * pay,
                "blok_kazanc": blok_kazanc,
                "blok_kappa": blok_kappa,
                "blok_Q": blok_q,
                "kappa_tutarli": all(
                    np.sign(blok_kappa[k]) == np.sign(blok_kappa[BLOKLAR[0]]) for k in BLOKLAR
                ),
            }
        )
        print(
            f"  {rejim.upper():5s} {ad:18s} G={len(desenler[ad]):>5d}"
            f"  kazanc_rejim={kaz:+.6f}  kazanc_toplam={kaz * pay:+.6f}"
            f"  kappa={[round(blok_kappa[k], 3) for k in BLOKLAR]}"
            f"  {'TUTARLI' if satirlar[-1]['kappa_tutarli'] else 'ISARET OYNAK'}"
        )
    return satirlar, desenler


def main() -> None:
    te = test_cercevesi()
    ts = te[te["sicak"]].reset_index(drop=True)
    tso = te[~te["sicak"]].reset_index(drop=True)
    print(f"TEST: sicak {len(ts):,}  soguk {len(tso):,}")

    tum_satir, tum_desen = [], {}

    # ---------------- SICAK ----------------
    bl = SIC.bloklari_kur()
    taban = {k: SIC.taban_r(bl[k]) for k in BLOKLAR}
    print("\nSICAK blok boyutlari:", {k: bl[k].n for k in BLOKLAR})
    sic_gruplar = {
        "ilce": lambda b, r: b.cerceve["ilce"].to_numpy(),
        "seviye_desili10": lambda b, r: desil(np.maximum(r + b.lgc, 0.0), 10),
        "seviye_desili20": lambda b, r: desil(np.maximum(r + b.lgc, 0.0), 20),
        "kva_kovasi": lambda b, r: b.cerceve["kova"].to_numpy(),
        "ay": lambda b, r: b.cerceve["ay"].to_numpy(),
        "ilce_x_kova": lambda b, r: (
            b.cerceve["ilce"].astype(str) + "|" + b.cerceve["kova"].astype(str)
        ).to_numpy(),
        "trafo": lambda b, r: b.cerceve["tanim"].to_numpy(),
    }
    s, d = rejim_kos(
        bl,
        taban,
        sic_gruplar,
        SICAK_PAY,
        "sicak",
        lambda b, r: b.lgy - np.maximum(r + b.lgc, 0.0),
    )
    tum_satir += s
    tum_desen["sicak"] = d
    del bl, taban

    # ---------------- SOGUK ----------------
    SOG = _soguk_modul()
    sb = SOG.tum_bloklar()
    stab = {k: SOG.taban_r(sb[k]) for k in BLOKLAR}
    print("\nSOGUK blok boyutlari:", {k: sb[k].n for k in BLOKLAR})
    sog_gruplar = {
        "ilce": lambda b, r: b.ilce,
        "seviye_desili10": lambda b, r: desil(r + b.lgc, 10),
        "seviye_desili20": lambda b, r: desil(r + b.lgc, 20),
        "kva_kovasi": lambda b, r: kova_kva(b.guc),
        "ay": lambda b, r: pd.to_datetime(b.tarih).month.to_numpy(),
        "ilce_x_kova": lambda b, r: np.char.add(
            np.char.add(b.ilce.astype(str), "|"), kova_kva(b.guc).astype(str)
        ),
        "trafo": lambda b, r: b.tanim,
    }
    s, d = rejim_kos(sb, stab, sog_gruplar, SOGUK_PAY, "soguk", lambda b, r: b.lgy - (r + b.lgc))
    tum_satir += s
    tum_desen["soguk"] = d

    (BURA / "prob_simulasyon.json").write_text(
        json.dumps(tum_satir, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (BURA / "desenler.json").write_text(
        json.dumps(tum_desen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nyazildi: prob_simulasyon.json + desenler.json")


if __name__ == "__main__":
    main()
