"""SOGUK UZMAN MIMARI HARMANI -- onbellekten, uc blokta, uretim son islemiyle.

Onbellek: data/interim/deney/soguk_tahmin_{blok}.npz  (tohum x {cat,xgb,lgbm},
LOG uzayinda, uretim-sadik kurgu: maske=1.00, yalin 105 kolon, ek_koken=False,
cat depth=7).  Meta: data/interim/{blok}_soguk_meta.parquet (tanim/tarih/guc/y).

Uretim son islem zinciri (sogugun gordugu haliyle):
    1) buzme     r' = ort(r) + beta*(r - ort(r))          r = log1p(tahmin)-log1p(guc)
    2) gun olcek log1p += (c-1)*b_gun    b_gun: iki yonlu sabit etki, satir-merkezli
                                         TEMIZ alt kume (yas>=7 gun, >=60 gunluk trafo)
    3) seviye    log1p += delta

Olcut: RMSLE, np.clip(np.expm1(.),0,None) kirpmasiyla.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]

BLOKLAR = ("yaz25", "guz25", "kis26")
AILELER = ("cat", "xgb", "lgbm")
P_SOGUK = 0.22159
MIN_YAS, MIN_GUN = 7, 60


def yukle(blok: str) -> tuple[dict[str, np.ndarray], pd.DataFrame, list[int]]:
    z = np.load(KOK / "data/interim/deney" / f"soguk_tahmin_{blok}.npz")
    tohumlar = sorted({int(k.split("_")[0]) for k in z.files})
    meta = pd.read_parquet(KOK / "data/interim" / f"{blok}_soguk_meta.parquet")
    # aile -> tohum ortalamasi (uretim 15 tohumu ONCE ortalar, sonra skorlar)
    aile_ort = {a: np.mean([z[f"{t}_{a}"] for t in tohumlar], axis=0) for a in AILELER}
    per = {f"{t}_{a}": z[f"{t}_{a}"] for t in tohumlar for a in AILELER}
    per.update(aile_ort)
    return per, meta, tohumlar


def iki_yonlu(v: np.ndarray, bi: np.ndarray, gi: np.ndarray, tur: int = 400) -> np.ndarray:
    nb, ng = int(bi.max()) + 1, int(gi.max()) + 1
    mu = float(v.mean())
    a = np.zeros(nb)
    b = np.zeros(ng)
    cb = np.maximum(np.bincount(bi, minlength=nb), 1)
    cg = np.maximum(np.bincount(gi, minlength=ng), 1)
    for _ in range(tur):
        a = np.bincount(bi, v - mu - b[gi], minlength=nb) / cb
        b = np.bincount(gi, v - mu - a[bi], minlength=ng) / cg
        b -= b.mean()
    return b


def gun_profili(lg: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Uretimdeki son_islem_soguk_gunolcek.py ile birebir: TEMIZ alt kumeden
    gun bileseni, sonra TUM soguk satirlara yazilir ve satir-merkezlenir."""
    sc = meta.copy()
    sc["lg"] = lg
    ilk = sc.groupby("tanim")["tarih"].transform("min")
    yas = (sc["tarih"] - ilk).dt.days.to_numpy()
    say = sc.groupby("tanim")["tanim"].transform("size").to_numpy()
    temiz = (yas >= MIN_YAS) & (say >= MIN_GUN)
    if temiz.sum() < 1000:
        return np.zeros(len(sc))
    t = sc.loc[temiz]
    bi, _ = pd.factorize(t["tanim"])
    gi, gun = pd.factorize(t["tarih"])
    b = iki_yonlu(t["lg"].to_numpy(dtype="float64"), bi, gi)
    profil = pd.Series(b, index=pd.Index(gun, name="tarih")).sort_index()
    ek = sc["tarih"].map(profil).to_numpy(dtype="float64")
    ek = np.nan_to_num(ek)
    return ek - float(ek.mean())


def rmsle(y: np.ndarray, lg_tahmin: np.ndarray) -> float:
    t = np.clip(np.expm1(lg_tahmin), 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(t) - np.log1p(y)) ** 2)))


def mse(y: np.ndarray, lg_tahmin: np.ndarray) -> float:
    return rmsle(y, lg_tahmin) ** 2


def zincir_coz(
    r: np.ndarray,
    log_guc: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    betalar,
    celer,
    deltalar,
) -> tuple[float, tuple[float, float, float]]:
    """(beta, c, delta) uzerinde kaba kuvvet; en dusuk MSE ve ayarlari dondurur."""
    en_iyi = (np.inf, (np.nan, np.nan, np.nan))
    ort = float(r.mean())
    for beta in betalar:
        lg = ort + beta * (r - ort) + log_guc
        ek = gun_profili(lg, meta)
        for c in celer:
            lg2 = lg + (c - 1.0) * ek
            for delta in deltalar:
                m = mse(y, lg2 + delta)
                if m < en_iyi[0]:
                    en_iyi = (m, (beta, c, delta))
    return en_iyi


def main() -> int:
    adaylar: dict[str, tuple[float, float, float]] = {
        "cat (URETIM)": (1, 0, 0),
        "xgb": (0, 1, 0),
        "lgbm": (0, 0, 1),
        "cat/lgbm 3/1": (3, 0, 1),
        "cat/lgbm 2/1": (2, 0, 1),
        "cat/lgbm 1/1": (1, 0, 1),
        "cat/xgb 3/1": (3, 1, 0),
        "cat/xgb 2/1": (2, 1, 0),
        "cat/xgb 1/1": (1, 1, 0),
        "uclu 3/1/1": (3, 1, 1),
        "uclu 4/1/1": (4, 1, 1),
        "uclu 6/1/1": (6, 1, 1),
        "uclu 2/1/1": (2, 1, 1),
        "uclu 1/1/1": (1, 1, 1),
        "uclu 8/1/1": (8, 1, 1),
    }
    betalar = (1.00, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30)
    celer = (1.00, 1.15, 1.3301, 1.50, 1.60, 1.80, 2.00, 2.20)
    deltalar = tuple(np.round(np.arange(-0.30, 0.4001, 0.02), 4))

    veri = {}
    for b in BLOKLAR:
        per, meta, tohumlar = yukle(b)
        y = meta["y"].to_numpy(dtype="float64")
        log_guc = np.log1p(meta["guc"].to_numpy(dtype="float64"))
        veri[b] = (per, meta, y, log_guc, tohumlar)
        print(f"{b}: {len(y):,} soguk satir, tohum {tohumlar}, trafo {meta.tanim.nunique():,}")

    sonuc: dict[str, dict[str, dict]] = {}
    print("\n" + "=" * 108)
    print("HAM (son islem YOK) -- cold RMSLE")
    print("=" * 108)
    print(f"{'aday':16}" + "".join(f"{b:>12}" for b in BLOKLAR))
    ham = {}
    for ad, w in adaylar.items():
        satir = []
        for b in BLOKLAR:
            per, meta, y, log_guc, _ = veri[b]
            lg = sum(wi * per[a] for a, wi in zip(AILELER, w)) / sum(w)
            satir.append(rmsle(y, lg))
        ham[ad] = satir
        print(f"{ad:16}" + "".join(f"{v:>12.5f}" for v in satir))

    print("\n" + "=" * 108)
    print("URETIM SON ISLEMI YENIDEN COZULMUS (beta,c,delta blok bazinda optimum)")
    print("=" * 108)
    hdr = f"{'aday':16}"
    for b in BLOKLAR:
        hdr += f"{b + ' MSE':>13}{'dMSE':>11}"
    print(hdr + f"{'test dMSE':>12}{'ayni_yon':>10}")

    taban_mse = {}
    for ad, w in adaylar.items():
        satir = f"{ad:16}"
        dler = []
        kayit = {}
        for b in BLOKLAR:
            per, meta, y, log_guc, _ = veri[b]
            lg = sum(wi * per[a] for a, wi in zip(AILELER, w)) / sum(w)
            r = lg - log_guc
            m, ayar = zincir_coz(r, log_guc, y, meta, betalar, celer, deltalar)
            if ad == "cat (URETIM)":
                taban_mse[b] = m
            d = m - taban_mse[b]
            dler.append(d)
            kayit[b] = {"mse": m, "dmse": d, "beta": ayar[0], "c": ayar[1], "delta": ayar[2]}
            satir += f"{m:>13.5f}{d:>+11.5f}"
        test_d = P_SOGUK * dler[2]  # HUKUM kis26 ile
        ayni = all(v < 0 for v in dler) or all(v > 0 for v in dler)
        satir += f"{test_d:>+12.5f}{str(ayni):>10}"
        print(satir)
        sonuc[ad] = kayit

    print("\nSECILEN AYARLAR (blok bazinda beta/c/delta optimumu)")
    for ad, k in sonuc.items():
        print(
            f"  {ad:16}"
            + "  ".join(
                f"{b}: b={k[b]['beta']:.2f} c={k[b]['c']:.4f} d={k[b]['delta']:+.2f}"
                for b in BLOKLAR
            )
        )

    cikti = KOK / "experiments" / "egitim_soguk" / "harman_tarama.json"
    cikti.write_text(json.dumps({"ham": ham, "zincir": sonuc}, indent=2), encoding="utf-8")
    print(f"\nyazildi: {cikti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
