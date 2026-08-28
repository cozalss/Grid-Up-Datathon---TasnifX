"""ADAYLARIN KARNESI -- CV RMSLE, span geometrisi ve CV-OLCULMUS hizalanma.

Uc olcum yapilir:

1. TEK BASINA CV RMSLE  (uc blok). Kapi: uretim referansinin IKI KATINDAN
   kotu olmasin. Amac kazanmak degil, "felaket degil"i gostermek.

2. GEOMETRI. Aday test tahmininin ``v102``ye gore yonu ``u``:
      q_perp  = span'a dik enerji
      q_yeni  = mevcut 9 dik boyuttan da arindirildiktan sonra kalan
      kosinus = mevcut dik envanterle
   Kapi: ``q_perp >= 0.005``.

3. CV-OLCULMUS HIZALANMA (kapi degil, KANIT). LB probunun aritmetigi CV'de
   birebir tekrarlanabilir: uretim tabanini yeniden kurup
      L = <gercek - taban, d>/n ,  Q = |d|^2/n ,  kappa* = L/Q ,  kazanc = L^2/Q
   olculur. Bu, "f" nin dogrudan CV kestirimidir. LB'de test dagilimi farkli
   oldugu icin BIREBIR tasinmaz -- ama isaretin ve buyuklugun tek on gostergesi
   budur.

Kaggle'a hicbir sey gondermez.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

import ortak

AGIRLIK = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 1.4}
TOHUM = (1000, 1001, 1002)
BETA_SOGUK = 0.60
DELTA_SOGUK = 0.1046
ONB_AILE = ortak.KOK / "data/interim/aile_onbellek"


# ----------------------------------------------------------------------------- uretim tabani (CV)
def uretim_tabani() -> dict[str, dict]:
    """Blok basina yaklasik URETIM tahmini (log1p uzayinda), egitim satir sirasinda."""
    e = ortak.egitim(["tanim", "tarih", "guc", "tuketim", "_blok", "soguk_mu"])
    cik: dict[str, dict] = {}
    for b in ortak.BLOKLAR:
        m = e["_blok"].to_numpy() == b
        eb = e[m].reset_index(drop=True)
        sg = eb["soguk_mu"].to_numpy().astype(bool)
        lgy = np.log1p(np.clip(eb["tuketim"].to_numpy("float64"), 0.0, None))
        lgc = np.log1p(eb["guc"].to_numpy("float64"))
        taban = np.zeros(len(eb), dtype="float64")

        # SICAK -- uretim harmani (aile agirliklari x tohum ortalamasi)
        pay = sum(AGIRLIK.values())
        s = np.zeros(int((~sg).sum()), dtype="float64")
        for a, w in AGIRLIK.items():
            v = np.mean(
                [np.load(ONB_AILE / f"{b}_{t}_{a}_uretim.npy").astype("float64") for t in TOHUM],
                axis=0,
            )
            s += w * v
        taban[~sg] = s / pay

        # SOGUK -- cat/xgb/lgbm harmani + JS buzmesi + seviye ofseti
        z = np.load(ortak.KOK / f"data/interim/deney/soguk_tahmin_{b}.npz")
        meta = pd.read_parquet(ortak.KOK / f"data/interim/{b}_soguk_meta.parquet")
        meta["tanim"] = meta["tanim"].astype(str)
        aile = []
        for a in ("cat", "xgb", "lgbm"):
            k = [f"{t}_{a}" for t in TOHUM if f"{t}_{a}" in z.files]
            if k:
                aile.append(np.mean([z[i] for i in k], axis=0))
        sc = np.mean(aile, axis=0)
        rc = sc - np.log1p(meta["guc"].to_numpy("float64"))
        rc = rc.mean() + BETA_SOGUK * (rc - rc.mean()) + DELTA_SOGUK
        sc = rc + np.log1p(meta["guc"].to_numpy("float64"))
        anah_m = pd.Index(meta["tanim"] + "|" + meta["tarih"].astype(str))
        eb_s = eb[sg]
        anah_e = pd.Index(eb_s["tanim"].astype(str) + "|" + eb_s["tarih"].dt.strftime("%Y-%m-%d"))
        yer = anah_m.get_indexer(anah_e)
        if (yer < 0).any():
            raise RuntimeError(f"{b}: soguk meta hizalanmadi ({int((yer < 0).sum())} satir)")
        taban[sg] = sc[yer]

        taban = taban + _kuresel_delta(lgy, taban)
        cik[b] = {"lgy": lgy, "taban": taban, "lgc": lgc, "soguk": sg}
    return cik


def _kuresel_delta(lgy: np.ndarray, taban: np.ndarray) -> float:
    """Kirpma altinda optimum kuresel seviye kaymasi (kaba tarama + incelme)."""

    def mse(d: float) -> float:
        e = lgy - np.maximum(taban + d, 0.0)
        return float(np.dot(e, e) / len(e))

    en, enm = 0.0, mse(0.0)
    adim = 0.08
    for _ in range(6):
        for d in np.arange(en - 4 * adim, en + 4.001 * adim, adim):
            m = mse(float(d))
            if m < enm:
                en, enm = float(d), m
        adim /= 4.0
    return en


# ----------------------------------------------------------------------------- maskeler
def maskeler() -> dict[str, np.ndarray]:
    """Test satirlari uzerinde alt-nufus maskeleri."""
    t = ortak.test(["tanim", "guc"])
    tr = pd.read_csv(ortak.KOK / "data/raw/train.csv", usecols=["tanim"], dtype={"tanim": str})
    gecmis = set(tr["tanim"].unique())
    sg = ~t["tanim"].astype(str).isin(gecmis).to_numpy()
    return {"tum": np.ones(len(t), dtype=bool), "sicak": ~sg, "soguk": sg}


def cv_maskeler(tab: dict) -> dict[str, dict[str, np.ndarray]]:
    return {
        b: {
            "tum": np.ones(len(tab[b]["lgy"]), dtype=bool),
            "sicak": ~tab[b]["soguk"],
            "soguk": tab[b]["soguk"],
        }
        for b in ortak.BLOKLAR
    }


# ----------------------------------------------------------------------------- ana
def main() -> None:
    adaylar = sorted(p.stem for p in ortak.ONB.glob("*.npz"))
    if not adaylar:
        print("aday yok")
        return
    g = ortak.geo()
    ref = ortak.uretim_referansi()
    tab = uretim_tabani()
    mk = maskeler()
    cmk = cv_maskeler(tab)
    print("uretim CV referansi (havuzlanmis RMSLE):")
    for b in ortak.BLOKLAR:
        kend = ortak.rmsle_log(tab[b]["lgy"], tab[b]["taban"])
        print(f"   {b}: docs {ref[b]:.5f}   yeniden kurulan taban {kend:.5f}")

    v102 = g.v102
    sonuc = []
    for ad in adaylar:
        cv, tp = ortak.yukle_aday(ad)
        lgp = np.log1p(np.clip(tp, 0.0, None))
        satir: dict = {"ad": ad}
        # 1. CV RMSLE
        for b in ortak.BLOKLAR:
            if b in cv:
                satir[f"rmsle_{b}"] = ortak.rmsle(np.expm1(tab[b]["lgy"]), cv[b])
                satir[f"oran_{b}"] = satir[f"rmsle_{b}"] / ref[b]
        # 3. CV hizalanma (maskeli varyantlar dahil)
        for isim in ("tum", "sicak", "soguk"):
            L = Q = 0.0
            n = 0
            for b in ortak.BLOKLAR:
                if b not in cv:
                    continue
                msk = cmk[b][isim]
                d = np.zeros(len(msk), dtype="float64")
                d[msk] = np.log1p(np.clip(cv[b][msk], 0.0, None)) - tab[b]["taban"][msk]
                r = tab[b]["lgy"] - tab[b]["taban"]
                L += float(r @ d)
                Q += float(d @ d)
                n += len(d)
            satir[f"cvL_{isim}"] = L / n
            satir[f"cvQ_{isim}"] = Q / n
            satir[f"cvkappa_{isim}"] = (L / Q) if Q > 0 else 0.0
            satir[f"cvkazanc_{isim}"] = (L * L / Q / n) if Q > 0 else 0.0
        # 2. geometri (test uzerinde), maskeli varyantlar
        satir["kor_v102"] = float(np.corrcoef(lgp, v102)[0, 1])
        for isim, msk in mk.items():
            u = np.zeros(g.n, dtype="float64")
            u[msk] = lgp[msk] - v102[msk]
            r = g.olc(f"{ad}|{isim}", u)
            satir[f"Q_{isim}"] = r["Q"]
            satir[f"qperp_{isim}"] = r["q_perp"]
            satir[f"qyeni_{isim}"] = r["q_yeni"]
            satir[f"mkos_{isim}"] = f"{r['maks_kos_ad']} {r['maks_kos']:+.3f}"
            satir[f"mkosv_{isim}"] = abs(float(r["maks_kos"]))
        sonuc.append(satir)

    _yaz(sonuc, ref)
    json.dump(sonuc, open(ortak.CIK / "c_olc.json", "w"), indent=2, default=float)


def _yaz(sonuc: list[dict], ref: dict) -> None:
    print("\n" + "=" * 118)
    print("1) TEK BASINA CV RMSLE  (parantez: uretim referansina oran; kapi < 2.00)")
    print("=" * 118)
    print(f"{'aday':22}" + "".join(f"{b:>22}" for b in ortak.BLOKLAR))
    for s in sonuc:
        h = f"{s['ad'][:22]:22}"
        for b in ortak.BLOKLAR:
            k = f"rmsle_{b}"
            h += f"{s[k]:>14.5f} ({s['oran_' + b]:.2f}) " if k in s else f"{'-':>22}"
        print(h)

    print("\n" + "=" * 118)
    print("2) GEOMETRI -- yon = log1p(aday) - log1p(v102), maskeye gore")
    print("=" * 118)
    print(
        f"{'aday|maske':30}{'Q':>11}{'q_perp':>11}{'q_yeni':>11}"
        f"{'span payi':>11}  mevcut diklerle maks kosinus"
    )
    for s in sonuc:
        for isim in ("tum", "sicak", "soguk"):
            Q, qp, qy = s[f"Q_{isim}"], s[f"qperp_{isim}"], s[f"qyeni_{isim}"]
            sp = 1.0 - qp / Q if Q > 0 else 0.0
            print(
                f"{s['ad'][:22] + '|' + isim:30}{Q:>11.5f}{qp:>11.5f}{qy:>11.5f}"
                f"{sp:>11.3f}  {s['mkos_' + isim]}"
            )

    print("\n" + "=" * 118)
    print("3) CV-OLCULMUS HIZALANMA  (L=<gercek-taban, d>/n, kappa*=L/Q, kazanc=L^2/Q)")
    print("=" * 118)
    print(
        f"{'aday':22}"
        + "".join(f"{'L_' + i:>11}{'Q_' + i:>11}{'k*_' + i:>9}{'kaz_' + i:>11}" for i in ("tum",))
        + f"{'k*_sicak':>10}{'kaz_sicak':>11}{'k*_soguk':>10}{'kaz_soguk':>11}"
    )
    for s in sonuc:
        print(
            f"{s['ad'][:22]:22}{s['cvL_tum']:>+11.5f}{s['cvQ_tum']:>11.5f}"
            f"{s['cvkappa_tum']:>+9.3f}{s['cvkazanc_tum']:>11.5f}"
            f"{s['cvkappa_sicak']:>+10.3f}{s['cvkazanc_sicak']:>11.5f}"
            f"{s['cvkappa_soguk']:>+10.3f}{s['cvkazanc_soguk']:>11.5f}"
        )


if __name__ == "__main__":
    main()
