"""H21 -- b_soguk ILK KEZ MEVSIMSEL IKIZDE (yaz25) OLCULUYOR.

NEDEN SIMDI MUMKUN
------------------
Bu gece ``scripts/uret_soguk_tahmin.py`` yazildi ve
``data/interim/deney/soguk_tahmin_yaz25.npz`` uretildi (5 tohum x
cat/xgb/lgbm, saf soguk uzman maske=1,00, cat depth 7 -- kis26 onbellegiyle
BIREBIR ayni ayar). Daha once yalnizca kis26 vardi.

KALICI KURAL 10: kis26'da olculen seviye kazanci kesme-etiket mevsim
bitisikliginden besleniyor ve teste tasinmiyor; boyle oneriler yaz25'te
olculmelidir. yaz25 = 2025-04-01..07-31 = TEST penceresinin mevsimsel ikizi.

ILK SAYILAR
-----------
    kis26 (5 tohum x 3 aile ort)   b = +0,3273
    yaz25 (5 tohum x 3 aile ort)   b = +0,1334
    docs/43 YOL 2 ikiz-capa        b = +0,1454   <- BAGIMSIZ yol, cok yakin
    on kayitli delta               0,16

AILE FARKI ONEMLI
-----------------
``gun_ekseni/*_taban.npy`` YALNIZ cat (deney_gun_ekseni_dogrula.py:121) ve
orada b = +0,0595 cikmisti. Uretim harmani ise cat/xgb/lgbm = 3/1/1
(deney_bayatlik.py:66, deney_soguk_buzme.py:49). Dogru sayi URETIM
HARMANIYLA olculmelidir -- bu betik onu yapar.

CIKTI: delta_soguk icin NOKTA TAHMIN (buzulmemis), kirpma tablosu, ve
kis26 ile karsilastirma.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
HARMAN = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0}
P_SOGUK = 0.22159


def blok_oku(blok: str):
    npz = KOK / f"data/interim/deney/soguk_tahmin_{blok}.npz"
    meta = KOK / f"data/interim/{blok}_soguk_meta.parquet"
    if not npz.exists() or not meta.exists():
        return None
    z = np.load(npz)
    m = pd.read_parquet(meta).reset_index(drop=True)
    tohum = sorted({k.split("_")[0] for k in z.files})
    pay = sum(HARMAN.values())
    tah = {}
    for t in tohum:
        var = [a for a in HARMAN if f"{t}_{a}" in z.files]
        if len(var) < len(HARMAN):
            continue
        tah[t] = sum(HARMAN[a] * z[f"{t}_{a}"].astype("float64") for a in var) / pay
    return m, tah, z


def main() -> int:
    for blok in ("yaz25", "guz25", "kis26"):
        r = blok_oku(blok)
        if r is None:
            print(f"{blok}: onbellek YOK (uretiliyor olabilir)\n")
            continue
        m, tah, z = r
        if not tah:
            print(f"{blok}: tam tohum yok\n")
            continue
        lgy = np.log1p(np.clip(m["y"].to_numpy(dtype="float64"), 0, None))
        print("=" * 88)
        print(
            f"{blok.upper()}   {len(m):,} satir, {m.tanim.nunique():,} trafo, "
            f"{len(tah)} tohum (harman cat/xgb/lgbm = 3/1/1)"
        )
        print("=" * 88)

        per = [float((lgy - v).mean()) for v in tah.values()]
        v = np.array(per)
        sh = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        print(f"  b = {v.mean():+.4f}   eslenik SH {sh:.4f}   t {v.mean() / sh:+.2f}")
        print(f"  tohumlar: {[round(x, 4) for x in per]}")

        # aile bazinda
        print("\n  AILE BAZINDA (tek basina):")
        for a in ("cat", "xgb", "lgbm"):
            pa = [
                float((lgy - z[f"{t}_{a}"].astype("float64")).mean())
                for t in tah
                if f"{t}_{a}" in z.files
            ]
            if pa:
                print(f"    {a:<5} b = {np.mean(pa):+.4f}  (tohum {[round(x, 3) for x in pa]})")

        # KIRPMA (kural 1)
        bi, _ = pd.factorize(m["tanim"])
        nb = int(bi.max()) + 1
        print(f"\n  KIRPMA TABLOSU (trafo bazli, {nb} trafo)")
        print(f"    {'K':>4} {'b':>9} {'SH':>8} {'t':>7} {'kalan trafo':>12} {'kalan satir':>12}")
        for Kk in (0, 1, 5, 10, 25, 50):
            if Kk >= nb:
                break
            pv, kt, ks = [], 0, 0
            for tv in tah.values():
                d = lgy - tv
                katki = np.bincount(bi, d, minlength=nb) / np.maximum(
                    np.bincount(bi, minlength=nb), 1
                )
                at = np.argsort(-np.abs(katki))[:Kk]
                tut = ~np.isin(bi, at) if Kk else np.ones(len(d), bool)
                pv.append(float(d[tut].mean()))
                kt, ks = nb - Kk, int(tut.sum())
            a2 = np.array(pv)
            s2 = a2.std(ddof=1) / np.sqrt(len(a2)) if len(a2) > 1 else float("nan")
            print(
                f"    {Kk:>4} {a2.mean():>+9.4f} {s2:>8.4f} "
                f"{a2.mean() / s2:>+7.2f} {kt:>12,} {ks:>12,}"
            )
        print()

    print("=" * 88)
    print("DELTA KARARI")
    print("=" * 88)
    print("  Kural 10: TEST'in geometrisi yaz25'inkidir -> yaz25 olcumu esas.")
    print("  Kuadratik kayipta optimum delta = E[b]; NOKTA TAHMIN yazilir,")
    print("  buzulmez (yanilmanin siralama maliyeti sifir: LB en iyiyi gosterir).")
    print("  Karsilastirma: docs/43 YOL 2 ikiz-capasi +0,1454 (BAGIMSIZ yol).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
