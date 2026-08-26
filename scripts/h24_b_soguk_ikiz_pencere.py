"""H24 -- b_soguk: TEST'in YAPISINI birebir tasiyan tek alt pencere.

SORUN
-----
delta_soguk=0,16 karari uc fold'un ortalamasina dayaniyordu:
    yaz25 +0,1056 | guz25 +0,0725 | kis26 +0,3017
Ama uc fold'un HICBIRI test'in yapisinda degil:

    fold    ufuk          hedefin MEVSIMSEL IKIZI egitimde mi?
    yaz25   GELECEK var   HAYIR (Nis-Tem hicbir yerde yok)
    guz25   GELECEK var   HAYIR
    kis26   yalniz gecmis KISMEN
    TEST    yalniz gecmis EVET (Nis-Tem 2025 train'de)

Iki bilesen:
    (i)  yillik surukleme -- model ekstrapole edemiyor
    (ii) mevsim ekstrapolasyon hatasi -- hedefin ikizi egitimde yoksa buyur

    gelecegi goren fold: (i)=0  -> b ~ (ii)      = 0,07..0,11
    kis26:               (i)+(ii) ikisi de var   = 0,3017
    TEST:                (i) VAR, (ii) KUCUK     = ?

COZUM -- kis26'nin ICINDE test yapisini bul
-------------------------------------------
``son_islem_seviye.py`` belgeliyor: kis26 icinde YALNIZCA 2026 Sub-Mar
gecerlidir, cunku mevsimsel ikizi (2025-02-01..03-31) o fold'un
ETIKETLERINDE vardir -- tipki test penceresinin ikizinin (2025 Nis-Tem)
uretim etiketlerinde olmasi gibi. Aralik/Ocak'in ikizi YOKTUR (train
2025-01-01'de basliyor), oradaki daha buyuk kayma mevsim ekstrapolasyonudur.

Yani **kis26 SOGUK, 2026 Sub-Mar** = test'in yapisinin BIREBIR analogu:
yalniz-gecmis ufuk VE hedefin mevsimsel ikizi egitimde.

Bu betik o alt pencereyi ayrı olcer. Cikan sayi E[b]'nin en iyi tek
kestirimidir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
HARMAN = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0}
P_SOGUK = 0.22159


def main() -> int:
    z = np.load(KOK / "data/interim/deney/soguk_tahmin_kis26.npz")
    m = pd.read_parquet(KOK / "data/interim/kis26_soguk_meta.parquet").reset_index(drop=True)
    tohum = sorted({k.split("_")[0] for k in z.files})
    pay = sum(HARMAN.values())
    tah = {
        t: sum(HARMAN[a] * z[f"{t}_{a}"].astype("float64") for a in HARMAN) / pay
        for t in tohum
        if all(f"{t}_{a}" in z.files for a in HARMAN)
    }
    lgy = np.log1p(np.clip(m["y"].to_numpy(dtype="float64"), 0, None))
    ay = m["tarih"].dt.to_period("M").astype(str).to_numpy()

    print("=" * 88)
    print("kis26 SOGUK -- AY BAZINDA yanlilik (mevsimsel ikiz VAR mi?)")
    print("=" * 88)
    print(f"\n  {'ay':<10} {'satir':>8} {'trafo':>7} {'ikiz egitimde?':<16} {'b':>9} {'SH':>8}")
    ikiz = {
        "2025-12": "HAYIR (2024-12 yok)",
        "2026-01": "HAYIR (2025-01 kismi)",
        "2026-02": "EVET (2025-02)",
        "2026-03": "EVET (2025-03)",
    }
    for a in sorted(set(ay)):
        msk = ay == a
        per = [float((lgy[msk] - v[msk]).mean()) for v in tah.values()]
        v = np.array(per)
        sh = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        print(
            f"  {a:<10} {int(msk.sum()):>8,} {m.loc[msk, 'tanim'].nunique():>7,} "
            f"{ikiz.get(a, '?'):<16} {v.mean():>+9.4f} {sh:>8.4f}"
        )

    print("\n" + "=" * 88)
    print("TEST YAPISININ ANALOGU: 2026 Sub-Mar (ikiz egitimde VAR)")
    print("=" * 88)
    gec = np.isin(ay, ["2026-02", "2026-03"])
    yok = np.isin(ay, ["2025-12", "2026-01"])
    for ad, msk in (
        ("IKIZ VAR  (Sub-Mar)", gec),
        ("IKIZ YOK  (Ara-Oca)", yok),
        ("TUM kis26", np.ones(len(ay), bool)),
    ):
        per = [float((lgy[msk] - v[msk]).mean()) for v in tah.values()]
        v = np.array(per)
        sh = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        print(
            f"  {ad:<22} {int(msk.sum()):>7,} satir  b = {v.mean():+.4f}  "
            f"SH {sh:.4f}  t {v.mean() / sh:+.2f}"
        )

    # kirpma (kural 1) -- ikiz-var alt penceresinde
    tan = m.loc[gec, "tanim"].to_numpy()
    bi, _ = pd.factorize(tan)
    nb = int(bi.max()) + 1
    print(f"\n  KIRPMA (Sub-Mar, {nb} trafo)")
    print(f"    {'K':>4} {'b':>9} {'kalan trafo':>12} {'kalan satir':>12}")
    for K in (0, 1, 5, 10, 25, 50):
        if nb <= K:
            break
        per = []
        for v in tah.values():
            d = lgy[gec] - v[gec]
            katki = np.bincount(bi, d, minlength=nb) / np.maximum(np.bincount(bi, minlength=nb), 1)
            at = np.argsort(-np.abs(katki))[:K]
            tut = ~np.isin(bi, at) if K else np.ones(len(d), bool)
            per.append(float(d[tut].mean()))
        print(f"    {K:>4} {np.mean(per):>+9.4f} {nb - K:>12,} {int(tut.sum()):>12,}")

    print("\n" + "=" * 88)
    print("E[b] KARARI")
    print("=" * 88)
    per = [float((lgy[gec] - v[gec]).mean()) for v in tah.values()]
    b_analog = float(np.mean(per))
    print(f"  test-yapisi analogu (kis26 soguk, Sub-Mar) = {b_analog:+.4f}")
    print("  fold-free ikiz capasi (docs/43 YOL 2)      = +0,1454")
    print("  bootstrap ort (docs/43)                    = +0,1764")
    print(
        f"\n  {'delta':>7} {'E[b]=0,145':>12} {'E[b]=0,175':>12} "
        f"{'E[b]=0,22':>12} {'E[b]=0,30':>12}"
    )
    for dlt in (0.16, 0.18, 0.20, 0.22, 0.25, 0.30):
        satir = f"  {dlt:>7.2f}"
        for eb in (0.145, 0.175, 0.22, 0.30):
            satir += f" {-(2 * P_SOGUK * dlt * eb - P_SOGUK * dlt**2):>+12.5f}"
        print(satir)
    print("\n  (NEGATIF = kazanc. Her sutunda en iyi delta = o sutunun E[b]'si.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
