# ruff: noqa
"""EKSEN 3 (son): URETIM MODELININ test penceresinde SEVIYE yanliligi var mi?

kis26/yaz25 foldlarindaki +0,19 / +0,14 yanliligi FOLD'un kendi blogunu
gormemesinden kaynakli olabilir (leave-one-out yapaylik). Uretim modeli
Nis-Tem 2025 etiketlerini GORUYOR. Ayrimi ETIKETSIZ yapabiliriz:

    beklenen_2026_seviye_i = gercek_2025_NisTem_i
                           + (gercek_2026_OcaMar_i - gercek_2025_OcaMar_i)   <- YoY kaymasi
                           + egim_CDD * (CDD_2026_NisTem - CDD_2025_NisTem)  <- hava duzeltmesi

Test etiketi KULLANILMAZ (kural 5). Karsilastirma: v50/v55'in tahmin ettigi seviye.

    python scripts/eksen3_h_seviye_capasi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

EGIM_CDD = 0.07048  # yaz25 & guz25 GERCEK gun ekseni egimlerinin ortalamasi


def ofs(t: pd.DataFrame) -> np.ndarray:
    return np.log1p(np.clip(t["tuketim"].to_numpy(dtype="float64"), 0, None)) - np.log1p(
        t["guc"].to_numpy(dtype="float64")
    )


def hava_cdd(ilce_w: pd.Series) -> pd.Series:
    h = pd.read_parquet(
        KOK / "data" / "external" / "hava_gunluk.parquet",
        columns=["ilce_key", "tarih", "sicaklik_ort"],
    )
    h["tarih"] = pd.to_datetime(h["tarih"]).dt.normalize()
    h = h[h["ilce_key"].isin(ilce_w.index)].copy()
    h["w"] = h["ilce_key"].map(ilce_w).astype("float64")
    h["sw"] = h["sicaklik_ort"] * h["w"]
    g = h.groupby("tarih")[["w", "sw"]].sum()
    return ((g["sw"] / g["w"]) - 22.0).clip(lower=0)


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "guc", "tarih", "tuketim"],
        dtype={"tanim": str},
    )
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "guc", "tarih", "lokasyon"],
        dtype={"tanim": str},
    )
    tr["tarih"] = pd.to_datetime(tr["tarih"])
    te["tarih"] = pd.to_datetime(te["tarih"])
    tr["ofs"] = ofs(tr)

    def pencere(a: str, b: str) -> pd.DataFrame:
        s = tr[(tr["tarih"] >= a) & (tr["tarih"] <= b)]
        return s.groupby("tanim").agg(ofs=("ofs", "mean"), n=("ofs", "size"))

    A = pencere("2025-04-01", "2025-07-31")  # gecen yil ayni pencere GERCEK
    C = pencere("2025-01-01", "2025-03-31")  # gecen yil kesme oncesi
    D = pencere("2026-01-01", "2026-03-31")  # bu yil kesme oncesi

    tp = pd.read_parquet(KOK / "data/interim/deney/test.parquet", columns=["ilce_key"])
    cdd = hava_cdd(tp["ilce_key"].value_counts(normalize=True))
    c25 = float(cdd[(cdd.index >= "2025-04-01") & (cdd.index <= "2025-07-31")].mean())
    c26 = float(cdd[(cdd.index >= "2026-04-01") & (cdd.index <= "2026-07-31")].mean())
    hava_duz = EGIM_CDD * (c26 - c25)

    print("=" * 100)
    print("ETIKETSIZ SEVIYE CAPASI -- test penceresi 2026-04-01..07-31")
    print("=" * 100)
    print(f"  ort CDD22  2025 Nis-Tem {c25:.3f}   2026 Nis-Tem {c26:.3f}   fark {c26 - c25:+.3f}")
    print(f"  hava duzeltmesi = {EGIM_CDD:.5f} x {c26 - c25:+.3f} = {hava_duz:+.4f} log birim")

    ortak = A.index.intersection(C.index).intersection(D.index)
    E = pd.DataFrame(
        {
            "A": A.loc[ortak, "ofs"],
            "C": C.loc[ortak, "ofs"],
            "D": D.loc[ortak, "ofs"],
        }
    )
    E["yoy"] = E["D"] - E["C"]
    E["beklenen"] = E["A"] + E["yoy"] + hava_duz
    print(f"\n  ortak trafo (2025 Nis-Tem + Oca-Mar 2025 + Oca-Mar 2026): {len(E):,}")
    print(
        f"  YoY kaymasi (Oca-Mar 2026 - Oca-Mar 2025):  ort {E['yoy'].mean():+.4f}"
        f"   medyan {E['yoy'].median():+.4f}   std {E['yoy'].std():.4f}"
    )

    for isim, dosya in (
        ("v50 (LB 1,01686)", "tuketim_v50_nihai30.csv"),
        ("v55 (LB 1,01591)", "tuketim_v55_gunolcek.csv"),
        ("v66 (c=1,335)", "tuketim_v66_c1335.csv"),
    ):
        yol = KOK / "submissions" / dosya
        if not yol.exists():
            print(f"\n  {isim}: dosya yok")
            continue
        s = pd.read_csv(yol)
        m = te.merge(s, on="id", how="left")
        m = m[(m["tarih"] >= "2026-04-01") & (m["tarih"] <= "2026-07-31")]
        m["ofs"] = ofs(m)
        P = m.groupby("tanim").agg(P=("ofs", "mean"), n=("ofs", "size"))
        X = E.join(P, how="inner").dropna()
        fark = X["beklenen"] - X["P"]
        w = X["n"]
        agir = float((fark * w).sum() / w.sum())
        print(f"\n  --- {isim}   ortak trafo {len(X):,}  test satiri {int(w.sum()):,}")
        print(
            f"      tahmin ort seviye {float((X['P'] * w).sum() / w.sum()):+.4f}"
            f"   beklenen {float((X['beklenen'] * w).sum() / w.sum()):+.4f}"
        )
        print(
            f"      KESTIRILEN SEVIYE YANLILIGI  b_hat = {agir:+.4f}"
            f"   (medyan {fark.median():+.4f}, std {fark.std():.4f})"
        )
        pay_hot = float(w.sum()) / len(te)
        print(
            f"      optimum sabit delta uygulanirsa dMSE = -b^2 * pay = "
            f"{-(agir**2) * pay_hot:+.5f}   (pay {pay_hot:.4f})"
        )
        print(f"      -> yeni RMSLE ~ {np.sqrt(1.03207 - (agir**2) * pay_hot):.5f}")
        # K en buyuk trafo atilinca (kural 1)
        kat = (fark * w).abs().sort_values(ascending=False)
        satir = "      K atilinca b_hat: "
        for K in (0, 1, 5, 10, 25, 50):
            idx = kat.index[K:]
            satir += f"K={K}:{float((fark[idx] * w[idx]).sum() / w[idx].sum()):+.4f}  "
        print(satir)

    print()
    print("=" * 100)
    print("KARSILASTIRMA: FOLDLARIN kendi yanliligi (leave-one-out) vs URETIM")
    print("=" * 100)
    print("  yaz25 foldu   b = +0,1401   (kendi blogunu GORMEDEN)")
    print("  kis26 foldu   b = +0,1886   (kendi blogunu GORMEDEN)")
    print("  guz25 foldu   b = -0,3419   (kendi blogunu GORMEDEN)")
    print("  URETIM (test) b_hat = yukarida -- uretim TUM bloklari gorur.")

    print()
    print("=" * 100)
    print("DUYARLILIK: hava duzeltmesi ve YoY varsayimi ne kadar onemli?")
    print("=" * 100)
    s = pd.read_csv(KOK / "submissions" / "tuketim_v55_gunolcek.csv")
    m = te.merge(s, on="id", how="left")
    m = m[(m["tarih"] >= "2026-04-01") & (m["tarih"] <= "2026-07-31")]
    m["ofs"] = ofs(m)
    P = m.groupby("tanim").agg(P=("ofs", "mean"), n=("ofs", "size"))
    X = E.join(P, how="inner").dropna()
    w = X["n"]
    print(f"  {'varyant':<44}{'b_hat':>10}{'dMSE':>11}")
    for isim, bek in (
        ("tam (YoY + hava duzeltmesi)", X["A"] + X["yoy"] + hava_duz),
        ("YoY var, hava duzeltmesi YOK", X["A"] + X["yoy"]),
        ("YoY YOK, hava duzeltmesi var", X["A"] + hava_duz),
        ("ne YoY ne hava (saf gecen yil)", X["A"]),
        ("YoY x0,5 + hava", X["A"] + 0.5 * X["yoy"] + hava_duz),
    ):
        b = float(((bek - X["P"]) * w).sum() / w.sum())
        print(f"  {isim:<44}{b:>+10.4f}{-(b**2) * float(w.sum()) / len(te):>+11.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
