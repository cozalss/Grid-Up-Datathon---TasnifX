"""H18b -- KUYRUK REJIMI: kesmenin son gunlerinde dogan "sicak" trafolar.

BULGU (h18)
-----------
Iki ORTUSMEYEN kesmede, kesmeden <=6 gun once dogmus "SICAK" trafolar
suregelen sicaklardan sistematik olarak DAHA COK az tahmin ediliyor:

    blok    kesme        KUYRUK<=6g            SUREN>180g      FAZLA
    guz25   2025-07-31   +0,1270 (182 trafo)   -0,3484         +0,475
    kis26   2025-11-30   +0,5623 (202 trafo)   +0,2091         +0,353

Isaret IKI BLOKTA DA ayni, 3/3 tohum, gercek n. Kural 9 geciyor.

NEDEN TASINIR: bu bir MUTLAK seviye degil, ayni blok icinde GRUPLAR ARASI
FARK. Fold'un gelecegi gormesi butun gruplari benzer etkiler, farkta buyuk
olcude sadelesir -- mutlak seviyede sadelesmez. (docs/41'in "yaz25/guz25
gelecegi goruyor" uyarisi mutlak seviye icindir.)

TEST TARAFI
-----------
2026-03-26..31 arasinda ILK KAYDI olan 356 trafonun 353'u TEST'te:
**29.873 satir = testin %4,18'i**. Train kayitlari: medyan 2 kayit (min 1,
max 6). Yani model onlari SICAK sayiyor (tanim train'de var) ama gecmisleri
pratikte YOK -- sicak/soguk ikili ayriminin yakalamadigi UCUNCU REJIM.

BU BETIK
--------
1. FAZLA'yi iki blokta tohum bazinda olcer, eslenik SH verir.
2. KIRPMA TABLOSU (kural 1) -- kazanc birkac trafodan mi?
3. Etkinin PARTI dogumundan mi yoksa KISA GECMIS'ten mi geldigini ayirir.
4. Test payiyla dMSE'yi hesaplar.
5. Modelin bunu zaten kismen bilip bilmedigini (bayatlik kolonlari) sinar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
BLOKLAR = {"guz25": "2025-07-31", "kis26": "2025-11-30"}
TEST_PAY = 29873 / 714688


def yukle(blok: str, kesme: str, ilk: pd.Series):
    z = np.load(KOK / f"data/interim/eksen5/kos_lgbm_{blok}.npz", allow_pickle=True)
    tan = z["tanim"].astype(str)
    r = np.log1p(np.clip(z["gercek"].astype("float64"), 0, None)) - z["lg"].astype("float64")
    tohum = sorted({k.split("_")[1] for k in z.files if k.startswith("Aplus_")})
    tah = {s: z[f"Aplus_{s}"].astype("float64") for s in tohum}
    K = pd.Timestamp(kesme)
    isl = pd.Series(tan).map(ilk)
    gg = (K - isl).dt.days.to_numpy() + 1
    return tan, r, tah, tohum, gg, isl


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    ilk = tr.groupby("tanim")["tarih"].min()
    parti = ilk.groupby(ilk).size()

    fazlalar = {}
    for blok, kesme in BLOKLAR.items():
        tan, r, tah, tohum, gg, isl = yukle(blok, kesme, ilk)
        kuy = (gg >= 1) & (gg <= 6)
        ref = gg > 180
        print("=" * 92)
        print(
            f"{blok}  kesme {kesme}   KUYRUK {int(kuy.sum()):,} satir / "
            f"{len(set(tan[kuy]))} trafo   REF {int(ref.sum()):,} satir / "
            f"{len(set(tan[ref]))} trafo"
        )
        print("=" * 92)
        per = []
        for s in tohum:
            per.append(float((r[kuy] - tah[s][kuy]).mean() - (r[ref] - tah[s][ref]).mean()))
        v = np.array(per)
        sh = v.std(ddof=1) / np.sqrt(len(v))
        fazlalar[blok] = (float(v.mean()), float(sh))
        print(
            f"  FAZLA (kuyruk - suren) = {v.mean():+.4f}  eslenik SH {sh:.4f}  "
            f"t {v.mean() / sh:+.2f}  tohumlar {[round(x, 3) for x in per]}"
        )

        # --- PARTI mi KISA GECMIS mi?
        bs = pd.Series(tan).map(ilk.map(parti)).to_numpy()
        print("\n  MEKANIZMA AYRIMI (kuyruk icinde):")
        for ad, m in (
            ("kuyruk & TOPLU >=100", kuy & (bs >= 100)),
            ("kuyruk & tekil <100", kuy & (bs < 100)),
        ):
            if m.sum() < 200:
                print(f"    {ad:<24} yetersiz ({int(m.sum())} satir)")
                continue
            p2 = [float((r[m] - tah[s][m]).mean() - (r[ref] - tah[s][ref]).mean()) for s in tohum]
            print(
                f"    {ad:<24} {int(m.sum()):>7,} satir "
                f"{len(set(tan[m])):>4} trafo  fazla {np.mean(p2):+.4f}"
            )

        # --- KIRPMA (kural 1)
        print("\n  KIRPMA TABLOSU (kuyruk grubu, trafo bazli)")
        print(
            f"    {'K':>4} {'fazla':>9} {'SH':>8} {'t':>7} {'kalan trafo':>12} {'kalan satir':>12}"
        )
        tk = tan[kuy]
        bi, _ = pd.factorize(tk)
        nb = int(bi.max()) + 1
        for Kk in (0, 1, 5, 10, 25, 50):
            if Kk >= nb:
                break
            pv, kt, ks = [], 0, 0
            for s in tohum:
                d = r[kuy] - tah[s][kuy]
                katki = np.bincount(bi, d, minlength=nb) / np.maximum(
                    np.bincount(bi, minlength=nb), 1
                )
                at = np.argsort(-katki)[:Kk]
                tut = ~np.isin(bi, at) if Kk else np.ones(len(d), bool)
                pv.append(float(d[tut].mean() - (r[ref] - tah[s][ref]).mean()))
                kt, ks = nb - Kk, int(tut.sum())
            a = np.array(pv)
            s2 = a.std(ddof=1) / np.sqrt(len(a))
            print(
                f"    {Kk:>4} {a.mean():>+9.4f} {s2:>8.4f} {a.mean() / s2:>+7.2f} "
                f"{kt:>12,} {ks:>12,}"
            )
        print()

    print("=" * 92)
    print("HUKUM")
    print("=" * 92)
    v = [f[0] for f in fazlalar.values()]
    print(
        "\n  bloklar: " + "  ".join(f"{k} {f[0]:+.4f} (SH {f[1]:.4f})" for k, f in fazlalar.items())
    )
    print(
        f"  ISARET TUTARLILIGI: {'GECTI' if all(x > 0 for x in v) or all(x < 0 for x in v) else 'KALDI'}"
    )
    ort = float(np.mean(v))
    # muhafazakar: iki blogun KUCUGU
    muh = float(min(np.abs(v))) * np.sign(ort)
    print(f"  ortalama fazla {ort:+.4f}   muhafazakar (kucuk blok) {muh:+.4f}")
    print(f"\n  TEST payi {TEST_PAY:.4f} (29.873 / 714.688)")
    for ad, b in (("ortalama", ort), ("muhafazakar", muh), ("%50 buzulmus", 0.5 * muh)):
        print(f"    delta={b:+.4f} ({ad:<12}) -> dMSE {-TEST_PAY * b**2:+.6f}")
    print("\n  ESIK -0,002.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
