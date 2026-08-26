"""H3-A -- 2026-05-11 TOPLU KATILIM KOHORTU DOGRULAMASI.

Sorular:
  1. Test soguk (test-only) trafolarin ILK GUN dagilimi nedir? 2026-05-11'de
     gercekten 1.326 trafo mi var, ve soguk satirlarin %68'ini mi tasiyor?
  2. Train'de ayni gun 100+ trafo dogan TOPLU KATILIM gunleri hangileri?
  3. yaz25 ikizi (2025-04-01..07-31) icinde hangi kohortlar var?
  4. 2026-05-11 kohortunun test ufku 11 Mayis'tan basliyor -- kismi gun/rampa?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
TR0, TR1 = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-03-31")
TE0, TE1 = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-07-31")


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
    )
    te = pd.read_csv(
        KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
    )
    tr_tanim = set(tr["tanim"].unique())
    te_tanim = set(te["tanim"].unique())
    soguk = te_tanim - tr_tanim
    sicak = te_tanim & tr_tanim
    print(f"train trafo {len(tr_tanim):,} | test trafo {len(te_tanim):,}")
    print(
        f"SOGUK (test-only) {len(soguk):,} | SICAK (ortak) {len(sicak):,} | "
        f"train-only {len(tr_tanim - te_tanim):,}"
    )

    te["soguk"] = te["tanim"].isin(soguk)
    n_te = len(te)
    n_soguk = int(te["soguk"].sum())
    print(f"test satir {n_te:,} | soguk satir {n_soguk:,} (%{100 * n_soguk / n_te:.3f})")

    # ---- 1. SOGUK ILK GUN DAGILIMI
    print("\n" + "=" * 92)
    print("1. TEST SOGUK TRAFOLARIN ILK GUN DAGILIMI")
    print("=" * 92)
    ts = te[te["soguk"]].copy()
    ilk = ts.groupby("tanim", observed=True)["tarih"].min()
    say = ilk.value_counts().sort_index()
    satir = ts.groupby("tanim", observed=True).size()
    df = pd.DataFrame({"ilk": ilk, "n_satir": satir})
    grup = df.groupby("ilk").agg(n_trafo=("n_satir", "size"), n_satir=("n_satir", "sum"))
    grup["satir_pay"] = grup["n_satir"] / n_soguk
    grup["test_pay"] = grup["n_satir"] / n_te
    buyuk = grup[grup["n_trafo"] >= 50].sort_values("n_satir", ascending=False)
    print(f"  farkli ilk-gun sayisi: {len(grup)}")
    print(f"\n  {'ilk_gun':<12} {'n_trafo':>8} {'n_satir':>9} {'soguk_pay':>10} {'test_pay':>9}")
    for t, r in buyuk.iterrows():
        print(
            f"  {t.date()!s:<12} {int(r.n_trafo):>8,} {int(r.n_satir):>9,} "
            f"{r.satir_pay:>10.4f} {r.test_pay:>9.4f}"
        )
    print(
        f"  --- 50+ kohortlar toplami: n_trafo {int(buyuk.n_trafo.sum()):,} "
        f"n_satir {int(buyuk.n_satir.sum()):,} soguk_pay {buyuk.satir_pay.sum():.4f}"
    )
    kucuk = grup[grup["n_trafo"] < 50]
    print(
        f"  --- <50 kalan: {len(kucuk)} gun, n_trafo {int(kucuk.n_trafo.sum()):,} "
        f"n_satir {int(kucuk.n_satir.sum()):,} soguk_pay {kucuk.satir_pay.sum():.4f}"
    )

    # ---- 2. TRAIN TOPLU KATILIM GUNLERI
    print("\n" + "=" * 92)
    print("2. TRAIN TOPLU KATILIM GUNLERI (ayni gun >=50 trafo dogumu)")
    print("=" * 92)
    tilk = tr.groupby("tanim", observed=True)["tarih"].min()
    tsay = tilk.value_counts().sort_index()
    tbuyuk = tsay[tsay >= 50]
    print(f"  {'dogum_gun':<12} {'n_trafo':>8}   {'blok'}")
    for t, n in tbuyuk.items():
        if t == TR0:
            blok = "PANEL BASI (kesme eseri)"
        elif pd.Timestamp("2025-04-01") <= t <= pd.Timestamp("2025-07-31"):
            blok = "yaz25 IKIZ"
        elif pd.Timestamp("2025-08-01") <= t <= pd.Timestamp("2025-11-30"):
            blok = "guz25"
        else:
            blok = "kis26"
        print(f"  {t.date()!s:<12} {int(n):>8,}   {blok}")
    print(f"  toplam {len(tbuyuk)} gun, {int(tbuyuk.sum()):,} trafo")

    # tam dagilim: parti buyuklugu histogrami
    print("\n  parti buyuklugu dagilimi (2025-01-02 sonrasi dogumlar):")
    tsay2 = tsay[tsay.index > TR0]
    for lo, hi in [(1, 1), (2, 9), (10, 19), (20, 49), (50, 99), (100, 10**9)]:
        m = (tsay2 >= lo) & (tsay2 <= hi)
        print(
            f"    parti {lo}-{hi if hi < 10**8 else '+'}: {int(m.sum()):>4} gun, "
            f"{int(tsay2[m].sum()):>6,} trafo"
        )

    # ---- 3. 05-11 KOHORTU ICINDE RAMPA / KISMI GUN
    print("\n" + "=" * 92)
    print("3. 2026-05-11 KOHORTU -- test icindeki yapisi")
    print("=" * 92)
    K = pd.Timestamp("2026-05-11")
    koh = set(ilk[ilk == K].index)
    print(f"  kohort trafo {len(koh):,}")
    tk = ts[ts["tanim"].isin(koh)]
    print(
        f"  kohort satir {len(tk):,} (soguk payi %{100 * len(tk) / n_soguk:.2f}, "
        f"test payi %{100 * len(tk) / n_te:.2f})"
    )
    gs = tk.groupby("tanim", observed=True)["tarih"].agg(["min", "max", "size"])
    print(f"  son gun dagilimi: {gs['max'].value_counts().head(5).to_dict()}")
    bekl = (TE1 - K).days + 1
    print(
        f"  11 May..31 Tem gun sayisi = {bekl}; satir/trafo medyan {gs['size'].median():.0f} "
        f"ort {gs['size'].mean():.2f} min {gs['size'].min()} max {gs['size'].max()}"
    )
    tam = int((gs["size"] == bekl).sum())
    print(f"  tam dolu (={bekl} satir) trafo: {tam:,} (%{100 * tam / len(gs):.1f})")

    # kohortun kVA dagilimi vs diger soguk vs sicak
    print("\n  kVA (guc) karsilastirmasi:")
    for ad, alt in [
        ("05-11 kohortu", te[te["tanim"].isin(koh)]),
        ("diger soguk", te[te["soguk"] & ~te["tanim"].isin(koh)]),
        ("sicak", te[~te["soguk"]]),
    ]:
        g = alt.drop_duplicates("tanim")["guc"]
        print(
            f"    {ad:<16} n_trafo {len(g):>6,}  medyan {g.median():>8.1f}  "
            f"ort {g.mean():>8.1f}  log1p ort {np.log1p(g).mean():.4f}"
        )

    # ---- 4. TEST SOGUK KOHORT TABLOSU KAYDET
    out = KOK / "data/interim/h3_kohort"
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame({"tanim": ilk.index, "ilk_gun": ilk.values})
    meta = meta.merge(te.drop_duplicates("tanim")[["tanim", "guc", "lokasyon"]], on="tanim")
    meta["n_satir"] = meta["tanim"].map(satir).astype(int)
    meta.to_parquet(out / "test_soguk_kohort.parquet", index=False)
    print(f"\n  yazildi: {out / 'test_soguk_kohort.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
