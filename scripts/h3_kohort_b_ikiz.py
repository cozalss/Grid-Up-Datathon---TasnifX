"""H3-B -- PARTI BUYUKLUGU SEVIYE ETKISI (train ikizleri).

SORU: 2026-05-11 kohortu bir TOPLU KATILIM (1.326 trafo). son_islem_olay.py
zaten gun-0'da parti buyuklugu ayrimini bulmustu (100+ dusus yok, <100 var).
H3 bunun gun-0 OTESINE gecip gecmedigini soruyor: toplu katilim trafolari
OLGUN seviyede mi dogar (geriye dolgu), yoksa kucuk parti trafolar gibi
rampa mi yapar?

Eger toplu katilim trafolari olgun seviyede dogar ve model onlara "yenidogan
rampasi" seviyesi verirse, b_soguk = +0,16'nin TAMAMI 05-11 kohortuna aittir
ve duzeltme kohorta OZGU olmalidir.

YONTEM (kural 6: gun etkisi cikarilir):
  r = log1p(tuketim) - log1p(guc)
  gun etkisi g_t: SABIT PANEL (train'in >=%90 gununde kaydi olan trafolar),
                  trafo FE cikarilmis gunluk ortalama
  seviye_i(a) = ort_t[ r_it - g_t ]  yas kovasi a icinde
  Karsilastirma: parti>=100 vs parti<100, kVA ve dogum ayi kontrollu.

BLOKLAR (kural 7/9 -- en az iki ortusmeyen kesme):
  yaz25 dogumlulari 2025-04-01..07-31   <- ZORUNLU (mevsimsel ikiz)
  guz25 dogumlulari 2025-08-01..11-30
  kis26 dogumlulari 2025-12-01..2026-03-31
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
TR0, TR1 = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-03-31")
DOLULUK = 0.90
BLOK = {
    "yaz25": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")),
    "guz25": (pd.Timestamp("2025-08-01"), pd.Timestamp("2025-11-30")),
    "kis26": (pd.Timestamp("2025-12-01"), pd.Timestamp("2026-03-31")),
}
YAS_KOVA = [(0, 0), (1, 6), (7, 20), (21, 40), (41, 81), (82, 200)]


def gun_etkisi(tr: pd.DataFrame) -> pd.Series:
    ngun = tr["tarih"].nunique()
    say = tr.groupby("tanim", observed=True)["tarih"].nunique()
    panel = say[say >= DOLULUK * ngun].index
    p = tr[tr["tanim"].isin(set(panel))].copy()
    fe = p.groupby("tanim", observed=True)["r"].mean()
    p["d"] = p["r"] - p["tanim"].map(fe).to_numpy()
    g = p.groupby("tarih")["d"].mean()
    print(
        f"  sabit panel {len(panel):,} trafo | gun etkisi std {g.std():.4f} "
        f"aralik [{g.min():+.4f}, {g.max():+.4f}]"
    )
    return g


def sh(x: np.ndarray) -> float:
    x = np.asarray(x, dtype="float64")
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
    )
    tr["r"] = np.log1p(tr["tuketim"].clip(lower=0.0)) - np.log1p(tr["guc"])
    print("GUN ETKISI")
    g = gun_etkisi(tr)
    tr["d"] = tr["r"] - tr["tarih"].map(g).to_numpy()

    ilk = tr.groupby("tanim", observed=True)["tarih"].min()
    parti = ilk.map(ilk.value_counts())
    tr["ilk_gun"] = tr["tanim"].map(ilk).to_numpy()
    tr["parti"] = tr["tanim"].map(parti).to_numpy()
    tr["yas"] = (tr["tarih"] - tr["ilk_gun"]).dt.days
    tr["lg"] = np.log1p(tr["guc"].to_numpy("float64"))

    # 2025-01-01 dogumlulari panel basi eseri -- ATILIR
    yeni = tr[tr["ilk_gun"] > TR0].copy()
    print(
        f"\n2025-01-01 sonrasi dogan trafo satiri {len(yeni):,} ({yeni['tanim'].nunique():,} trafo)"
    )

    print("\n" + "=" * 100)
    print("A. YAS x PARTI SEVIYE TABLOSU (gun etkisi cikarilmis, trafo-ortalamasi degil ham)")
    print("=" * 100)
    for ad, (b0, b1) in BLOK.items():
        alt = yeni[(yeni["ilk_gun"] >= b0) & (yeni["ilk_gun"] <= b1)]
        nt = alt["tanim"].nunique()
        print(f"\n  --- {ad} dogumlulari: {nt:,} trafo, {len(alt):,} satir")
        for lo, hi in YAS_KOVA:
            m = alt[(alt["yas"] >= lo) & (alt["yas"] <= hi)]
            if len(m) == 0:
                continue
            buyuk = m[m["parti"] >= 100]["d"].to_numpy()
            kucuk = m[m["parti"] < 100]["d"].to_numpy()
            s1 = f"{buyuk.mean():+.4f}+-{sh(buyuk):.4f} (n={len(buyuk):,})" if len(buyuk) else "--"
            s2 = f"{kucuk.mean():+.4f}+-{sh(kucuk):.4f} (n={len(kucuk):,})" if len(kucuk) else "--"
            fark = (buyuk.mean() - kucuk.mean()) if (len(buyuk) and len(kucuk)) else np.nan
            print(
                f"    yas {lo:>3}-{hi:<3}  parti100+ {s1:<32}  parti<100 {s2:<32}  fark {fark:+.4f}"
                if np.isfinite(fark)
                else f"    yas {lo:>3}-{hi:<3}  parti100+ {s1:<32}  parti<100 {s2:<32}"
            )

    # ---- B. KVA + AY KONTROLLU: her blok icin yas>=7 sabit-durum seviyesi
    print("\n" + "=" * 100)
    print("B. SABIT DURUM SEVIYESI (yas>=7), TRAFO BAZINDA -- kVA/ay kontrollu OLS")
    print("=" * 100)
    print("  model: seviye_i ~ 1 + toplu_i + log1p(guc)_i + dogum_ayi kuklalari")
    for ad, (b0, b1) in BLOK.items():
        alt = yeni[(yeni["ilk_gun"] >= b0) & (yeni["ilk_gun"] <= b1) & (yeni["yas"] >= 7)]
        tb = alt.groupby("tanim", observed=True).agg(
            sev=("d", "mean"),
            n=("d", "size"),
            lg=("lg", "first"),
            parti=("parti", "first"),
            ilk_gun=("ilk_gun", "first"),
        )
        tb = tb[tb["n"] >= 14]
        tb["toplu"] = (tb["parti"] >= 100).astype(float)
        if tb["toplu"].nunique() < 2:
            print(
                f"\n  --- {ad}: toplu/kucuk ayrimi YOK (toplu n={int(tb['toplu'].sum())}) "
                f"-- olculemez"
            )
            continue
        ay = pd.get_dummies(tb["ilk_gun"].dt.to_period("M").astype(str), drop_first=True)
        X = np.column_stack(
            [
                np.ones(len(tb)),
                tb["toplu"].to_numpy(),
                tb["lg"].to_numpy(),
                ay.to_numpy(dtype="float64"),
            ]
        )
        y = tb["sev"].to_numpy("float64")
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        dof = len(y) - X.shape[1]
        s2 = float(res @ res) / dof
        XtXi = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(XtXi) * s2)
        print(
            f"\n  --- {ad}: {len(tb):,} trafo (toplu {int(tb['toplu'].sum()):,} / "
            f"kucuk {int((1 - tb['toplu']).sum()):,})"
        )
        print(
            f"      ham ortalama    toplu {y[tb['toplu'] == 1].mean():+.4f}  "
            f"kucuk {y[tb['toplu'] == 0].mean():+.4f}  "
            f"HAM FARK {y[tb['toplu'] == 1].mean() - y[tb['toplu'] == 0].mean():+.4f}"
        )
        print(
            f"      TOPLU katsayisi {beta[1]:+.4f} +- {se[1]:.4f}  "
            f"t={beta[1] / se[1]:+.2f}   (kVA katsayisi {beta[2]:+.4f})"
        )

    # ---- C. 2025-07-28 KOHORTU (yaz25 icindeki tek 100+ parti) yakin plan
    print("\n" + "=" * 100)
    print("C. TOPLU KATILIM KOHORTLARI TEK TEK (yas>=7 sabit durum, trafo bazinda)")
    print("=" * 100)
    tb = (
        yeni[yeni["yas"] >= 7]
        .groupby("tanim", observed=True)
        .agg(
            sev=("d", "mean"),
            n=("d", "size"),
            lg=("lg", "first"),
            parti=("parti", "first"),
            ilk_gun=("ilk_gun", "first"),
        )
    )
    tb = tb[tb["n"] >= 14]
    # kVA-esitlenmis referans: ayni ay + benzer kVA'daki KUCUK partililer
    for k in sorted(tb.loc[tb["parti"] >= 50, "ilk_gun"].unique()):
        k = pd.Timestamp(k)
        koh = tb[tb["ilk_gun"] == k]
        ay = k.to_period("M")
        ref = tb[(tb["ilk_gun"].dt.to_period("M") == ay) & (tb["parti"] < 50)]
        # +-3 ay penceresinde kucuk partililer (ay referansi seyrekse)
        ref2 = tb[
            (tb["ilk_gun"] >= k - pd.Timedelta(days=45))
            & (tb["ilk_gun"] <= k + pd.Timedelta(days=45))
            & (tb["parti"] < 50)
        ]
        print(f"\n  kohort {k.date()} (n={len(koh):,}, gun sayisi ort {koh['n'].mean():.0f})")
        print(
            f"    seviye {koh['sev'].mean():+.4f} +- {sh(koh['sev'].to_numpy()):.4f}  "
            f"log1p(guc) ort {koh['lg'].mean():.3f}"
        )
        for rad, r in [("ayni ay kucuk", ref), ("+-45g kucuk", ref2)]:
            if len(r) < 5:
                print(f"    {rad}: n={len(r)} -- yetersiz")
                continue
            print(
                f"    {rad}: n={len(r):,} seviye {r['sev'].mean():+.4f} "
                f"+- {sh(r['sev'].to_numpy()):.4f}  log1p(guc) ort {r['lg'].mean():.3f}  "
                f"FARK {koh['sev'].mean() - r['sev'].mean():+.4f}"
            )

    tb.to_parquet(KOK / "data/interim/h3_kohort/train_yeni_seviye.parquet")
    print(f"\n  yazildi: data/interim/h3_kohort/train_yeni_seviye.parquet ({len(tb):,} trafo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
