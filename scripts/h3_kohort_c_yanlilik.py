"""H3-C -- KOHORT DUZEYINDE MODEL YANLILIGI: 05-11 kohortu farkli mi?

FIKIR
-----
b_soguk (~+0,16) tum soguk satirlara ait bir DUZ KAYMA olarak on kayitli.
H3 soruyor: bu kayma aslinda 2026-05-11 kohortuna mi ait?

Kohortlar arasi KARSILASTIRMA yaparsak, kuresel drift (panel yil kaymasi,
mevsim ekstrapolasyonu) HER kohortta ayni oldugu icin SADELESIR. Geriye
kohorta OZGU yanlilik kalir.

OLCUM
-----
  panel taban: train'in >=%90 gununde kaydi olan trafolar, trafo FE = ort r
  g_t (train) = ort_i (r_it - fe_i)         gercek gun ekseni
  g_t (test)  = ort_i (rhat_it - fe_i)      MODELIN ima ettigi gun ekseni
                (ayni panel trafolari, ayni FE -- sampiyon gonderiminden)

  seviye_i(test, model) = ort_t [ rhat_it - g_t(test) ]     panel tabanina gore
  seviye_i(train, gercek) = ort_t [ r_it - g_t(train) ]     ayni birim

  CAPA: yaz25 dogumlulari uzerinde  seviye ~ 1 + log1p(guc) + toplu + yas kovasi
  Her test soguk kohortuna capa uygulanir -> beklenen GERCEK seviye
  yanlilik(kohort) = capa_seviye - model_seviye

KURAL 6: gun etkisi iki tarafta da cikarildi.
KURAL 1: kirpma tablosu asagida.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
SUB = KOK / "submissions/tuketim_v67_c1335_olay.csv"
DOLULUK = 0.90
ORTAK0 = pd.Timestamp("2026-05-11")  # butun buyuk kohortlarin ORTAK penceresi
ORTAK1 = pd.Timestamp("2026-07-31")
KOHORT = pd.Timestamp("2026-05-11")


def sh(x) -> float:
    x = np.asarray(x, dtype="float64")
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
    )
    te = pd.read_csv(
        KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
    )
    sub = pd.read_csv(SUB, encoding="utf-8")
    ornek = pd.read_csv(KOK / "data/raw/sample_submission.csv", encoding="utf-8")
    if not sub["id"].equals(ornek["id"]):
        raise RuntimeError("id sirasi bozuk")
    te = te.merge(sub, on="id", validate="one_to_one")
    tr["r"] = np.log1p(tr["tuketim"].clip(lower=0.0)) - np.log1p(tr["guc"])
    te["rhat"] = np.log1p(te["tuketim"].clip(lower=0.0)) - np.log1p(te["guc"])

    # ---- PANEL TABANI + FE (tek kaynak: train)
    ngun = tr["tarih"].nunique()
    say = tr.groupby("tanim", observed=True)["tarih"].nunique()
    panel = set(say[say >= DOLULUK * ngun].index)
    ortak_panel = panel & set(te["tanim"].unique())
    p = tr[tr["tanim"].isin(ortak_panel)]
    fe = p.groupby("tanim", observed=True)["r"].mean()
    print(f"panel {len(panel):,} | testte de olan {len(ortak_panel):,}")

    g_tr = p["r"] - p["tanim"].map(fe).to_numpy()
    g_tr = pd.Series(g_tr.to_numpy(), index=p["tarih"].to_numpy()).groupby(level=0).mean()
    q = te[te["tanim"].isin(ortak_panel)]
    g_te = q["rhat"] - q["tanim"].map(fe).to_numpy()
    g_te = pd.Series(g_te.to_numpy(), index=q["tarih"].to_numpy()).groupby(level=0).mean()
    print(f"gun ekseni: train std {g_tr.std():.4f} | test(model ima) std {g_te.std():.4f}")

    tr["d"] = tr["r"] - tr["tarih"].map(g_tr).to_numpy()
    te["d"] = te["rhat"] - te["tarih"].map(g_te).to_numpy()

    # ---- CAPA: yaz25 dogumlulari (2025-04-01..07-31), yas>=7 sabit durum
    ilk_tr = tr.groupby("tanim", observed=True)["tarih"].min()
    parti_tr = ilk_tr.map(ilk_tr.value_counts())
    tr["ilk_gun"] = tr["tanim"].map(ilk_tr).to_numpy()
    tr["yas"] = (tr["tarih"] - tr["ilk_gun"]).dt.days
    tr["parti"] = tr["tanim"].map(parti_tr).to_numpy()
    tr["lg"] = np.log1p(tr["guc"].to_numpy("float64"))

    capa_alt = tr[
        (tr["ilk_gun"] >= pd.Timestamp("2025-04-01"))
        & (tr["ilk_gun"] <= pd.Timestamp("2025-07-31"))
        & (tr["yas"] >= 7)
    ]
    A = capa_alt.groupby("tanim", observed=True).agg(
        sev=("d", "mean"), n=("d", "size"), lg=("lg", "first"), parti=("parti", "first")
    )
    A = A[A["n"] >= 14]
    A["toplu"] = (A["parti"] >= 100).astype(float)
    X = np.column_stack([np.ones(len(A)), A["lg"].to_numpy(), A["toplu"].to_numpy()])
    y = A["sev"].to_numpy("float64")
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = float(res @ res) / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * s2)
    print(
        f"\nCAPA (yaz25 dogumlulari, n={len(A):,}): sev = {beta[0]:+.4f} "
        f"+ {beta[1]:+.4f}*log1p(guc) + {beta[2]:+.4f}*toplu"
    )
    print(f"  SH: {se[0]:.4f} / {se[1]:.4f} / {se[2]:.4f}   artik std {np.sqrt(s2):.4f}")

    # ---- TEST SOGUK KOHORTLARI, ORTAK PENCERE
    tr_tanim = set(tr["tanim"].unique())
    te["soguk"] = ~te["tanim"].isin(tr_tanim)
    ilk_te = te[te["soguk"]].groupby("tanim", observed=True)["tarih"].min()
    ts = te[te["soguk"] & (te["tarih"] >= ORTAK0) & (te["tarih"] <= ORTAK1)].copy()
    ts["ilk_gun"] = ts["tanim"].map(ilk_te).to_numpy()
    ts["yas"] = (ts["tarih"] - ts["ilk_gun"]).dt.days
    ts = ts[ts["yas"] >= 7]  # sabit durum -- capayla ayni tanim
    ts["lg"] = np.log1p(ts["guc"].to_numpy("float64"))
    T = ts.groupby("tanim", observed=True).agg(
        msev=("d", "mean"), n=("d", "size"), lg=("lg", "first"), ilk_gun=("ilk_gun", "first")
    )
    T = T[T["n"] >= 14]
    T["kohort"] = np.where(T["ilk_gun"] == KOHORT, "05-11", "diger")
    # kohort buyuklugu -> toplu bayragi (train ile ayni esik 100)
    kb = T["ilk_gun"].map(ilk_te.value_counts())
    T["toplu"] = (kb >= 100).astype(float)
    T["capa"] = beta[0] + beta[1] * T["lg"] + beta[2] * T["toplu"]
    T["yanlilik"] = T["capa"] - T["msev"]
    print(
        f"\nORTAK PENCERE {ORTAK0.date()}..{ORTAK1.date()}, yas>=7, >=14 gun: "
        f"{len(T):,} soguk trafo"
    )

    print("\n" + "=" * 96)
    print("KOHORT DUZEYI YANLILIK  (capa_gercek - model_ima)  -- kuresel drift ICINDE")
    print("=" * 96)
    print(
        f"  {'kohort':<10} {'n_trafo':>8} {'n_satir':>9} {'model_sev':>10} {'capa':>9} "
        f"{'YANLILIK':>10} {'SH':>8}"
    )
    ozet = {}
    for ad, alt in [
        ("05-11", T[T.kohort == "05-11"]),
        ("diger", T[T.kohort == "diger"]),
        ("TUMU", T),
    ]:
        b = alt["yanlilik"].to_numpy()
        ozet[ad] = (b.mean(), sh(b), len(alt), int(alt["n"].sum()))
        print(
            f"  {ad:<10} {len(alt):>8,} {int(alt['n'].sum()):>9,} "
            f"{alt['msev'].mean():>10.4f} {alt['capa'].mean():>9.4f} "
            f"{b.mean():>+10.4f} {sh(b):>8.4f}"
        )
    f = ozet["05-11"][0] - ozet["diger"][0]
    sf = np.sqrt(ozet["05-11"][1] ** 2 + ozet["diger"][1] ** 2)
    print(f"\n  FARK (05-11 eksi diger) = {f:+.4f} +- {sf:.4f}   t = {f / sf:+.2f}")
    print("  -> t buyukse kayma KOHORTA OZGU; kucukse kayma TUM soguga ait (duz).")

    # ---- kohortlar tek tek (50+ olanlar)
    print("\n  buyuk kohortlar tek tek:")
    for k in sorted(T["ilk_gun"].unique()):
        alt = T[T["ilk_gun"] == k]
        if len(alt) < 30:
            continue
        b = alt["yanlilik"].to_numpy()
        print(
            f"    {pd.Timestamp(k).date()}  n={len(alt):>5,}  lg={alt['lg'].mean():.3f}  "
            f"model_sev {alt['msev'].mean():+.4f}  capa {alt['capa'].mean():+.4f}  "
            f"yanlilik {b.mean():+.4f} +- {sh(b):.4f}"
        )

    T.to_parquet(KOK / "data/interim/h3_kohort/test_soguk_yanlilik.parquet")

    # ---- KIRPMA TABLOSU: kohort farkini en cok tasiyan trafolari at
    print("\n" + "=" * 96)
    print("KIRPMA TABLOSU -- 05-11 eksi diger farki, en buyuk |katki| K trafo atilarak")
    print("=" * 96)
    a = T[T.kohort == "05-11"]["yanlilik"].to_numpy()
    d = T[T.kohort == "diger"]["yanlilik"].to_numpy()
    print(f"  {'K':>4} {'fark':>10} {'SH':>8} {'t':>7}")
    for K in (0, 1, 5, 10, 25, 50):
        aa = np.sort(np.abs(a - a.mean()))
        idx_a = np.argsort(-np.abs(a - a.mean()))[K:] if len(a) > K else []
        idx_d = np.argsort(-np.abs(d - d.mean()))[K:] if len(d) > K else []
        a2, d2 = a[idx_a], d[idx_d]
        ff = a2.mean() - d2.mean()
        ss = np.sqrt(sh(a2) ** 2 + sh(d2) ** 2)
        print(f"  {K:>4} {ff:>+10.4f} {ss:>8.4f} {ff / ss:>+7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
