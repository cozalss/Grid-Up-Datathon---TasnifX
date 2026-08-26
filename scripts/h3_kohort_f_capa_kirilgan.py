"""H3-F -- CAPANIN kVA EGIMI TEK KOHORTTAN MI GELIYOR?

H3-E gosterdi ki 05-11 kohort farkinin yarisi capanin kVA EGIMINDEN geliyor
(yaz25 +0,3460, guz25 +0,2017) -- ama YERLESIK trafolarda gercek egim
+0,0342 +- 0,0375, yani SIFIR. Bu celiski iki sekilde cozulebilir:

  (A) GERCEK MEKANIZMA: yuksek kVA'li yenidogan = olgun sahanin geriye dolgusu,
      dusuk kVA'li yenidogan = gercek yeni baglanti (sifirdan rampa).
      Bu testte de gecerli olur.
  (B) ESER: egim, yaz25'teki TEK toplu kohorttan (2025-07-28, lg 6,318)
      kestiriliyor. O kohort atilinca egim cokerse mekanizma yok.

Ayrica: toplu kohortlar atilinca kohort farki Delta ne oluyor?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
SUB = KOK / "submissions/tuketim_v67_c1335_olay.csv"
DOLULUK = 0.90
ORTAK0, ORTAK1 = pd.Timestamp("2026-05-11"), pd.Timestamp("2026-07-31")
KOHORT = pd.Timestamp("2026-05-11")
P_SOGUK = 0.22159


def sh(x) -> float:
    x = np.asarray(x, dtype="float64")
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    s2 = float(r @ r) / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * s2)
    return b, se


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
    )
    te = pd.read_csv(
        KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
    )
    te = te.merge(pd.read_csv(SUB, encoding="utf-8"), on="id", validate="one_to_one")
    tr["r"] = np.log1p(tr["tuketim"].clip(lower=0.0)) - np.log1p(tr["guc"])
    te["rhat"] = np.log1p(te["tuketim"].clip(lower=0.0)) - np.log1p(te["guc"])
    ngun = tr["tarih"].nunique()
    say = tr.groupby("tanim", observed=True)["tarih"].nunique()
    panel = set(say[say >= DOLULUK * ngun].index) & set(te["tanim"].unique())
    p = tr[tr["tanim"].isin(panel)]
    fe = p.groupby("tanim", observed=True)["r"].mean()
    g_tr = (
        pd.Series((p["r"] - p["tanim"].map(fe).to_numpy()).to_numpy(), index=p["tarih"].to_numpy())
        .groupby(level=0)
        .mean()
    )
    q = te[te["tanim"].isin(panel)]
    g_te = (
        pd.Series(
            (q["rhat"] - q["tanim"].map(fe).to_numpy()).to_numpy(), index=q["tarih"].to_numpy()
        )
        .groupby(level=0)
        .mean()
    )
    tr["d"] = tr["r"] - tr["tarih"].map(g_tr).to_numpy()
    te["d"] = te["rhat"] - te["tarih"].map(g_te).to_numpy()
    tr["lg"] = np.log1p(tr["guc"].to_numpy("float64"))
    te["lg"] = np.log1p(te["guc"].to_numpy("float64"))
    ilk_tr = tr.groupby("tanim", observed=True)["tarih"].min()
    parti_tr = ilk_tr.map(ilk_tr.value_counts())
    tr["ilk_gun"] = tr["tanim"].map(ilk_tr).to_numpy()
    tr["yas"] = (tr["tarih"] - tr["ilk_gun"]).dt.days
    tr["parti"] = tr["tanim"].map(parti_tr).to_numpy()

    te["soguk"] = ~te["tanim"].isin(set(tr["tanim"].unique()))
    ilk_te = te[te["soguk"]].groupby("tanim", observed=True)["tarih"].min()
    ts = te[te["soguk"] & (te["tarih"] >= ORTAK0) & (te["tarih"] <= ORTAK1)].copy()
    ts["ilk_gun"] = ts["tanim"].map(ilk_te).to_numpy()
    ts["yas"] = (ts["tarih"] - ts["ilk_gun"]).dt.days
    ts = ts[ts["yas"] >= 7]
    T = ts.groupby("tanim", observed=True).agg(
        msev=("d", "mean"), n=("d", "size"), lg=("lg", "first"), ilk_gun=("ilk_gun", "first")
    )
    T = T[T["n"] >= 14]
    T["a"] = T["ilk_gun"] == KOHORT

    def capa_tablo(b0, b1, toplu_at=False):
        a = tr[
            (tr["ilk_gun"] >= pd.Timestamp(b0))
            & (tr["ilk_gun"] <= pd.Timestamp(b1))
            & (tr["yas"] >= 7)
        ]
        if toplu_at:
            a = a[a["parti"] < 100]
        A = a.groupby("tanim", observed=True).agg(
            sev=("d", "mean"), n=("d", "size"), lg=("lg", "first"), parti=("parti", "first")
        )
        return A[A["n"] >= 14]

    print("=" * 96)
    print("CAPA kVA EGIMI -- toplu kohortlar ATILINCA")
    print("=" * 96)
    print(f"  {'capa':<28} {'n':>7} {'egim':>10} {'SH':>8} {'t':>7}")
    egimler = {}
    for blok, (b0, b1) in [
        ("yaz25", ("2025-04-01", "2025-07-31")),
        ("guz25", ("2025-08-01", "2025-11-30")),
        ("kis26", ("2025-12-01", "2026-03-31")),
    ]:
        for at in (False, True):
            A = capa_tablo(b0, b1, at)
            if len(A) < 40:
                print(f"  {blok + (' toplusuz' if at else ' hepsi'):<28} n={len(A)} yetersiz")
                continue
            b, s = ols(np.column_stack([np.ones(len(A)), A["lg"].to_numpy()]), A["sev"].to_numpy())
            ad = blok + (" TOPLUSUZ" if at else " hepsi")
            egimler[ad] = b
            print(f"  {ad:<28} {len(A):>7,} {b[1]:>+10.4f} {s[1]:>8.4f} {b[1] / s[1]:>+7.2f}")

    # yerlesik referans
    sicak = set(tr["tanim"].unique()) & set(te["tanim"].unique())
    m = tr[
        tr["tanim"].isin(sicak)
        & (tr["tarih"] >= pd.Timestamp("2025-04-01"))
        & (tr["tarih"] <= pd.Timestamp("2025-07-31"))
    ]
    S = m.groupby("tanim", observed=True).agg(
        sev=("d", "mean"), n=("d", "size"), lg=("lg", "first")
    )
    S = S[S["n"] >= 30]
    b, s = ols(np.column_stack([np.ones(len(S)), S["lg"].to_numpy()]), S["sev"].to_numpy())
    print(
        f"  {'YERLESIK (sicak, yaz25)':<28} {len(S):>7,} {b[1]:>+10.4f} {s[1]:>8.4f} "
        f"{b[1] / s[1]:>+7.2f}"
    )

    print("\n" + "=" * 96)
    print("KOHORT FARKI Delta -- toplusuz capa ile (en dusuk varsayimli spesifikasyon)")
    print("=" * 96)
    print(f"  {'capa':<28} {'Delta':>9} {'SH':>8} {'t':>7} {'dMSE':>10}   kirpma K=0/5/25/50")
    w_a = float((te.loc[te["soguk"], "tanim"].map(ilk_te) == KOHORT).mean())
    w_d = 1.0 - w_a
    for blok, (b0, b1) in [
        ("yaz25", ("2025-04-01", "2025-07-31")),
        ("guz25", ("2025-08-01", "2025-11-30")),
        ("kis26", ("2025-12-01", "2026-03-31")),
    ]:
        for at in (False, True):
            A = capa_tablo(b0, b1, at)
            if len(A) < 40:
                continue
            bb, _ = ols(np.column_stack([np.ones(len(A)), A["lg"].to_numpy()]), A["sev"].to_numpy())
            yan = (bb[0] + bb[1] * T["lg"].to_numpy()) - T["msev"].to_numpy()
            ya, yd = yan[T["a"].to_numpy()], yan[~T["a"].to_numpy()]
            f = ya.mean() - yd.mean()
            ss = np.sqrt(sh(ya) ** 2 + sh(yd) ** 2)
            dm = -P_SOGUK * w_a * w_d * f * f
            kir = []
            for K in (0, 5, 25, 50):
                ia = np.argsort(-np.abs(ya - ya.mean()))[K:]
                idd = np.argsort(-np.abs(yd - yd.mean()))[K:]
                kir.append(f"{ya[ia].mean() - yd[idd].mean():+.4f}")
            ad = blok + (" TOPLUSUZ" if at else " hepsi")
            print(
                f"  {ad:<28} {f:>+9.4f} {ss:>8.4f} {f / ss:>+7.2f} {dm:>+10.5f}   " + " ".join(kir)
            )

    print("\n  NOT: Delta = (b_capa - b_model)*dlg + kohort_kuklasi.")
    print("       kohort_kuklasi = +0,0672 (model tarafi, capadan BAGIMSIZ, t=-4,91)")
    print("       geri kalan tamamen capanin kVA egimine bagli.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
