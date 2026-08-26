"""H3-G -- YAS ESLESTIRMESI: capa egimi test ufkunda da ayakta mi?

TESPIT: capanin kVA egimi bloklar arasi MONOTON coküyor
    yaz25 +0,3460   guz25 +0,2017   kis26 +0,0319   yerlesik +0,0342
ve bu tam olarak GOZLEM PENCERESI ile birlikte kisaliyor (yaz25 dogumlulari
365 gune kadar izleniyor, kis26 dogumlulari 120 gune kadar).

TEST SOGUK 05-11 KOHORTU YALNIZCA 7..81 GUN YASINDA GOZLENIYOR.
Dolayisiyla capa AYNI YAS PENCERESINDE kurulmalidir (kural 8'in ruhu:
pencere hedefin geometrisiyle eslesmeli). Aksi halde olculen sey kVA degil
YASLANMA'dir ve teste TASINMAZ.

Bu betik capayi yas 7..81'e kirpip her seyi yeniden olcer.
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
YAS_LO, YAS_HI = 7, 81


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
    tr["ilk_gun"] = tr["tanim"].map(ilk_tr).to_numpy()
    tr["yas"] = (tr["tarih"] - tr["ilk_gun"]).dt.days
    tr["parti"] = tr["tanim"].map(ilk_tr.map(ilk_tr.value_counts())).to_numpy()

    te["soguk"] = ~te["tanim"].isin(set(tr["tanim"].unique()))
    ilk_te = te[te["soguk"]].groupby("tanim", observed=True)["tarih"].min()
    ts = te[te["soguk"] & (te["tarih"] >= ORTAK0) & (te["tarih"] <= ORTAK1)].copy()
    ts["ilk_gun"] = ts["tanim"].map(ilk_te).to_numpy()
    ts["yas"] = (ts["tarih"] - ts["ilk_gun"]).dt.days
    ts = ts[(ts["yas"] >= YAS_LO) & (ts["yas"] <= YAS_HI)]
    T = ts.groupby("tanim", observed=True).agg(
        msev=("d", "mean"), n=("d", "size"), lg=("lg", "first"), ilk_gun=("ilk_gun", "first")
    )
    T = T[T["n"] >= 14]
    T["a"] = T["ilk_gun"] == KOHORT
    print(
        f"TEST soguk (yas {YAS_LO}..{YAS_HI}): {len(T):,} trafo, "
        f"05-11 {int(T['a'].sum()):,} / diger {int((~T['a']).sum()):,}"
    )
    bm, sm = ols(
        np.column_stack([np.ones(len(T)), T["lg"].to_numpy(), T["a"].to_numpy(dtype=float)]),
        T["msev"].to_numpy(),
    )
    print(
        f"MODEL soguk egim {bm[1]:+.4f} +- {sm[1]:.4f} | "
        f"kohort kuklasi {bm[2]:+.4f} +- {sm[2]:.4f} t={bm[2] / sm[2]:+.2f}"
    )

    BLOK = [
        ("yaz25", "2025-04-01", "2025-07-31"),
        ("guz25", "2025-08-01", "2025-11-30"),
        ("kis26", "2025-12-01", "2026-03-31"),
    ]

    print("\n" + "=" * 100)
    print(f"CAPA kVA EGIMI -- YAS PENCERESI ESLESTIRILMIS ({YAS_LO}..{YAS_HI}) vs ESLESTIRILMEMIS")
    print("=" * 100)
    print(f"  {'blok':<10} {'pencere':<14} {'n':>7} {'egim':>10} {'SH':>8} {'t':>7}")
    capalar = {}
    for ad, b0, b1 in BLOK:
        for pad, (lo, hi) in [
            ("yas>=7 (tum)", (7, 10**6)),
            (f"yas {YAS_LO}-{YAS_HI}", (YAS_LO, YAS_HI)),
        ]:
            a = tr[
                (tr["ilk_gun"] >= pd.Timestamp(b0))
                & (tr["ilk_gun"] <= pd.Timestamp(b1))
                & (tr["yas"] >= lo)
                & (tr["yas"] <= hi)
            ]
            A = a.groupby("tanim", observed=True).agg(
                sev=("d", "mean"), n=("d", "size"), lg=("lg", "first")
            )
            A = A[A["n"] >= 14]
            if len(A) < 40:
                continue
            b, s = ols(np.column_stack([np.ones(len(A)), A["lg"].to_numpy()]), A["sev"].to_numpy())
            capalar[(ad, pad)] = b
            print(
                f"  {ad:<10} {pad:<14} {len(A):>7,} {b[1]:>+10.4f} {s[1]:>8.4f} "
                f"{b[1] / s[1]:>+7.2f}"
            )

    print("\n" + "=" * 100)
    print("KOHORT FARKI Delta -- yas eslestirilmis capa ile")
    print("=" * 100)
    w_a = float((te.loc[te["soguk"], "tanim"].map(ilk_te) == KOHORT).mean())
    w_d = 1.0 - w_a
    print(f"  {'capa':<28} {'Delta':>9} {'SH':>8} {'t':>7} {'dMSE':>10}   kirpma K=0/1/5/10/25/50")
    for (ad, pad), b in capalar.items():
        yan = (b[0] + b[1] * T["lg"].to_numpy()) - T["msev"].to_numpy()
        ya, yd = yan[T["a"].to_numpy()], yan[~T["a"].to_numpy()]
        f = ya.mean() - yd.mean()
        ss = np.sqrt(sh(ya) ** 2 + sh(yd) ** 2)
        dm = -P_SOGUK * w_a * w_d * f * f
        kir = []
        for K in (0, 1, 5, 10, 25, 50):
            ia = np.argsort(-np.abs(ya - ya.mean()))[K:]
            idd = np.argsort(-np.abs(yd - yd.mean()))[K:]
            kir.append(f"{ya[ia].mean() - yd[idd].mean():+.4f}")
        print(
            f"  {ad + ' / ' + pad:<28} {f:>+9.4f} {ss:>8.4f} {f / ss:>+7.2f} {dm:>+10.5f}   "
            + " ".join(kir)
        )

    # sadece kohort kuklasi kalirsa
    dk = -P_SOGUK * w_a * w_d * bm[2] ** 2
    print(f"\n  YALNIZ kohort kuklasi (kVA terimi sifir kabul): Delta {-bm[2]:+.4f} dMSE {dk:+.6f}")

    # ---- capa egiminin YAS ile seyri (mekanizma teshisi)
    print("\n" + "=" * 100)
    print("MEKANIZMA -- capa kVA egiminin YAS ile seyri (yaz25 + guz25 dogumlulari)")
    print("=" * 100)
    a = tr[
        (tr["ilk_gun"] >= pd.Timestamp("2025-04-01"))
        & (tr["ilk_gun"] <= pd.Timestamp("2025-11-30"))
    ]
    print(f"  {'yas kovasi':<14} {'n_trafo':>8} {'egim':>10} {'SH':>8} {'t':>7}")
    for lo, hi in [(7, 20), (21, 40), (41, 81), (82, 150), (151, 250), (251, 400)]:
        m = a[(a["yas"] >= lo) & (a["yas"] <= hi)]
        A = m.groupby("tanim", observed=True).agg(
            sev=("d", "mean"), n=("d", "size"), lg=("lg", "first")
        )
        A = A[A["n"] >= 10]
        if len(A) < 40:
            print(f"  {f'{lo}-{hi}':<14} n={len(A)} yetersiz")
            continue
        b, s = ols(np.column_stack([np.ones(len(A)), A["lg"].to_numpy()]), A["sev"].to_numpy())
        print(f"  {f'{lo}-{hi}':<14} {len(A):>8,} {b[1]:>+10.4f} {s[1]:>8.4f} {b[1] / s[1]:>+7.2f}")
    print("  -> egim yasla BUYUYORSA capanin uzun pencereli hali test ufkuna TASINMAZ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
