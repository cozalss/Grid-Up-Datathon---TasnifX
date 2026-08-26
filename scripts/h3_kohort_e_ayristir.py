"""H3-E -- KOHORT FARKINI BILESENLERINE AYIR + RAMPA DENETIMI.

AYRISTIRMA
----------
  yanlilik_i = (a + b_capa*lg_i) - msev_i
  msev = b_model*lg + kohort_kuklasi + artik   (TEST SOGUK icinde)

  => yanlilik_a - yanlilik_d = (b_capa - b_model) * (lg_a - lg_d)  -  kukla

  Ilk terim: kVA EGIM UYUSMAZLIGI (kohorta ozgu DEGIL -- kohort sadece yuksek
             kVA'li oldugu icin en cok o etkileniyor). YASAK BOLGE'ye komsu.
  Ikinci terim: KOHORTA OZGU seviye (ayni kVA'da model 05-11'e farkli sey mi
             veriyor?). H3'un gercek sorusu BU.

RAMPA DENETIMI (H3 adim 6)
--------------------------
  05-11 kohortunun ufku 11 Mayis'ta basliyor. yas 0..6'da model ne veriyor,
  train'deki toplu kohortlar ne yapiyor?

EGIM CAPRAZ DENETIMI
--------------------
  Model SICAK tarafta kVA egimini dogru veriyorsa ama SOGUK tarafta
  duzlestiriyorsa, bu gercek bir eksiklik. Ikisi de duzse capa yanlis.
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

    # ---------- TEST SOGUK TABLOSU
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
    T["a"] = (T["ilk_gun"] == KOHORT).astype(float)

    lg_a = float(T.loc[T.a == 1, "lg"].mean())
    lg_d = float(T.loc[T.a == 0, "lg"].mean())
    dlg = lg_a - lg_d
    print("=" * 96)
    print("AYRISTIRMA")
    print("=" * 96)
    print(f"  lg(05-11) {lg_a:.4f}  lg(diger) {lg_d:.4f}  DELTA_lg {dlg:+.4f}")

    X = np.column_stack([np.ones(len(T)), T["lg"].to_numpy(), T["a"].to_numpy()])
    bm, sm = ols(X, T["msev"].to_numpy())
    print(f"  MODEL soguk: msev = {bm[0]:+.4f} {bm[1]:+.4f}*lg {bm[2]:+.4f}*kohort05_11")
    print(
        f"               egim SH {sm[1]:.4f} | KOHORT KUKLASI SH {sm[2]:.4f} t={bm[2] / sm[2]:+.2f}"
    )

    for blok, (b0, b1) in [
        ("yaz25", ("2025-04-01", "2025-07-31")),
        ("guz25", ("2025-08-01", "2025-11-30")),
    ]:
        a = tr[
            (tr["ilk_gun"] >= pd.Timestamp(b0))
            & (tr["ilk_gun"] <= pd.Timestamp(b1))
            & (tr["yas"] >= 7)
        ]
        A = a.groupby("tanim", observed=True).agg(
            sev=("d", "mean"), n=("d", "size"), lg=("lg", "first")
        )
        A = A[A["n"] >= 14]
        ba, sa = ols(np.column_stack([np.ones(len(A)), A["lg"].to_numpy()]), A["sev"].to_numpy())
        egim_terim = (ba[1] - bm[1]) * dlg
        kohort_terim = -bm[2]
        toplam = egim_terim + kohort_terim
        print(f"\n  --- capa {blok}: b_capa {ba[1]:+.4f} (SH {sa[1]:.4f})")
        print(
            f"      kVA EGIM UYUSMAZLIGI terimi (b_capa-b_model)*dlg = "
            f"({ba[1]:+.4f} - {bm[1]:+.4f}) * {dlg:+.4f} = {egim_terim:+.4f}"
        )
        print(f"      KOHORTA OZGU terim (-kukla)                      = {kohort_terim:+.4f}")
        print(f"      TOPLAM kohort farki                              = {toplam:+.4f}")
        print(f"      kohorta ozgu PAY = %{100 * abs(kohort_terim) / abs(toplam):.1f}")

    print("\n  YORUM: kohorta ozgu terim MODEL kuklasidir ve capadan BAGIMSIZDIR;")
    print("         kVA terimi capanin egimine bagli ve capa spesifikasyonu ile oynuyor.")

    # ---------- SICAK/SOGUK EGIM CAPRAZ DENETIMI
    print("\n" + "=" * 96)
    print("EGIM CAPRAZ DENETIMI -- model kVA egimini SICAK tarafta veriyor mu?")
    print("=" * 96)
    # train'de SICAK trafolar icin gercek seviye (2025-04-01..07-31 penceresinde)
    sicak_tanim = set(tr["tanim"].unique()) & set(te["tanim"].unique())
    m = tr[
        tr["tanim"].isin(sicak_tanim)
        & (tr["tarih"] >= pd.Timestamp("2025-04-01"))
        & (tr["tarih"] <= pd.Timestamp("2025-07-31"))
    ]
    S = m.groupby("tanim", observed=True).agg(
        sev=("d", "mean"), n=("d", "size"), lg=("lg", "first")
    )
    S = S[S["n"] >= 30]
    bs, ss_ = ols(np.column_stack([np.ones(len(S)), S["lg"].to_numpy()]), S["sev"].to_numpy())
    print(f"  GERCEK sicak (train yaz25, n={len(S):,}):  egim {bs[1]:+.4f} +- {ss_[1]:.4f}")
    ms = te[~te["soguk"] & (te["tarih"] >= ORTAK0) & (te["tarih"] <= ORTAK1)]
    MS = ms.groupby("tanim", observed=True).agg(
        msev=("d", "mean"), n=("d", "size"), lg=("lg", "first")
    )
    MS = MS[MS["n"] >= 30]
    bms, sms = ols(np.column_stack([np.ones(len(MS)), MS["lg"].to_numpy()]), MS["msev"].to_numpy())
    print(f"  MODEL  sicak (test,     n={len(MS):,}):  egim {bms[1]:+.4f} +- {sms[1]:.4f}")
    print(f"  MODEL  soguk (test,     n={len(T):,}):  egim {bm[1]:+.4f} +- {sm[1]:.4f}")
    print(f"\n  sicak tarafta model egimi gercege {abs(bms[1] - bs[1]):.4f} uzakta;")
    print(f"  soguk tarafta capaya {abs(bm[1] - 0.346):.4f} uzakta (yaz25 capasi).")
    print("  -> ikisi de duzse capanin egimi kompozisyon eseri olabilir.")

    # ---------- RAMPA DENETIMI
    print("\n" + "=" * 96)
    print("RAMPA DENETIMI (H3 adim 6) -- 05-11 kohortunun ilk gunleri")
    print("=" * 96)
    tsr = te[te["soguk"]].copy()
    tsr["ilk_gun"] = tsr["tanim"].map(ilk_te).to_numpy()
    tsr["yas"] = (tsr["tarih"] - tsr["ilk_gun"]).dt.days
    ref = tsr[(tsr["ilk_gun"] == KOHORT)].groupby("tanim")["d"].mean()
    print(
        f"  {'yas':>6} {'MODEL 05-11':>14} {'n':>8}   {'TRAIN toplu(>=100)':>20} {'n':>7}"
        f"   {'TRAIN kucuk(<100)':>19} {'n':>7}"
    )
    trn = tr[tr["ilk_gun"] > pd.Timestamp("2025-01-01")]
    for lo, hi in [(0, 0), (1, 2), (3, 6), (7, 13), (14, 27), (28, 60), (61, 81)]:
        mm = tsr[(tsr["ilk_gun"] == KOHORT) & (tsr["yas"] >= lo) & (tsr["yas"] <= hi)]
        v = (mm["d"] - mm["tanim"].map(ref).to_numpy()).to_numpy()
        tb = trn[(trn["parti"] >= 100) & (trn["yas"] >= lo) & (trn["yas"] <= hi)]
        tk = trn[(trn["parti"] < 100) & (trn["yas"] >= lo) & (trn["yas"] <= hi)]
        rb = tb.groupby("tanim")["d"].transform("mean") if len(tb) else None
        s1 = f"{tb['d'].mean():+.4f}" if len(tb) else "--"
        s2 = f"{tk['d'].mean():+.4f}" if len(tk) else "--"
        print(
            f"  {lo:>2}-{hi:<3} {v.mean():>+14.4f} {len(v):>8,}   {s1:>20} {len(tb):>7,}"
            f"   {s2:>19} {len(tk):>7,}"
        )
    print("  (MODEL sutunu: kohortun kendi trafo ortalamasina gore SAPMA -- rampa gorunur)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
