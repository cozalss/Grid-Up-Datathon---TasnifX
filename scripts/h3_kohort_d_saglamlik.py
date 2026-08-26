"""H3-D -- KOHORT FARKI GERCEK MI? Capa spesifikasyonuna duyarlilik.

H3-C'de 05-11 kohortu ile diger soguk arasinda yanlilik farki -0,1588 (t=-7,65)
cikti. AMA bu farkin TAMAMI capadaki ``toplu`` kuklasindan gelebilir ve o kukla
TRAIN'DE TEK BIR KOHORTTAN (2025-07-28, n=172) kestiriliyor. Ustelik ay
kuklalari eklenince isaret degistirmisti (+0,0667).

Bu betik farki bilesenlerine ayirir:
  (i)   capadan ``toplu`` cikarilirsa fark ne olur?  -> fark kVA karisimindan mi?
  (ii)  capa guz25 dogumlulariyla kurulursa isaret ayni mi? (kural 7/9)
  (iii) ``toplu`` katsayisi kohort-disi-birak ile ne kadar kararli?
  (iv)  MODELIN soguk kVA egimi capanin egimiyle ayni mi? (fark kVA egimi
        hatasiysa bu YASAK BOLGE'deki "soguk kVA kovasi" ekseni demektir)
  (v)   dMSE: kohort baglantisini ayirmak duz kaymaya gore ne kazandirir?
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
BLOK = {
    "yaz25": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")),
    "guz25": (pd.Timestamp("2025-08-01"), pd.Timestamp("2025-11-30")),
}


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

    ilk_tr = tr.groupby("tanim", observed=True)["tarih"].min()
    parti_tr = ilk_tr.map(ilk_tr.value_counts())
    tr["ilk_gun"] = tr["tanim"].map(ilk_tr).to_numpy()
    tr["yas"] = (tr["tarih"] - tr["ilk_gun"]).dt.days
    tr["parti"] = tr["tanim"].map(parti_tr).to_numpy()
    tr["lg"] = np.log1p(tr["guc"].to_numpy("float64"))

    def capa_tablo(b0, b1):
        a = tr[(tr["ilk_gun"] >= b0) & (tr["ilk_gun"] <= b1) & (tr["yas"] >= 7)]
        A = a.groupby("tanim", observed=True).agg(
            sev=("d", "mean"),
            n=("d", "size"),
            lg=("lg", "first"),
            parti=("parti", "first"),
            ilk_gun=("ilk_gun", "first"),
        )
        A = A[A["n"] >= 14]
        A["toplu"] = (A["parti"] >= 100).astype(float)
        return A

    # ---- TEST SOGUK
    te["soguk"] = ~te["tanim"].isin(set(tr["tanim"].unique()))
    ilk_te = te[te["soguk"]].groupby("tanim", observed=True)["tarih"].min()
    ts = te[te["soguk"] & (te["tarih"] >= ORTAK0) & (te["tarih"] <= ORTAK1)].copy()
    ts["ilk_gun"] = ts["tanim"].map(ilk_te).to_numpy()
    ts["yas"] = (ts["tarih"] - ts["ilk_gun"]).dt.days
    ts = ts[ts["yas"] >= 7]
    ts["lg"] = np.log1p(ts["guc"].to_numpy("float64"))
    T = ts.groupby("tanim", observed=True).agg(
        msev=("d", "mean"), n=("d", "size"), lg=("lg", "first"), ilk_gun=("ilk_gun", "first")
    )
    T = T[T["n"] >= 14]
    T["toplu"] = (T["ilk_gun"].map(ilk_te.value_counts()) >= 100).astype(float)
    T["a"] = T["ilk_gun"] == KOHORT

    print("=" * 96)
    print("(iv) EGIM KARSILASTIRMASI -- MODEL vs CAPA, soguk kVA egimi")
    print("=" * 96)
    bm, sm = ols(np.column_stack([np.ones(len(T)), T["lg"].to_numpy()]), T["msev"].to_numpy())
    print(f"  MODEL (test soguk):  msev = {bm[0]:+.4f} + {bm[1]:+.4f}*lg   SH egim {sm[1]:.4f}")
    for ad, (b0, b1) in BLOK.items():
        A = capa_tablo(b0, b1)
        ba, sa = ols(np.column_stack([np.ones(len(A)), A["lg"].to_numpy()]), A["sev"].to_numpy())
        print(
            f"  CAPA  ({ad}, n={len(A):,}): sev = {ba[0]:+.4f} + {ba[1]:+.4f}*lg  "
            f"SH egim {sa[1]:.4f}   EGIM FARKI {ba[1] - bm[1]:+.4f}"
        )
    print("  NOT: egim farki varsa kohort farki kVA KARISIMINDAN dogar --")
    print("       bu YASAK BOLGE'deki 'soguk kVA kovasi' ekseni (+0,005 zarar).")

    print("\n" + "=" * 96)
    print("(iii) ``toplu`` KATSAYISI -- kohort-disi-birak kararliligi (yas>=7)")
    print("=" * 96)
    A_all = capa_tablo(pd.Timestamp("2025-01-02"), pd.Timestamp("2026-03-31"))
    kohortlar = sorted(A_all.loc[A_all["toplu"] == 1, "ilk_gun"].unique())
    print(
        f"  train'de toplu(>=100) kohort sayisi: {len(kohortlar)} -> "
        f"{[str(pd.Timestamp(k).date()) for k in kohortlar]}"
    )
    print("  KURAL 3: soguk tarafta uc tohum/kohort yetmez.")
    for birak in [None] + kohortlar:
        A = A_all if birak is None else A_all[A_all["ilk_gun"] != birak]
        if A["toplu"].nunique() < 2:
            continue
        ay = pd.get_dummies(A["ilk_gun"].dt.to_period("M").astype(str), drop_first=True)
        X = np.column_stack(
            [
                np.ones(len(A)),
                A["lg"].to_numpy(),
                A["toplu"].to_numpy(),
                ay.to_numpy(dtype="float64"),
            ]
        )
        b, s = ols(X, A["sev"].to_numpy())
        ad = "hepsi" if birak is None else f"-{pd.Timestamp(birak).date()}"
        print(
            f"    {ad:<14} n={len(A):>5,} toplu={int(A['toplu'].sum()):>4}  "
            f"toplu_kats {b[2]:+.4f} +- {s[2]:.4f}  t={b[2] / s[2]:+.2f}"
        )

    print("\n" + "=" * 96)
    print("(i)+(ii) KOHORT FARKI, CAPA SPESIFIKASYONUNA GORE")
    print("=" * 96)
    print(f"  {'capa':<34} {'toplu_k':>9} {'fark(05-11 - diger)':>20} {'SH':>8} {'t':>7}")
    sonuc = {}
    for blok_ad, (b0, b1) in BLOK.items():
        A = capa_tablo(b0, b1)
        for spec in ("lg", "lg+toplu", "lg+toplu+ay"):
            cols = [np.ones(len(A)), A["lg"].to_numpy()]
            if "toplu" in spec:
                cols.append(A["toplu"].to_numpy())
            if "ay" in spec:
                ay = pd.get_dummies(A["ilk_gun"].dt.to_period("M").astype(str), drop_first=True)
                cols.append(ay.to_numpy(dtype="float64"))
            X = np.column_stack(cols)
            if np.linalg.matrix_rank(X) < X.shape[1]:
                continue
            b, s = ols(X, A["sev"].to_numpy())
            capa = b[0] + b[1] * T["lg"].to_numpy()
            tk = np.nan
            if "toplu" in spec:
                capa = capa + b[2] * T["toplu"].to_numpy()
                tk = b[2]
            yan = capa - T["msev"].to_numpy()
            ya, yd = yan[T["a"].to_numpy()], yan[~T["a"].to_numpy()]
            f = ya.mean() - yd.mean()
            ss = np.sqrt(sh(ya) ** 2 + sh(yd) ** 2)
            sonuc[(blok_ad, spec)] = (f, ss, ya, yd)
            tks = f"{tk:+.4f}" if np.isfinite(tk) else "  --"
            print(f"  {blok_ad + ' / ' + spec:<34} {tks:>9} {f:>+20.4f} {ss:>8.4f} {f / ss:>+7.2f}")

    print("\n  -> ``toplu`` capadan cikinca fark ne oluyor? Ust satirlara bak.")
    print("  -> yaz25 ve guz25 capalari AYNI ISARETI veriyor mu? (kural 7)")

    # ---- (v) dMSE
    print("\n" + "=" * 96)
    print("(v) dMSE -- kohortu ayri knob yapmak duz kaymaya gore ne kazandirir?")
    print("=" * 96)
    n_a = int(T.loc[T["a"], "n"].sum())
    n_d = int(T.loc[~T["a"], "n"].sum())
    # kohort satir paylari TUM soguk uzerinden (yas<7 ve kisa trafolar dahil)
    tum = te[te["soguk"]]
    w_a = float((tum["tanim"].map(ilk_te) == KOHORT).mean())
    w_d = 1.0 - w_a
    print(
        f"  soguk icinde satir paylari: 05-11 {w_a:.4f} | diger {w_d:.4f} "
        f"(soguk test payi {P_SOGUK:.5f})"
    )
    print(f"  {'capa':<34} {'Delta':>9} {'dMSE':>10}")
    for (blok_ad, spec), (f, ss, ya, yd) in sonuc.items():
        dmse = -P_SOGUK * w_a * w_d * f * f
        print(f"  {blok_ad + ' / ' + spec:<34} {f:>+9.4f} {dmse:>+10.5f}")
    print("  formul: duz optimumdan sapma = w_a*w_d*Delta^2 (iki gruplu en kucuk kareler)")

    # ---- KIRPMA TABLOSU, en saglam spesifikasyon icin
    print("\n" + "=" * 96)
    print("KIRPMA TABLOSU -- her spesifikasyon, K en buyuk katkili trafo atilarak")
    print("=" * 96)
    print(f"  {'capa':<30} " + " ".join(f"{'K=' + str(k):>9}" for k in (0, 1, 5, 10, 25, 50)))
    for (blok_ad, spec), (f, ss, ya, yd) in sonuc.items():
        sat = []
        for K in (0, 1, 5, 10, 25, 50):
            ia = np.argsort(-np.abs(ya - ya.mean()))[K:]
            idd = np.argsort(-np.abs(yd - yd.mean()))[K:]
            sat.append(f"{ya[ia].mean() - yd[idd].mean():>+9.4f}")
        print(f"  {blok_ad + ' / ' + spec:<30} " + " ".join(sat))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
