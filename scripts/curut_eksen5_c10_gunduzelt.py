"""CURUTME -- KALICILIK CAPASI, GUN KOMPOZISYONU DUZELTILMIS (Kural 6).
Kismi kapsamali trafolar (or. 05-11'de baslayanlar) yazin SICAK kismini
gorur; duzeltmeden kald26 mekanik olarak sisiyor.
Ayrica S25 tabakalanmis (ilce x guc) olarak atfediliyor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path("C:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX")
tr = pd.read_csv(
    KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
)
tr["r"] = np.log1p(tr["tuketim"].clip(lower=0.0)) - np.log1p(tr["guc"])
te = pd.read_csv(
    KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
)
v55 = pd.read_csv(KOK / "submissions/tuketim_v55_gunolcek.csv", encoding="utf-8")
te = te.merge(v55, on="id", validate="one_to_one")
te["r"] = np.log1p(te["tuketim"].clip(lower=0.0)) - np.log1p(te["guc"])
tr["ay"] = tr["tarih"].dt.to_period("M")

SICAK = set(tr["tanim"].unique())
q126 = tr[(tr["tarih"] >= "2026-01-01") & (tr["tarih"] <= "2026-03-31")]
HEDEF = SICAK & set(q126.loc[q126["tuketim"] > 0, "tanim"].unique())

g = (
    tr.groupby(["tanim", "ay"], observed=True)
    .agg(nz=("tuketim", lambda s: int((s > 0).sum())))
    .reset_index()
)
ays = tr.groupby("tanim")["ay"].nunique()
nza = g[g["nz"] > 0].groupby("tanim")["ay"].nunique()
PANEL = set(ays[ays == 15].index) & set(nza[nza == 15].index)


# ---- GUN PROFILLERI (trafo etkisi cikarilmis; Kural 6)
def profil(df, tanimlar, a, b):
    q = df[df["tanim"].isin(tanimlar) & (df["tarih"] >= a) & (df["tarih"] <= b)].copy()
    q["rc"] = q["r"] - q.groupby("tanim", observed=True)["r"].transform("mean")
    f = q.groupby("tarih")["rc"].mean()
    return f - f.mean()


f_te = profil(te, PANEL, "2026-04-01", "2026-07-31")  # v55'in gun profili
f_q1 = profil(tr, PANEL, "2026-01-01", "2026-03-31")  # 2026 Q1 gun profili
print(
    "gun profili: test sd %.4f (n=%d)  Q1 sd %.4f (n=%d)"
    % (f_te.std(), len(f_te), f_q1.std(), len(f_q1))
)

th = te[te["tanim"].isin(HEDEF)].copy()
th["f"] = th["tarih"].map(f_te)
qh = q126[q126["tanim"].isin(HEDEF)].copy()
qh["f"] = qh["tarih"].map(f_q1)

P = th.groupby("tanim", observed=True).agg(P26=("r", "mean"), fte=("f", "mean"), nte=("r", "size"))
Q = qh.groupby("tanim", observed=True).agg(
    L26q1=("r", "mean"), fq1=("f", "mean"), nq1=("r", "size")
)
A = P.join(Q, how="inner")
A["kald26_ham"] = A["P26"] - A["L26q1"]
A["kald26"] = (A["P26"] - A["fte"]) - (A["L26q1"] - A["fq1"])  # gun-duzeltilmis
print(f"\ntrafo {len(A):,} satir {int(A['nte'].sum()):,}")

w1 = tr[(tr["tarih"] >= "2025-01-01") & (tr["tarih"] <= "2025-03-31")].groupby("tanim")["r"].size()
w3 = tr[(tr["tarih"] >= "2025-04-01") & (tr["tarih"] <= "2025-07-31")].groupby("tanim")["r"].size()
A["olculebilir"] = (
    (A.index.map(w1).fillna(0) >= 20) & (A.index.map(w3).fillna(0) >= 20) & (A["nq1"] >= 20)
)
A["w"] = A["nte"]


def ag(x, w):
    return float((x * w).sum() / w.sum())


print("\n=== kald26: HAM vs GUN-DUZELTILMIS ===")
for k, q in A.groupby("olculebilir"):
    print(
        f"  olculebilir={k}  satir {int(q['w'].sum()):>7,}"
        f"  ham {ag(q['kald26_ham'], q['w']):+.5f}  gun-duz {ag(q['kald26'], q['w']):+.5f}"
        f"  (gun kompozisyon etkisi {ag(q['kald26_ham'] - q['kald26'], q['w']):+.5f})"
    )
print(
    f"  TUMU              satir {int(A['w'].sum()):>7,}"
    f"  ham {ag(A['kald26_ham'], A['w']):+.5f}  gun-duz {ag(A['kald26'], A['w']):+.5f}"
)

# ---- S25: TABAKALANMIS ATIF (ilce x guc kovasi), tum uygun trafolardan
ref = tr[(w1.reindex(tr["tanim"]).fillna(0).to_numpy() >= 20)]
o1 = tr[(tr["tarih"] >= "2025-01-01") & (tr["tarih"] <= "2025-03-31")].groupby("tanim")["r"].mean()
o3 = tr[(tr["tarih"] >= "2025-04-01") & (tr["tarih"] <= "2025-07-31")].groupby("tanim")["r"].mean()
k25 = (o3 - o1).dropna()
k25 = k25[(w1.reindex(k25.index).fillna(0) >= 20) & (w3.reindex(k25.index).fillna(0) >= 20)]
meta = te.drop_duplicates("tanim").set_index("tanim")
ilce = meta["lokasyon"].str.split(">").str[-1]
guc = meta["guc"]
kov = pd.qcut(guc, 6, labels=False, duplicates="drop")
K = pd.DataFrame({"k25": k25})
K["ilce"] = ilce.reindex(K.index)
K["kov"] = kov.reindex(K.index)
K = K.dropna()
print(f"\nS25 donor havuzu: {len(K):,} trafo   genel ort {K['k25'].mean():+.5f}")
st = K.groupby(["ilce", "kov"])["k25"].agg(["mean", "size"])
st_ilce = K.groupby("ilce")["k25"].mean()
A["ilce"] = ilce.reindex(A.index)
A["kov"] = kov.reindex(A.index)
key = list(zip(A["ilce"], A["kov"]))
s_hat = pd.Series(
    [st["mean"].get(k, np.nan) if st["size"].get(k, 0) >= 15 else np.nan for k in key],
    index=A.index,
)
s_hat = s_hat.fillna(A["ilce"].map(st_ilce)).fillna(K["k25"].mean())
A["S25"] = s_hat
print(
    "atfedilen S25:",
    {k: round(float(ag(q["S25"], q["w"])), 5) for k, q in A.groupby("olculebilir")},
)

print("\n=== YANLILIK b = (S25_atf + hava) - kald26_gunduz ===")
print(f"{'hava':>9} {'TUM':>10} {'olculebilir':>12} {'olculemez':>11}")
for hava in (0.0, -0.0518, -0.0574, -0.0845):
    A["b"] = (A["S25"] + hava) - A["kald26"]
    r = [ag(A["b"], A["w"])]
    for k in (True, False):
        q = A[A["olculebilir"] == k]
        r.append(ag(q["b"], q["w"]))
    print(f"{hava:+9.4f} {r[0]:+10.5f} {r[1]:+12.5f} {r[2]:+11.5f}")

A["b"] = (A["S25"] - 0.0574) - A["kald26"]
print("\nKIRPMA TABLOSU (b, satir agirlikli, tum hedef)")
s = A.reindex(A["b"].abs().sort_values(ascending=False).index)
for Kk in (0, 1, 5, 10, 25, 50, 100, 200, 400):
    q = s.iloc[Kk:]
    print(
        f"  K={Kk:>4} n={len(q):>5} satir={int(q['w'].sum()):>7,}"
        f" b satir-ag {ag(q['b'], q['w']):+.5f}  medyan {q['b'].median():+.5f}"
    )
A.to_csv(KOK / "reports" / "_c10_b.csv")
print("\nyazildi reports/_c10_b.csv")
