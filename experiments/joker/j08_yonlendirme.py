"""J8 - sicak/soguk yonlendirmesi TEST'te dogru mu?

Tezgahta 'gecmisi yok' == 'soguk' (3/3 blokta birebir). TEST'te de oyle mi?
Gecmisi olmayan bir trafo SICAK modele giderse t_* kolonlari NaN olur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KOK = r"C:\Users\Cem\Desktop\Datahon_Laptop\Grid-Up-Datathon---TasnifX"


def p(*a):
    print(*a, flush=True)


tp = pd.read_parquet(
    KOK + r"\data\interim\deney\test.parquet",
    columns=[
        "tanim",
        "tarih",
        "guc",
        "soguk_mu",
        "t_log_ort",
        "t_gun_sayisi",
        "t_kuyruk_sifir",
        "t_olu_mu",
        "p_gun_sayisi",
        "ufuk_gun",
    ],
)
te = pd.read_csv(KOK + r"\data\raw\test.csv", dtype={"tanim": str})
tr = pd.read_csv(
    KOK + r"\data\raw\train.csv", dtype={"tanim": str}, usecols=["tanim", "tarih", "tuketim"]
)
p("test.parquet satir:", len(tp), " test.csv satir:", len(te))
tp["tanim"] = tp["tanim"].astype(str)
tr_set = set(tr["tanim"].unique())
tp["yeni"] = ~tp["tanim"].isin(tr_set)
p()
p("=== capraz tablo: soguk_mu x gecmisi-yok ===")
p(pd.crosstab(tp["soguk_mu"], tp["yeni"], margins=True).to_string())
p()
p(
    "soguk satir:",
    int((tp["soguk_mu"] == 1).sum()),
    " (test'in %.4f'u)" % (tp["soguk_mu"] == 1).mean(),
)
p("sicak satir:", int((tp["soguk_mu"] != 1).sum()))
p(
    "v83 sicak cekirdek 526.446 -> sicak satir sayisiyla ayni mi:",
    int((tp["soguk_mu"] != 1).sum()) == 526446,
)

p()
p("=== gecmisi olmayan ama SICAK'a giden satirlar ===")
kotu = tp[(tp["soguk_mu"] != 1) & tp["yeni"]]
p("n =", len(kotu), " trafo =", kotu["tanim"].nunique())
if len(kotu):
    p(kotu[["t_log_ort", "t_gun_sayisi", "t_kuyruk_sifir"]].describe().to_string())

p()
p("=== gecmisi VAR ama SOGUK'a giden satirlar ===")
kotu2 = tp[(tp["soguk_mu"] == 1) & ~tp["yeni"]]
p("n =", len(kotu2), " trafo =", kotu2["tanim"].nunique())
if len(kotu2):
    d = kotu2.groupby("tanim").agg(
        n=("guc", "size"), gun=("t_gun_sayisi", "first"), logort=("t_log_ort", "first")
    )
    p(d.describe().to_string())
    # bu trafolarin train'deki son kaydi
    son = tr.groupby("tanim")["tarih"].max()
    ss = son.reindex(kotu2["tanim"].unique())
    p("train'deki son kayit tarihi dagilimi (en sik 8):")
    p(ss.value_counts().head(8).to_string())
    p("train satir sayisi dagilimi:")
    trn = tr.groupby("tanim").size().reindex(kotu2["tanim"].unique())
    p(trn.describe().to_string())

p()
p("=== SICAK tarafta NaN gecmis kolonu var mi ===")
sic = tp[tp["soguk_mu"] != 1]
for c in ["t_log_ort", "t_gun_sayisi", "t_kuyruk_sifir", "t_olu_mu"]:
    p(
        "  %-16s NaN=%d  min=%s"
        % (c, int(sic[c].isna().sum()), np.nanmin(sic[c].to_numpy(dtype=float)))
    )

p()
p("=== v83 tahminleri gruba gore ===")
v83 = pd.read_csv(KOK + r"\submissions\tuketim_v83_sicak_optimum.csv")
m = te[["id"]].merge(v83, on="id", how="left")
lp = np.log1p(m["tuketim"].to_numpy())
# test.parquet sirasi test.csv ile ayni mi
ayni = bool((tp["tanim"].to_numpy() == te["tanim"].to_numpy()).all())
p("test.parquet sirasi test.csv ile ayni:", ayni)
if ayni:
    s = (tp["soguk_mu"] == 1).to_numpy()
    p("  sicak n=%d ort log1p=%.4f" % ((~s).sum(), lp[~s].mean()))
    p("  soguk n=%d ort log1p=%.4f" % (s.sum(), lp[s].mean()))
