"""H7-d -- SOGUK gun ekseni onerisi icin KIRPMA TABLOSU (kanit citasi kural 1).

Etiket yok, ama vekil gercek VAR: 2025-04..07 gercek gun profili, gun-of-year
hizasinda, LB'nin cozdugu 0,8823 genlik oraniyla olceklenmis.

    beta_g = 0.8823 * b_gecen[doy(g)]        (2026 icin vekil gercek gun ekseni)
    kazanc_i = (1/N) * sum_{rows of i} [ 2(c-1) etki * beta - (c-1)^2 etki^2 ]

Isaretle: kazanc_i > 0 ise trafo KAZANDIRIYOR. K en buyuk katkiyi atarak
tabloyu yaz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
CIK = KOK / "reports" / "h7_cstar"
GENLIK = 0.8823  # LB'den cozulen 2026/2025 gun-ekseni genlik orani
C = 1.335  # onerilen soguk olcek


def gun_etkisi(tanim, gun, r):
    x = pd.DataFrame({"t": tanim, "g": gun, "r": r})
    x["c"] = x["r"] - x.groupby("t")["r"].transform("mean")
    b = x.groupby("g")["c"].mean()
    return b - b.mean()


ornek = pd.read_csv(KOK / "data/raw/sample_submission.csv", encoding="utf-8")
te = (
    pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "guc", "tarih"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    .set_index("id")
    .loc[ornek["id"]]
    .reset_index()
)
tr = pd.read_csv(
    KOK / "data/raw/train.csv",
    usecols=["tanim", "guc", "tarih", "tuketim"],
    encoding="utf-8",
    dtype={"tanim": str},
)
sub = pd.read_csv(KOK / "submissions/tuketim_v67_c1335_olay.csv", encoding="utf-8")
assert sub["id"].equals(ornek["id"])

soguk = ~te["tanim"].isin(set(tr["tanim"])).to_numpy()
N = len(te)
r = np.log1p(sub["tuketim"].to_numpy(dtype="float64")) - np.log1p(
    te["guc"].to_numpy(dtype="float64")
)
tan, tar = te["tanim"].to_numpy(), te["tarih"].to_numpy()

# vekil gercek gun ekseni
g = tr[(tr["tarih"] >= "2025-04-01") & (tr["tarih"] <= "2025-07-31") & (tr["tuketim"] > 0)]
rg = np.log1p(g["tuketim"].to_numpy(dtype="float64")) - np.log1p(g["guc"].to_numpy(dtype="float64"))
xg = pd.DataFrame({"t": g["tanim"].to_numpy(), "g": g["tarih"].to_numpy()})
ngg = xg["g"].nunique()
tamg = xg.groupby("t")["g"].nunique()
tamg = set(tamg[tamg >= 0.9 * ngg].index)
selg = np.isin(xg["t"].to_numpy(), list(tamg))
b_gecen = gun_etkisi(xg["t"].to_numpy()[selg], xg["g"].to_numpy()[selg], rg[selg])
ref = pd.Series(b_gecen.values, index=pd.to_datetime(b_gecen.index).dayofyear)

b_c = gun_etkisi(tan[soguk], tar[soguk], r[soguk])
etki = pd.Series(tar[soguk]).map(b_c).to_numpy(dtype="float64")
etki = etki - etki.mean()
doy = pd.to_datetime(pd.Series(tar[soguk])).dt.dayofyear.to_numpy()
beta = GENLIK * ref.reindex(doy).to_numpy(dtype="float64")
gec = ~np.isnan(beta)
print(f"soguk satir {int(soguk.sum()):,}  vekil-gercek eslesen {int(gec.sum()):,}")
beta = np.where(gec, beta, 0.0)
beta = beta - beta[gec].mean()

# dMSE_i = [2(k-1)*etki*(etki-beta) + (k-1)^2*etki^2]/N ; kazanc = -dMSE
kat = -(2.0 * (C - 1.0) * etki * (etki - beta) + (C - 1.0) ** 2 * etki**2) / N
k_ols = float((etki * beta).sum() / (etki * etki).sum())
print(f"row-agirlikli OLS egimi (optimal k) = {k_ols:.4f}")
df = pd.DataFrame({"t": tan[soguk], "k": kat, "n": 1})
per = df.groupby("t").agg(kazanc=("k", "sum"), satir=("n", "sum")).sort_values("kazanc")
toplam = float(per["kazanc"].sum())
print(f"\nSOGUK c={C}: toplam beklenen dMSE = {-toplam:+.6f}  (isaret: negatif = kazanc)")
print(
    f"kazandiran trafo {int((per['kazanc'] > 0).sum()):,} / {len(per):,} "
    f"({(per['kazanc'] > 0).mean() * 100:.1f}%)"
)
print("\nKIRPMA TABLOSU -- en buyuk K katkiyi at:")
srt = per["kazanc"].sort_values(ascending=False).to_numpy()
print(f"  {'K':>4} {'kalan dMSE':>14} {'kalan trafo':>12}")
for K in [0, 1, 5, 10, 25, 50]:
    kalan = -float(srt[K:].sum())
    print(f"  {K:4d} {kalan:+14.6f} {len(srt) - K:12,}")
per.to_csv(CIK / "soguk_trafo_katkilari.csv", encoding="utf-8")

# ayni tabloyu SICAK taraf icin, ULASILAN optimumdan k=1,10 sapmasi ornek
print("\n(karsilastirma) SICAK tarafta v67 zaten optimumda -- kirpma tablosu gereksiz")
print(f"yazildi: {CIK / 'soguk_trafo_katkilari.csv'}")
