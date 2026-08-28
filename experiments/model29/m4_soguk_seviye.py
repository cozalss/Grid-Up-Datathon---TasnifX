"""SOGUK trafonun SEVIYESI ne kadar tahmin edilebilir? Sinyal envanteri."""

import numpy as np
import pandas as pd
from m1_geriteste import kes, yukle

tr = yukle()
kesim = "2025-10-31"
gec, hed = kes(tr, kesim)
gec = gec.copy()
gec["ly"] = np.log1p(gec.tuketim)
hed = hed.copy()
hed["ly"] = np.log1p(hed.tuketim)

# trafo duzeyinde calis: soguk trafolarin GERCEK seviyesi
soguk = (
    hed[hed.soguk]
    .groupby("tanim")
    .agg(
        y=("ly", "mean"),
        n=("ly", "size"),
        guc=("guc", "first"),
        il=("il", "first"),
        bolge=("bolge", "first"),
        ilce=("ilce", "first"),
        ilk=("tarih", "min"),
    )
)
# gecmisteki trafolarin seviyeleri (referans havuz)
ref = gec.groupby("tanim").agg(
    y=("ly", "mean"),
    guc=("guc", "first"),
    il=("il", "first"),
    bolge=("bolge", "first"),
    ilce=("ilce", "first"),
)

print(f"soguk trafo {len(soguk)}, seviye std {soguk.y.std():.4f}  (aciklanacak varyans)")
g0 = ref.y.mean()


def skor(tahmin, ad, agirlikli=True):
    t = np.asarray(tahmin, float)
    t = np.where(np.isnan(t), g0, t)
    w = soguk.n.values if agirlikli else np.ones(len(soguk))
    e = np.sqrt(np.average((t - soguk.y.values) ** 2, weights=w))
    r2 = 1 - np.average((t - soguk.y.values) ** 2, weights=w) / np.average(
        (soguk.y.values - np.average(soguk.y.values, weights=w)) ** 2, weights=w
    )
    print(f"  {ad:38s} seviye-RMSE {e:.4f}   R2 {r2:+.4f}")
    return e


skor(np.full(len(soguk), g0), "global sabit")
for anah in ["guc", "ilce", "bolge"]:
    skor(soguk[anah].map(ref.groupby(anah).y.mean()), f"grup: {anah}")
skor(
    pd.MultiIndex.from_frame(soguk[["guc", "bolge"]]).map(ref.groupby(["guc", "bolge"]).y.mean()),
    "grup: guc+bolge",
)
skor(
    pd.MultiIndex.from_frame(soguk[["guc", "ilce"]]).map(ref.groupby(["guc", "ilce"]).y.mean()),
    "grup: guc+ilce",
)
# log(guc) dogrusal

c = np.polyfit(np.log(ref.guc.clip(lower=1)), ref.y, 1)
skor(np.polyval(c, np.log(soguk.guc.clip(lower=1))), "log(guc) dogrusal")

# --- ID KOMSULUGU ---
ref_s = ref.sort_index()
refnum = pd.to_numeric(pd.Series(ref_s.index.astype(str)), errors="coerce").values
ok = ~np.isnan(refnum)
ordr = np.argsort(refnum[ok])
ids = refnum[ok][ordr]
vals = ref_s.y.values[ok][ordr]
snum = pd.to_numeric(pd.Series(soguk.index.astype(str)), errors="coerce").values
for k in [1, 3, 5, 10, 25, 50]:
    pos = np.searchsorted(ids, np.nan_to_num(snum, nan=-1))
    out = np.empty(len(soguk))
    for i, p in enumerate(pos):
        lo = max(0, p - k)
        hi = min(len(ids), p + k)
        out[i] = np.median(vals[lo:hi]) if hi > lo else np.nan
    skor(out, f"ID komsulugu (+-{k} medyan)")

# ID blogu (ust basamaklar)
for d in [2, 3, 4]:
    key = np.floor(snum / 10**d)
    rk = pd.Series(vals, index=np.floor(ids / 10**d)).groupby(level=0).mean()
    skor(pd.Series(key).map(rk).values, f"ID blok //10^{d}")
