"""v102'nin soguk seviyesi gercekten yuksek mi? guc/lokasyon bilesimini kontrol et."""

import os

import numpy as np
import pandas as pd
from m1_geriteste import KOK, kes, yukle

tr = pd.read_csv(
    os.path.join(KOK, "data/raw/train.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
te = pd.read_csv(
    os.path.join(KOK, "data/raw/test.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
sub = pd.read_csv(os.path.join(KOK, "submissions/tuketim_v102_kappa_optimum.csv"))
te["p"] = sub.tuketim.values
te["lp"] = np.log1p(te.p)
te["soguk"] = ~te.tanim.isin(set(tr.tanim))
tr["ly"] = np.log1p(tr.tuketim)

print("=== guc dagilimi: TEST soguk vs TEST sicak vs TRAIN ===")
for ad, s in [
    ("test soguk", te[te.soguk].guc),
    ("test sicak", te[~te.soguk].guc),
    ("train", tr.guc),
]:
    print(
        f"  {ad:11s} medyan {s.median():7.0f}  ort {s.mean():8.0f}  log-ort {np.log(s.clip(lower=1)).mean():.3f}"
    )

# guc-grubu modeli (tum train'den) test soguk satirlarina ne der?
gucm = tr.groupby("guc").ly.mean()
g0 = tr.ly.mean()
tahm_soguk = te.loc[te.soguk, "guc"].map(gucm).fillna(g0)
tahm_sicak = te.loc[~te.soguk, "guc"].map(gucm).fillna(g0)
print("\nguc-grubu modelinin ongordugu log-seviye:")
print(
    f"  test soguk {tahm_soguk.mean():.4f}   test sicak {tahm_sicak.mean():.4f}   fark {tahm_soguk.mean() - tahm_sicak.mean():+.4f}"
)
print("v102'nin verdigi:")
print(
    f"  test soguk {te[te.soguk].lp.mean():.4f}   test sicak {te[~te.soguk].lp.mean():.4f}   fark {te[te.soguk].lp.mean() - te[~te.soguk].lp.mean():+.4f}"
)
print(
    f"  -> v102 soguk sapmasi (guc-grubu tabanina gore): {te[te.soguk].lp.mean() - tahm_soguk.mean():+.4f}"
)

print("\n=== ayni olcum GERI-TESTTE: guc-grubu tahmini vs GERCEK ===")
tr2 = yukle()
for kesim in ["2025-08-31", "2025-09-30", "2025-10-31", "2025-11-30"]:
    gec, hed = kes(tr2, kesim)
    gec = gec.copy()
    gec["ly"] = np.log1p(gec.tuketim)
    hed = hed.copy()
    hed["ly"] = np.log1p(hed.tuketim)
    gm = gec.groupby("guc").ly.mean()
    gg = gec.ly.mean()
    s = hed[hed.soguk]
    t = s.guc.map(gm).fillna(gg)
    print(
        f"  {kesim}: guc-grubu tahmini {t.mean():.4f}  GERCEK {s.ly.mean():.4f}  yanlilik {t.mean() - s.ly.mean():+.4f}"
    )

print("\n=== SABIT KAYDIRMA KAZANC HESABI (test uzerinde, varsayimla) ===")
mse = 1.00553**2
pay = te.soguk.mean()
for b in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    yeni = mse - pay * b * b
    print(f"  soguk yanlilik b={b:.2f} -> MSE {mse:.6f} -> {yeni:.6f}  RMSLE {np.sqrt(yeni):.5f}")
