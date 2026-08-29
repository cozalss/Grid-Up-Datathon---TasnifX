import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
KOK = Path(__file__).resolve().parents[2]
te = pd.read_csv(KOK / "data/raw/test.csv", usecols=["id"])
ss = pd.read_csv(KOK / "data/raw/sample_submission.csv")
print("test satir:", len(te), " sample kolon:", list(ss.columns), " sample satir:", len(ss))
print("test.id == sample.id ?", bool((te.id.values == ss.iloc[:, 0].values).all()))

for ad in [
    "tuketim_y46_amnezik_kirpik.csv",
    "tuketim_g7_span_tau3.csv",
    "tuketim_m6_ikiyon.csv",
    "tuketim_y45_mevsimsel_kirpik.csv",
    "tuketim_y40_sota_temiz.csv",
]:
    t = pd.read_csv(KOK / "submissions" / ad)
    v = t.iloc[:, 1].to_numpy(dtype="float64")
    lg = np.log1p(v)
    print(f"\n--- {ad}")
    print(
        f"  kolonlar={list(t.columns)} satir={len(t)} id_birebir={bool((t.iloc[:, 0].values == ss.iloc[:, 0].values).all())}"
    )
    print(
        f"  NaN={int(np.isnan(v).sum())} sonsuz={int(np.isinf(v).sum())} negatif={int((v < 0).sum())} tam_sifir={int((v == 0).sum())}"
    )
    print(
        f"  ham: min={v.min():.6g} maks={v.max():.6g} ort={v.mean():.4f} medyan={np.median(v):.4f}"
    )
    print(f"  log1p: ort={lg.mean():.5f} std={lg.std():.5f} maks={lg.max():.4f} min={lg.min():.4f}")
    print(
        f"  <1kWh orani={float((v < 1).mean()):.5f}  >1e5 satir={int((v > 1e5).sum())}  >1e6 satir={int((v > 1e6).sum())}"
    )
    q = np.quantile(v, [0.001, 0.01, 0.5, 0.99, 0.999, 0.99999])
    print(
        f"  yuzdelikler .1%={q[0]:.4f} 1%={q[1]:.4f} 50%={q[2]:.4f} 99%={q[3]:.2f} 99.9%={q[4]:.2f} 99.999%={q[5]:.2f}"
    )
