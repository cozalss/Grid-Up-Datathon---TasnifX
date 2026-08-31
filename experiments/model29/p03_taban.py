"""p03 taban: yaz25 tezgahinda temel model + hata ayristirmasi (RMSLE nerede?)."""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p03_tezgah as T  # noqa: E402

K = T.__file__.rsplit("experiments", 1)[0].rstrip("/\\")
ARA = os.environ.get("ARA", os.path.join(os.path.dirname(os.path.abspath(__file__))))
t0 = time.time()
tr, te = T.ortam()
Xe, ye, Xd, yd, hd, d_soguk = T.veri(tr)
print(f"egitim {Xe.shape} deger {Xd.shape} ({time.time() - t0:.0f}s)", flush=True)

PK = dict(
    objective="l2", metric="l2", learning_rate=0.04, num_leaves=63,
    min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, lambda_l2=5.0, num_threads=8, verbose=-1, seed=7,
)
ds = lgb.Dataset(Xe, ye)
m = lgb.train(PK, ds, num_boost_round=600)
p = m.predict(Xd)
taban = T.rmsle(yd, p)
print(f"TABAN RMSLE(yaz25) = {taban:.5f} ({time.time() - t0:.0f}s)", flush=True)

np.save(os.path.join(ARA, "p03_p_taban.npy"), p)
np.save(os.path.join(ARA, "p03_yd.npy"), yd)
hd[["tanim", "tarih", "guc", "lokasyon", "tuketim"]].to_parquet(
    os.path.join(ARA, "p03_hed.parquet")
)

# --- hata ayristirmasi ---
y = hd.tuketim.to_numpy()
e2 = (p - yd) ** 2
soguk = hd.tanim.isin(d_soguk).to_numpy()
kesik = [-1, 0, 1, 10, 50, 100, 500, 1000, 5000, 1e5, 1e12]
etik = ["=0", "(0,1]", "(1,10]", "(10,50]", "(50,100]", "(100,500]",
        "(500,1e3]", "(1e3,5e3]", "(5e3,1e5]", ">1e5"]
kova = pd.cut(y, bins=kesik, labels=etik)
df = pd.DataFrame({"kova": kova, "e2": e2, "soguk": soguk, "y": y, "p": p, "yd": yd})
tab = df.groupby("kova", observed=False).agg(
    n=("e2", "size"), pay_satir=("e2", lambda s: len(s) / len(df)),
    e2_ort=("e2", "mean"), pay_hata=("e2", lambda s: s.sum() / e2.sum()),
    sapma=("e2", lambda s: 0.0),
)
tab["sapma"] = df.groupby("kova", observed=False).apply(
    lambda d: float((d.p - d.yd).mean()), include_groups=False
)
print("\n--- tuketim kovasina gore hata ---")
print(tab.to_string())

sg = df.groupby("soguk").agg(n=("e2", "size"), e2_ort=("e2", "mean"),
                             pay_hata=("e2", lambda s: s.sum() / e2.sum()))
sg["rmsle"] = np.sqrt(sg.e2_ort)
sg["sapma"] = df.groupby("soguk").apply(lambda d: float((d.p - d.yd).mean()),
                                        include_groups=False)
print("\n--- soguk/sicak ---")
print(sg.to_string())

R = {
    "taban_rmsle": taban,
    "egitim_satir": int(len(Xe)),
    "deger_satir": int(len(Xd)),
    "ozellik": int(Xe.shape[1]),
    "kova": json.loads(tab.reset_index().to_json(orient="records")),
    "soguk": json.loads(sg.reset_index().to_json(orient="records")),
    "genel_sapma": float((p - yd).mean()),
}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p03_taban.json"),
          "w", encoding="utf-8") as f:
    json.dump(R, f, indent=1, ensure_ascii=False)
print(f"\nbitti {time.time() - t0:.0f}s")
