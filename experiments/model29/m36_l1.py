"""L1/quantile etrafinda ucuncu supurme. Neden L1 kazaniyor -> optimum quantile nerede?"""

import json
import time

import numpy as np
from m34_supurme import kos

DENEY = [
    ("l1 taban", dict(objective="l1")),
    ("quantile .45", dict(objective="quantile", alpha=0.45)),
    ("quantile .55", dict(objective="quantile", alpha=0.55)),
    ("quantile .60", dict(objective="quantile", alpha=0.60)),
    ("l1 + yaprak127", dict(objective="l1", num_leaves=127)),
    ("l1 + yaprak31", dict(objective="l1", num_leaves=31)),
    ("l1 + lr .08", dict(objective="l1", learning_rate=0.08)),
    ("l1 + min_data 50", dict(objective="l1", min_data_in_leaf=50)),
    ("l1 + min_data 800", dict(objective="l1", min_data_in_leaf=800)),
    ("l1 + ff1.0 bf1.0", dict(objective="l1", feature_fraction=1.0, bagging_fraction=1.0)),
]
sonuc = {}
for dog in ["2025-11-30", "2025-09-30"]:
    print(f"\n########## DOGRULAMA {dog} ##########", flush=True)
    t0 = time.time()
    for ad, kw in DENEY:
        r = kos(dog, **kw)
        print(
            f"  {ad:26s} tur {r['tur']:4d} RMSLE {r['rmsle']:.4f} soguk {r['soguk']:.4f} "
            f"sicak {r['sicak']:.4f} | test-karisimi {r['karisik']:.4f}  ({time.time() - t0:.0f}s)",
            flush=True,
        )
        sonuc.setdefault(ad, {})[dog] = r
json.dump(sonuc, open("m36_l1.json", "w"), indent=1)
print("\n=== IKI KESIM ORTALAMASI ===")
for ad, d in sorted(sonuc.items(), key=lambda kv: np.mean([v["karisik"] for v in kv[1].values()])):
    print(
        f"  {ad:26s} {np.mean([v['karisik'] for v in d.values()]):.4f}  "
        + "  ".join(f"{k}:{v['karisik']:.4f}" for k, v in d.items())
    )
