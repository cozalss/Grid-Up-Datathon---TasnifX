"""Huber etrafinda ikinci supurme + kayip ailesi karsilastirmasi."""

import json
import time

import numpy as np
from m34_supurme import kos

DENEY = [
    ("huber a=1.0", dict(objective="huber", alpha=1.0)),
    ("huber a=2.0", dict(objective="huber", alpha=2.0)),
    ("huber a=3.0", dict(objective="huber", alpha=3.0)),
    ("huber a=5.0", dict(objective="huber", alpha=5.0)),
    ("fair c=2", dict(objective="fair", fair_c=2.0)),
    ("l1 (mae)", dict(objective="l1")),
    ("huber a2 + min_data500", dict(objective="huber", alpha=2.0, min_data_in_leaf=500)),
    ("huber a2 + l2=20", dict(objective="huber", alpha=2.0, lambda_l2=20.0)),
    (
        "huber a2 + yaprak31 lr.03",
        dict(objective="huber", alpha=2.0, num_leaves=31, learning_rate=0.03),
    ),
    ("huber a2 + yaprak127", dict(objective="huber", alpha=2.0, num_leaves=127)),
]
sonuc = {}
for dog in ["2025-11-30", "2025-09-30"]:
    print(f"\n########## DOGRULAMA {dog} ##########", flush=True)
    t0 = time.time()
    for ad, kw in DENEY:
        r = kos(dog, **kw)
        print(
            f"  {ad:30s} tur {r['tur']:4d} RMSLE {r['rmsle']:.4f} soguk {r['soguk']:.4f} "
            f"sicak {r['sicak']:.4f} | test-karisimi {r['karisik']:.4f}  ({time.time() - t0:.0f}s)",
            flush=True,
        )
        sonuc.setdefault(ad, {})[dog] = r
json.dump(sonuc, open("m35_huber.json", "w"), indent=1)
print("\n=== IKI KESIMIN ORTALAMASI (test-karisimi) ===")
for ad, d in sorted(sonuc.items(), key=lambda kv: np.mean([v["karisik"] for v in kv[1].values()])):
    print(
        f"  {ad:30s} {np.mean([v['karisik'] for v in d.values()]):.4f}   "
        + "  ".join(f"{k}:{v['karisik']:.4f}" for k, v in d.items())
    )
