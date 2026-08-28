"""Huber modelinin GLOBAL SEVIYE yanliligi var mi? Geri-testte olc + optimum kaydirma."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from m33_durust import VARSAYILAN, hizala
from m34_supurme import al


def kos_tam(dog, **pk):
    Xva, yva = al(dog, None)
    sog = Xva.soguk.values.astype(bool)
    AY = [
        "2025-03-31",
        "2025-04-30",
        "2025-05-31",
        "2025-06-30",
        "2025-07-31",
        "2025-08-31",
        "2025-09-30",
        "2025-10-31",
        "2025-11-30",
        "2025-12-31",
    ]
    Xs, ys = [], []
    for k in [x for x in AY if x < dog]:
        r = al(k, dog)
        if r:
            Xs.append(r[0])
            ys.append(r[1])
    Xtr = pd.concat(Xs, ignore_index=True)
    ytr = np.concatenate(ys)
    Xtr, Xva2 = hizala(Xtr.copy(), Xva.copy())
    p = dict(VARSAYILAN)
    p.update(pk)
    m = lgb.train(
        p,
        lgb.Dataset(Xtr, ytr),
        4000,
        valid_sets=[lgb.Dataset(Xva2, yva)],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    return m.predict(Xva2, num_iteration=m.best_iteration), yva, sog


for dog in ["2025-11-30", "2025-09-30"]:
    pv, yva, sog = kos_tam(dog, objective="huber", alpha=2.0, lambda_l2=20.0)
    b = float((pv - yva).mean())
    e0 = np.sqrt(((pv - yva) ** 2).mean())
    e1 = np.sqrt(((pv - b - yva) ** 2).mean())
    bs = float((pv[sog] - yva[sog]).mean())
    bw = float((pv[~sog] - yva[~sog]).mean())
    print(f"{dog}: GLOBAL yanlilik {b:+.4f}  (soguk {bs:+.4f}  sicak {bw:+.4f})")
    print(f"    RMSLE {e0:.4f} -> yanlilik giderilince {e1:.4f}  (kazanc {e0 - e1:.4f})")
    # ayri kaydirma
    p2 = pv.copy()
    p2[sog] -= bs
    p2[~sog] -= bw
    print(f"    rejim bazli kaydirma ile {np.sqrt(((p2 - yva) ** 2).mean()):.4f}")
