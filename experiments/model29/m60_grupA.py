"""GRUP A (gecmisi bastan sona sifir olan SICAK trafolar): model ne diyor, gercek ne?
Geri-testte olc. Bulunursa test'te 25.566 satir (%3,6)."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from m33_durust import VARSAYILAN, hizala
from m34_supurme import al, tr

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


def kos(dog):
    Xva, yva = al(dog, None)
    sog = Xva.soguk.values.astype(bool)
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
    p.update(objective="huber", alpha=2.0, lambda_l2=20.0)
    m = lgb.train(
        p,
        lgb.Dataset(Xtr, ytr),
        4000,
        valid_sets=[lgb.Dataset(Xva2, yva)],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    ph = m.predict(Xva2, num_iteration=m.best_iteration)
    p2 = dict(VARSAYILAN)
    p2.update(objective="l1")
    m2 = lgb.train(
        p2,
        lgb.Dataset(Xtr, ytr),
        4000,
        valid_sets=[lgb.Dataset(Xva2, yva)],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    return (ph + m2.predict(Xva2, num_iteration=m2.best_iteration)) / 2, yva, sog, Xva


for dog in ["2025-11-30", "2025-09-30"]:
    pv, yva, sog, Xva = kos(dog)
    k = pd.Timestamp(dog)
    gec = tr[tr.tarih <= k]
    maxt = gec.groupby("tanim").tuketim.max()
    hed = tr[(tr.tarih > k) & (tr.tarih <= k + pd.DateOffset(months=4))]
    grupA = set(maxt[maxt < 1].index)
    mA = hed.tanim.isin(grupA).values & ~sog
    print(f"\n===== {dog}  toplam RMSLE {np.sqrt(((pv - yva) ** 2).mean()):.4f} =====")
    print(
        f"GRUP A (gecmis tamamen sifir, sicak): {mA.sum():,} satir (%{100 * mA.mean():.2f}), "
        f"{hed[mA].tanim.nunique()} trafo"
    )
    if mA.sum() == 0:
        continue
    print(
        f"  GERCEK log-ort {yva[mA].mean():.4f}   MODEL log-ort {pv[mA].mean():.4f}   "
        f"yanlilik {pv[mA].mean() - yva[mA].mean():+.4f}"
    )
    print(
        f"  bu dilimde RMSLE {np.sqrt(((pv[mA] - yva[mA]) ** 2).mean()):.4f}, "
        f"toplam kutlenin %{100 * ((pv[mA] - yva[mA]) ** 2).sum() / ((pv - yva) ** 2).sum():.1f}'i"
    )
    print(f"  hedefte uyanan (y>1) satir orani %{100 * (yva[mA] > 1).mean():.1f}")
    # sabit ile degistirmenin kazanci
    L0 = ((pv - yva) ** 2).mean()
    print(f"  {'sabit c':>8s} {'toplam RMSLE':>13s} {'kazanc':>8s}")
    for c in [0.5, 0.75, 1.0, 1.5, 2.0, yva[mA].mean()]:
        q = pv.copy()
        q[mA] = c
        print(
            f"  {c:8.3f} {np.sqrt(((q - yva) ** 2).mean()):13.4f} {np.sqrt(L0) - np.sqrt(((q - yva) ** 2).mean()):+8.4f}"
        )
    # kaydirma ile
    for d in [-0.5, -1.0, -1.5]:
        q = pv.copy()
        q[mA] = pv[mA] + d
        print(
            f"  kaydirma {d:+.1f}      {np.sqrt(((q - yva) ** 2).mean()):.4f} {np.sqrt(L0) - np.sqrt(((q - yva) ** 2).mean()):+8.4f}"
        )
