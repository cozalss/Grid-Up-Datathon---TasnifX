"""SIZINTISIZ ileri-pencere tezgahi.
Iki sizinti kapatildi:
  (1) egitim kesimlerinin hedef penceresi DOGRULAMA KESIMINDE kesilir
      -> ayni (trafo,gun) satiri hem egitimde hem dogrulamada olamaz
  (2) dogrulamada SOGUK olan trafolarin TUM satirlari egitimden atilabilir
      -> ID/idnum ezber kanali kapanir (soguk hukmu durustlesir)
"""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from m30_ozellik import kur, yukle_ham


def parca(tr, kesim, ay=4, tavan=None, at_trafo=None):
    k = pd.Timestamp(kesim)
    son = k + pd.DateOffset(months=ay)
    if tavan is not None:
        son = min(son, pd.Timestamp(tavan))
    gec = tr[tr.tarih <= k]
    hed = tr[(tr.tarih > k) & (tr.tarih <= son)]
    if len(hed) == 0:
        return None
    sicak = set(gec.tanim)
    if at_trafo is not None:
        gec = gec[~gec.tanim.isin(at_trafo)]
        hed = hed[~hed.tanim.isin(at_trafo)]
        if len(hed) == 0:
            return None
    X = kur(gec, hed, kesim, sicak)
    return X, np.log1p(hed.tuketim.values)


def hizala(Xtr, Xva):
    for c in ("il", "bolge", "ilce"):
        cats = Xtr[c].cat.categories.union(Xva[c].cat.categories)
        Xtr[c] = Xtr[c].cat.set_categories(cats)
        Xva[c] = Xva[c].cat.set_categories(cats)
    return Xtr, Xva


VARSAYILAN = dict(
    objective="l2",
    metric="l2",
    learning_rate=0.04,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    num_threads=14,
    verbose=-1,
    seed=7,
)


def egit_dogrula(tr, egit_kesimleri, dog_kesim, drop=(), durust_soguk=False, tur=None, **pk):
    Xva, yva = parca(tr, dog_kesim)
    sogv = Xva.soguk.values.astype(bool)
    at = None
    if durust_soguk:
        gec_v = tr[tr.tarih <= pd.Timestamp(dog_kesim)]
        hed_v = tr[(tr.tarih > pd.Timestamp(dog_kesim))]
        at = set(hed_v.tanim) - set(gec_v.tanim)  # dogrulamada soguk olanlar
    Xs, ys = [], []
    for k in egit_kesimleri:
        r = parca(tr, k, tavan=dog_kesim, at_trafo=at)
        if r is None:
            continue
        Xs.append(r[0])
        ys.append(r[1])
    Xtr = pd.concat(Xs, ignore_index=True)
    ytr = np.concatenate(ys)
    Xtr = Xtr.drop(columns=list(drop))
    Xva2 = Xva.drop(columns=list(drop))
    Xtr, Xva2 = hizala(Xtr, Xva2)
    p = dict(VARSAYILAN)
    p.update(pk)
    ds = lgb.Dataset(Xtr, ytr)
    if tur is None:
        dv = lgb.Dataset(Xva2, yva, reference=ds)
        m = lgb.train(
            p, ds, 4000, valid_sets=[dv], callbacks=[lgb.early_stopping(150, verbose=False)]
        )
        n = m.best_iteration
    else:
        m = lgb.train(p, ds, tur)
        n = tur
    pv = m.predict(Xva2, num_iteration=n)
    L = (pv - yva) ** 2
    return (
        dict(
            n_egit=len(Xtr),
            tur=int(n),
            rmsle=float(np.sqrt(L.mean())),
            soguk=float(np.sqrt(L[sogv].mean())),
            sicak=float(np.sqrt(L[~sogv].mean())),
            soguk_kutle=float(L[sogv].sum() / L.sum()),
        ),
        m,
        pv,
        yva,
        sogv,
    )


if __name__ == "__main__":
    tr, te = yukle_ham()
    DOG = "2025-11-30"
    # hedef penceresi dogrulama kesimini asmayacak sekilde kirpilir -> cakisma yok
    EGIT = [
        "2025-04-30",
        "2025-05-31",
        "2025-06-30",
        "2025-07-31",
        "2025-08-31",
        "2025-09-30",
        "2025-10-31",
    ]
    print("kis26 penceresi (2025-12-01..2026-03-31) -- URETIM HATTI KAYDI:")
    print("  v83  sicak 0.77826  soguk 1.90610")
    print("  YIGIN(en iyi CV) sicak 0.76150  soguk 1.86720\n")
    t0 = time.time()
    for ad, kw in [
        ("cakisma kapali", dict()),
        ("cakisma+ezber kapali", dict(durust_soguk=True)),
        ("cakisma+ezber kapali, idnum yok", dict(durust_soguk=True, drop=("idnum",))),
    ]:
        r, _, _, _, _ = egit_dogrula(tr, EGIT, DOG, **kw)
        print(
            f"{ad:34s} egitim {r['n_egit']:8,d} tur {r['tur']:4d}  "
            f"RMSLE {r['rmsle']:.4f}  soguk {r['soguk']:.4f}  sicak {r['sicak']:.4f}  ({time.time() - t0:.0f}s)",
            flush=True,
        )
