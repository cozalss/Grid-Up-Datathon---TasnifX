"""Yapisal olarak FARKLI iki GBM adayi -- gurultu yok, farkli tumevarim yanliligi.

A) AMNEZIK   : trafo duzeyi GECMIS ozellikleri (h_*) TAMAMEN atilir.
               Model her trafoyu "soguk" gibi gorur: yalniz guc + lokasyon +
               takvim + grup seviyeleri + varlik deseni. Uretim hattinin
               omurgasi olan gecmis-ekstrapolasyonu yerine kesitsel
               heterojenlikten tahmin eder -> hatalari yapisal olarak baska yerde.
B) KAPASITE  : hedef log1p(tuketim) - log1p(guc). Kayip geometrisi degisir;
               model mutlak seviye yerine KAPASITE KULLANIM ORANINI ogrenir.

Ikisi de rejim bazinda (soguk / kuyruk / cekirdek) m6'nin seviyesine capalanir:
seviye yalnizca LB'de olculur, bu yuzden aday SEKIL tasisin, seviye tasimasin.
"""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)

from m30_ozellik import KOK, kur, yukle_ham  # noqa: E402
from m33_durust import VARSAYILAN, hizala, parca  # noqa: E402

KESIM = "2026-03-31"
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
TOHUMLAR = (7, 17)

t0 = time.time()
tr, te = yukle_ham()
Xs, ys = [], []
for k in AY:
    r = parca(tr, k, tavan=KESIM)
    if r:
        Xs.append(r[0])
        ys.append(r[1])
        print(f"  koken {k}: {len(r[0]):,} ({time.time() - t0:.0f}s)", flush=True)
Xtr = pd.concat(Xs, ignore_index=True)
ytr = np.concatenate(ys)
del Xs, ys
Xte = kur(tr, te, KESIM, set(tr.tanim))
Xtr, Xte = hizala(Xtr, Xte)
Xte = Xte[Xtr.columns]
print(f"  egitim {Xtr.shape} test {Xte.shape} ({time.time() - t0:.0f}s)", flush=True)

GECMIS = [c for c in Xtr.columns if c.startswith("h_")]
print(f"  atilacak gecmis kolonlari ({len(GECMIS)}): {GECMIS}")

# ------------------------------------------------------------------ rejimler
ilk = tr.groupby("tanim").tarih.min()
soguk = (~te.tanim.isin(set(tr.tanim))).values
kuyruk = (~soguk) & (te.tanim.map(ilk) >= pd.Timestamp("2026-03-26")).values
cek = ~soguk & ~kuyruk
REJIM = [("soguk", soguk), ("kuyruk", kuyruk), ("cekirdek", cek)]

A6 = np.log1p(pd.read_csv(os.path.join(KOK, "submissions/tuketim_m6_ikiyon.csv")).tuketim.values)
SS = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))

PK = dict(objective="huber", alpha=2.0, lambda_l2=20.0)
TUR = 170


def egit(Xa, ya, Xb):
    ds = lgb.Dataset(Xa, ya)
    tp = []
    for s in TOHUMLAR:
        p = dict(VARSAYILAN)
        p.update(PK)
        p.update(seed=s, bagging_seed=s + 1, feature_fraction_seed=s + 2)
        tp.append(lgb.train(p, ds, TUR).predict(Xb))
        print(f"    tohum {s} ({time.time() - t0:.0f}s)", flush=True)
    return np.mean(tp, axis=0)


def capala_yaz(lg, ad, dosya):
    lg2 = lg.copy()
    rap = {}
    for nm, m in REJIM:
        d = float(A6[m].mean() - lg[m].mean())
        lg2[m] = lg[m] + d
        rap[nm] = dict(satir=int(m.sum()), kaydirma=d)
        print(f"    capa {nm}: {d:+.4f}")
    y = np.clip(np.expm1(lg2), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    yol = os.path.join(KOK, "submissions", dosya)
    out.to_csv(yol, index=False)
    kapi = dict(
        satir=len(out),
        id_birebir=bool((out.id.values == SS.iloc[:, 0].values).all()),
        nan=int(out.tuketim.isna().sum()),
        negatif=int((out.tuketim < 0).sum()),
    )
    assert (
        kapi["satir"] == 714688 and kapi["id_birebir"] and not kapi["nan"] and not kapi["negatif"]
    )
    print(f"  YAZILDI {yol}  kapi={kapi}  ({time.time() - t0:.0f}s)", flush=True)
    return dict(ad=ad, dosya=dosya, capa=rap, kapi=kapi)


rapor = []

print("\n=== A) AMNEZIK (gecmis kolonlari yok) ===", flush=True)
lg = egit(Xtr.drop(columns=GECMIS), ytr, Xte.drop(columns=GECMIS))
rapor.append(capala_yaz(lg, "amnezik", "tuketim_y31_amnezik.csv"))

print("\n=== B) KAPASITE OFSETLI HEDEF ===", flush=True)
ofs_tr = np.log1p(Xtr.guc.to_numpy(dtype=float))
ofs_te = np.log1p(Xte.guc.to_numpy(dtype=float))
lg = egit(Xtr, ytr - ofs_tr, Xte) + ofs_te
rapor.append(capala_yaz(lg, "kapasite", "tuketim_y32_kapasite.csv"))

json.dump(rapor, open(os.path.join(BURA, "y2_yapisal.json"), "w"), indent=1)
print(f"\nTAMAM ({time.time() - t0:.0f}s)")
