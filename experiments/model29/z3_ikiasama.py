"""Z3 -- IKI ASAMALI: P(olu gun) x kosullu seviye.

Uretim hatti (m1/m4/y31/y32) tek bir REGRESYONLA E[log1p(y)] kestiriyor.
Ama hedef dagilim ACIK BICIMDE KARISIM: satirlarin ~%5'i tam sifir, gerisi
genis bir surekli seviye. Tek regresyon bu iki bileseni ayni agacta
harmanladigi icin sifir kutlesini "seviyeyi asagi cekerek" temsil eder --
yani AKTIF gunlerde asagi yanli, OLU gunlerde yukari yanli olur.

Dogru ayristirma (MSLE = log uzayinda L2 oldugu icin kosullu beklenti):
    E[L] = (1-p) * E[L | aktif] + p * E[L | olu]
    p    : ikili siniflandirici           (tuketim < 1 olasiligi)
    E[L|aktif] : YALNIZ aktif satirlarda egitilmis regresyon
    E[L|olu]   : olu satirlarin ampirik log1p ortalamasi (~0)

Bu bir GBM'dir ama KAYIP GEOMETRISI ve hedef ayristirmasi farklidir; hatalari
yapisal olarak baska satirlarda birikir. Hicbir yerde gurultu yoktur.

Not: siniflandirici, uretim hattinin hic modellemedigi bir seyi -- olme
OLASILIGINI -- ogrenir; regresyon ise sifirlardan arindirilmis, cok daha
homojen bir hedefe bakar.
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
import z1_ortak as Z  # noqa: E402
from m30_ozellik import kur, yukle_ham  # noqa: E402
from m33_durust import VARSAYILAN, hizala  # noqa: E402

KESIM = "2026-03-31"
KESIMLER = [
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
ESIK = 1.0  # "olu gun" tanimi: tuketim < 1 kWh

t0 = time.time()
tr, te = yukle_ham()  # idnum / ly kolonlari m30 tezgahi icin sart
tr["L"] = tr.ly
msk = Z.maskeler(tr, te)
A6 = Z.taban()

Xs, ys = [], []
for k in KESIMLER:
    kk = pd.Timestamp(k)
    son = min(kk + pd.DateOffset(months=4), pd.Timestamp(KESIM))
    gec = tr[tr.tarih <= kk]
    hed = tr[(tr.tarih > kk) & (tr.tarih <= son)]
    if not len(hed):
        continue
    Xs.append(kur(gec, hed, k, set(gec.tanim)))
    ys.append(hed.tuketim.values)
    print(f"  kesim {k}: {len(hed):,} satir ({time.time() - t0:.0f}s)", flush=True)
Xtr = pd.concat(Xs, ignore_index=True)
ytr = np.concatenate(ys)
del Xs, ys
Xte = kur(tr, te, KESIM, set(tr.tanim))
Xtr, Xte = hizala(Xtr, Xte)
Xte = Xte[Xtr.columns]
Ltr = np.log1p(ytr)
olu = ytr < ESIK
M_OLU = float(Ltr[olu].mean())
print(
    f"egitim {Xtr.shape} test {Xte.shape}; olu oran {olu.mean():.4f}, "
    f"E[L|olu]={M_OLU:.4f} ({time.time() - t0:.0f}s)",
    flush=True,
)


def egit(par, X, y, Xp, tur):
    ds = lgb.Dataset(X, y)
    p = []
    for s in TOHUMLAR:
        q = dict(VARSAYILAN)
        q.update(par)
        q.update(seed=s, bagging_seed=s + 1, feature_fraction_seed=s + 2)
        p.append(lgb.train(q, ds, tur).predict(Xp))
        print(f"    tohum {s} ({time.time() - t0:.0f}s)", flush=True)
    return np.mean(p, axis=0)


print("\n=== A) OLME OLASILIGI (ikili) ===", flush=True)
P = egit(
    dict(objective="binary", metric="binary_logloss", learning_rate=0.05, lambda_l2=10.0),
    Xtr,
    olu.astype(np.int8),
    Xte,
    250,
)
P = np.clip(P, 0.0, 1.0)
print(f"  p ortalama {P.mean():.4f} (egitimdeki oran {olu.mean():.4f})", flush=True)

print("\n=== B) KOSULLU SEVIYE (yalniz aktif satirlar) ===", flush=True)
akt = ~olu
RG = egit(
    dict(objective="huber", alpha=2.0, lambda_l2=20.0),
    Xtr[akt],
    Ltr[akt],
    Xte,
    200,
)
print(f"  E[L|aktif] ortalama {RG.mean():.4f}", flush=True)

L = (1.0 - P) * RG + P * M_OLU
L = np.clip(L, 0.0, 14.0)
rap = Z.bitir(L, te, msk, A6, "tuketim_z3_ikiasama.csv", kirp=2.0)
rap["parametreler"] = dict(esik=ESIK, E_L_olu=M_OLU, p_ort=float(P.mean()), tohumlar=list(TOHUMLAR))
np.save(os.path.join(BURA, "z3_p.npy"), P)
np.save(os.path.join(BURA, "z3_rg.npy"), RG)
json.dump(rap, open(os.path.join(BURA, "z3_ikiasama.json"), "w"), indent=1)
print(f"TAMAM ({time.time() - t0:.0f}s)")
