"""T2 -- ESLESTIRILMIS uc model: TABAN / TABAN+TURIZM / TABAN+TURIZM+TATIL.

Neden esleştirilmis (paired)?
-----------------------------
Tek basina "m30 + turizm" modeli yazsaydik, uretilen yon f = L_model - L_m6'nin
buyuk kismi TURIZMDEN degil, "baska bir GBM kosusu"ndan gelirdi ve m4 ekseniyle
0,5-0,9 kosinus verirdi -- secim olcutu (|kos| <= 0,20) daha basta duserdi.
Bu yuzden AYNI veri, AYNI tohum, AYNI tur sayisiyla ucu birden egitilir ve
adaylar FARK olarak alinir:
    t1 = pred(TABAN+TURIZM) - pred(TABAN)
    t2 = pred(TABAN+TURIZM+TATIL) - pred(TABAN)
Bu, turizm bilgisinin MARJINAL katkisidir; genel model gurultusu birinci
mertebeden dusar. Buyukluk sonra olceklenir (harman katsayisi zaten LB'de
aranir); ham buyukluk t1_turizm.json'da RAPORLANIR.

SOGUK ODAK: egitimde soguk satirlara agirlik 3, cunku soguk trafonun elindeki
tek seviye bilgisi 2026-03-31'de biten bir KIS ozetidir; turizm kolonlarinin
tasiyabilecegi bilgi orada yogunlasir.
"""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))
sys.path.insert(0, os.path.join(KOK, "src"))
sys.path.insert(0, BURA)

import z1_ortak as Z  # noqa: E402
from m30_ozellik import kur, yukle_ham  # noqa: E402
from m33_durust import VARSAYILAN, hizala  # noqa: E402
from t2_turizm import TH_KOL, TZ_KOL, Turizm  # noqa: E402

from gridup.turkish import join_key  # noqa: E402

KESIM = "2026-03-31"
KESIMLER = ["2025-03-31", "2025-05-31", "2025-07-31", "2025-09-30", "2025-11-30"]
TOHUM = (7, 17)
HUB = dict(objective="huber", alpha=2.0, lambda_l2=20.0)
L1 = dict(objective="l1")
TUR_HUB, TUR_L1 = 185, 520
SOGUK_AGIRLIK = 3.0
Q_HEDEF = 0.06

t0 = time.time()
V = dict(VARSAYILAN)
V["num_threads"] = int(os.environ.get("IPLIK", "14"))
TUR = Turizm()


def anahtarla(d):
    p = d.lokasyon.str.split(">")
    d["il_key"] = p.str[0].str.strip().map(join_key)
    d["ilce_key"] = p.str[-1].str.strip().map(join_key)
    return d


tr, te = yukle_ham()
anahtarla(tr)
anahtarla(te)
print(f"yuklendi tr={len(tr):,} te={len(te):,} ({time.time() - t0:.0f}s)", flush=True)


def zengin(X, il_key, ilce_key, tarih):
    kol = TUR.kolonlar(il_key, ilce_key, tarih, X.tatil.to_numpy())
    return pd.concat([X.reset_index(drop=True), pd.DataFrame(kol)], axis=1)


Xs, ys = [], []
for k in KESIMLER:
    kk = pd.Timestamp(k)
    gec = tr[tr.tarih <= kk]
    hed = tr[(tr.tarih > kk) & (tr.tarih <= kk + pd.DateOffset(months=4))]
    X = kur(gec, hed, k, set(gec.tanim))
    X = zengin(X, hed.il_key.to_numpy(), hed.ilce_key.to_numpy(), hed.tarih.to_numpy())
    Xs.append(X)
    ys.append(np.log1p(hed.tuketim.to_numpy()))
    print(f"  kesim {k}: {len(X):,} satir ({time.time() - t0:.0f}s)", flush=True)
Xtr = pd.concat(Xs, ignore_index=True)
ytr = np.concatenate(ys)
del Xs, ys

Xte = kur(tr, te, KESIM, set(tr.tanim))
Xte = zengin(Xte, te.il_key.to_numpy(), te.ilce_key.to_numpy(), te.tarih.to_numpy())
Xtr, Xte = hizala(Xtr, Xte)
Xte = Xte[Xtr.columns]
w = 1.0 + (SOGUK_AGIRLIK - 1.0) * Xtr.soguk.to_numpy(dtype=float)
print(
    f"  EGITIM {Xtr.shape} | TEST {Xte.shape} | soguk egt %{100 * Xtr.soguk.mean():.1f} "
    f"test %{100 * Xte.soguk.mean():.1f} ({time.time() - t0:.0f}s)",
    flush=True,
)

nan_rap = {c: float(Xte[c].isna().mean()) for c in TZ_KOL + TH_KOL if c in Xte.columns}
print("TEST turizm kolon NaN payi:", json.dumps({k: round(v, 4) for k, v in nan_rap.items()}))

VARYANT = {
    "taban": [c for c in Xtr.columns if c not in TZ_KOL + TH_KOL],
    "turizm": [c for c in Xtr.columns if c not in TH_KOL],
    "tatil": list(Xtr.columns),
}
tahmin, onem = {}, {}
for ad, kols in VARYANT.items():
    ds = lgb.Dataset(Xtr[kols], ytr, weight=w, free_raw_data=False)
    acc, imp = [], None
    for nm, pk, tur in [("huber", HUB, TUR_HUB), ("l1", L1, TUR_L1)]:
        for s in TOHUM:
            p = dict(V)
            p.update(pk)
            p.update(seed=s, bagging_seed=s + 1, feature_fraction_seed=s + 2)
            m = lgb.train(p, ds, tur)
            acc.append(m.predict(Xte[kols]))
            g = pd.Series(m.feature_importance("gain"), index=kols)
            imp = g if imp is None else imp + g
            print(f"  {ad}/{nm}/{s} bitti ({time.time() - t0:.0f}s)", flush=True)
    tahmin[ad] = np.mean(acc, axis=0)
    onem[ad] = (imp / imp.sum()).sort_values(ascending=False)
    del ds

np.save(os.path.join(BURA, "t2_tahmin.npy"), np.array([tahmin[a] for a in VARYANT]))

# --------------------------------------------------------------------- adaylar
tr2, te2 = Z.yukle()
msk = Z.maskeler(tr2, te2)
A6 = Z.taban()
assert (te2.id.values == te.id.values).all()

rap = dict(
    kesimler=KESIMLER,
    egitim_satir=int(len(Xtr)),
    ozellik=int(Xtr.shape[1]),
    soguk_agirlik=SOGUK_AGIRLIK,
    tur=[TUR_HUB, TUR_L1],
    tohum=list(TOHUM),
    test_turizm_nan=nan_rap,
    turizm_kazanim_payi={
        a: float(onem[a][[c for c in TZ_KOL + TH_KOL if c in onem[a].index]].sum()) for a in VARYANT
    },
    en_onemli_10={a: {k: round(float(v), 4) for k, v in onem[a].head(10).items()} for a in VARYANT},
    adaylar={},
)

for ad, dosya in [("t1", "tuketim_t1_turizm.csv"), ("t2", "tuketim_t2_tatil.csv")]:
    kaynak = "turizm" if ad == "t1" else "tatil"
    d = tahmin[kaynak] - tahmin["taban"]
    d = np.where(np.isfinite(d), d, 0.0)
    Qham = float((d**2).mean())
    s = float(np.sqrt(Q_HEDEF / Qham)) if Qham > 0 else 0.0
    r = Z.bitir(A6 + s * d, te2, msk, A6, dosya, kirp=2.0)
    r["ham_delta_Q"] = Qham
    r["ham_delta_rms"] = float(np.sqrt(Qham))
    r["olcek"] = s
    r["ham_delta_rejim_payi"] = {
        k: float((d[msk[k]] ** 2).sum() / max(1e-30, (d**2).sum()))
        for k in ("soguk", "kuyruk", "cekirdek")
    }
    r["ham_delta_ay_ort"] = {
        str(int(m)): round(float(d[(te2.tarih.dt.month == m).to_numpy()].mean()), 5)
        for m in sorted(te2.tarih.dt.month.unique())
    }
    rap["adaylar"][ad] = r
    print(f"  {ad}: ham delta rms={np.sqrt(Qham):.4f} olcek={s:.2f}", flush=True)

# tam model yonlerinin (fark degil) m6'ya gore Q'su -- kiyas icin
rap["tam_model_Q"] = {a: float(((tahmin[a] - A6) ** 2).mean()) for a in VARYANT}
json.dump(rap, open(os.path.join(BURA, "t2_model.json"), "w"), indent=1)
print(f"BITTI ({time.time() - t0:.0f}s)")
