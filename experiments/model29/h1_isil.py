"""H1 -- UZUN ISIL PENCERE + MEVSIM YONU adayi.

Tez (koordinator olcumu): isil pencereler 3/7/14 gun -- hepsi KISA. Bu yuzden
"mevsim yukseliyor mu iniyor mu" degiskeni kurulamiyor. Ayni sicaklikta
SONBAHAR ilkbahardan 0,06-0,18 log birim YUKSEK (histerezis). Test tumuyle
YUKSELEN tarafta (Nisan->Temmuz); egitim iki tarafi da iceriyor, model
ortaliyor -> sistematik hata.

Cozum: m61_hava panelinden 30/60 gunluk isil pencereler + FARK (yon) kolonlari.
m61_hava.py DEGISTIRILMEZ; kolonlar burada, ayni (ilce_key, tarih) ekseninde
turetilir ve zenginlestir() ciktisina eklenir.

Eklenen 11 kolon (hepsi yalniz dis hava panelinden; hedef sizintisi YOK):
  l_sic_ort30/60, l_cdd22_ort30/60, l_hdd18_ort30/60
  l_sic_yon    = sicaklik_ort_ort14 - l_sic_ort60
  l_cdd22_yon  = cdd22_ort14        - l_cdd22_ort60
  l_hdd18_yon  = hdd18_ort14        - l_hdd18_ort60
  l_sic_yon30  = l_sic_ort30        - l_sic_ort60
  l_sic_sapma60= sicaklik_ort       - l_sic_ort60

Panel 2024-06-01'de basliyor -> 60 gunluk pencere 2024-07-30'dan itibaren
DOLU; en erken egitim kesimi 2025-03-31. Yani her blokta dolu.

Cikti:
  submissions/tuketim_h1_isil_ham.csv   (ham model ciktisi)
  submissions/tuketim_h1_isil.csv       (curuk bilesen temizlenmis + rejim capasi)
  h1_isil.json
Ayrica MARJINAL olcum: ayni matris, l_* kolonlari sabitlenmis -> Q(uzun-kisa).
"""

import gc
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)

import m61_hava as h  # noqa: E402
from m30_ozellik import KOK, kur  # noqa: E402
from m33_durust import VARSAYILAN as V0  # noqa: E402
from m33_durust import hizala  # noqa: E402

KESIM = "2026-03-31"
AILE = ("A", "C", "G", "E")
TUR_HUB = int(sys.argv[1]) if len(sys.argv) > 1 else 200
TUR_L1 = int(sys.argv[2]) if len(sys.argv) > 2 else 350
V = dict(V0)
V["num_threads"] = int(os.environ.get("IPLIK", "14"))
HUB = dict(objective="huber", alpha=2.0, lambda_l2=20.0)
L1 = dict(objective="l1")

# ------------------------------------------------------------------ uzun pencere


def uzun_panel(gun):
    """(ilce_key, tarih) ekseninde 30/60 gunluk isil pencereler + yon farklari."""
    g = gun.reset_index().sort_values(["ilce_key", "tarih"])
    gr = g.groupby("ilce_key", observed=True)
    y = pd.DataFrame(index=g.index)
    ESLEME = [("sicaklik_ort", "sic"), ("cdd22", "cdd22"), ("hdd18", "hdd18")]
    for kol, kisa in ESLEME:
        for p in (30, 60):
            y[f"l_{kisa}_ort{p}"] = gr[kol].transform(
                lambda s, _p=p: s.rolling(_p, min_periods=5).mean()
            )
    y["l_sic_yon"] = g["sicaklik_ort_ort14"].to_numpy() - y["l_sic_ort60"].to_numpy()
    y["l_cdd22_yon"] = g["cdd22_ort14"].to_numpy() - y["l_cdd22_ort60"].to_numpy()
    y["l_hdd18_yon"] = g["hdd18_ort14"].to_numpy() - y["l_hdd18_ort60"].to_numpy()
    y["l_sic_yon30"] = y["l_sic_ort30"].to_numpy() - y["l_sic_ort60"].to_numpy()
    y["l_sic_sapma60"] = g["sicaklik_ort"].to_numpy() - y["l_sic_ort60"].to_numpy()
    y.index = pd.MultiIndex.from_arrays([g.ilce_key.to_numpy(), g.tarih.to_numpy()])
    return y.astype(np.float32)


L_KOL = None


def uzun_ekle(X, meta, up):
    global L_KOL
    L_KOL = list(up.columns)
    idx = pd.MultiIndex.from_arrays([meta.ilce_key.to_numpy(), meta.tarih.to_numpy()])
    d = up.reindex(idx)
    for c in L_KOL:
        X[c] = d[c].to_numpy(dtype=np.float32)
    return X


# ------------------------------------------------------------------ matris

t0 = time.time()
o = h.ortam()
tr = o["tr"]
te = o["te"]
UP = uzun_panel(o["gun"])
print(f"  uzun panel {UP.shape} ({time.time() - t0:.0f}s)", flush=True)

Xs, ys = [], []
for k in h.AY:
    r = h.baz(k, KESIM)
    if r is None:
        continue
    Xs.append(uzun_ekle(h.zenginlestir(r[0], r[2], r[3], AILE), r[2], UP))
    ys.append(r[1])
    print(f"  kesim {k}: {len(r[0]):,} ({time.time() - t0:.0f}s)", flush=True)
Xtr = pd.concat(Xs, ignore_index=True)
ytr = np.concatenate(ys)
del Xs, ys
gc.collect()

Xte = kur(tr, te, KESIM, set(tr.tanim))
meta = pd.DataFrame(
    {
        "ilce_key": te.ilce_key.to_numpy(),
        "il_key": te.il_key.to_numpy(),
        "tanim": te.tanim.to_numpy(),
        "tarih": te.tarih.to_numpy(),
    }
)
meta["ilce_key"] = meta.ilce_key.astype(object)
meta["il_key"] = meta.il_key.astype(object)
meta["ay"] = meta.tarih.dt.month.astype("int64")
egt = h.trafo_egim(tr, o["gun"], KESIM)
Xte = uzun_ekle(h.zenginlestir(Xte, meta, egt, AILE), meta, UP)
Xtr, Xte = hizala(Xtr, Xte)
Xte = Xte[Xtr.columns]
nan_tr = {c: float(Xtr[c].isna().mean()) for c in L_KOL}
nan_te = {c: float(Xte[c].isna().mean()) for c in L_KOL}
print(f"  TEST {len(Xte):,} satir, {Xte.shape[1]} ozellik ({time.time() - t0:.0f}s)", flush=True)
print(f"  l_* NaN egitim maks {max(nan_tr.values()):.5f}  test maks {max(nan_te.values()):.5f}")

# ------------------------------------------------------------------ egitim


def egit(Xd, pk, tur, tohumlar):
    ds = lgb.Dataset(Xd, ytr, params={"feature_pre_filter": False})
    acc = []
    for s in tohumlar:
        p = dict(V)
        p.update(pk)
        p.update(seed=s, bagging_seed=s + 1, feature_fraction_seed=s + 2)
        acc.append(lgb.train(p, ds, tur).predict(Xte))
        print(f"    tohum {s} bitti ({time.time() - t0:.0f}s)", flush=True)
    del ds
    gc.collect()
    return acc


print("  == UZUN huber", flush=True)
ph = egit(Xtr, HUB, TUR_HUB, (7, 17, 27))
print("  == UZUN l1", flush=True)
pl = egit(Xtr, L1, TUR_L1, (7, 17, 27))
lg = (np.mean(ph, axis=0) + np.mean(pl, axis=0)) / 2
np.save(os.path.join(BURA, "h1_p_uzun_hub2.npy"), np.mean(ph[:2], axis=0))
np.save(os.path.join(BURA, "h1_p_uzun.npy"), lg)

# marjinal: l_* kolonlarini SABITLE (sabit kolon LightGBM'de bolme uretmez)
YED = {c: Xtr[c].copy() for c in L_KOL}
YEDT = {c: Xte[c].copy() for c in L_KOL}
for c in L_KOL:
    Xtr[c] = np.zeros(len(Xtr), dtype=np.int8)
    Xte[c] = np.zeros(len(Xte), dtype=np.int8)
print("  == KISA (l_* sabit) huber", flush=True)
pk2 = egit(Xtr, HUB, TUR_HUB, (7, 17))
kisa = np.mean(pk2, axis=0)
np.save(os.path.join(BURA, "h1_p_kisa_hub2.npy"), kisa)
for c in L_KOL:
    Xtr[c] = YED[c]
    Xte[c] = YEDT[c]

d = np.mean(ph[:2], axis=0) - kisa
Q_marj = float((d**2).mean())
z = d / np.sqrt(Q_marj)
print(f"\n  MARJINAL (uzun-kisa, huber 2 tohum) Q={Q_marj:.6f} kurtoz={(z**4).mean():.1f}")

# ------------------------------------------------------------------ aday

S = os.path.join(KOK, "submissions")
y = np.clip(np.expm1(lg), 0.0, None)
pd.DataFrame({"id": te.id.values, "tuketim": y}).to_csv(
    os.path.join(S, "tuketim_h1_isil_ham.csv"), index=False
)
print(f"  ham yazildi ({time.time() - t0:.0f}s)", flush=True)

from y1_temizle import temizle  # noqa: E402

rap_t = temizle("tuketim_h1_isil_ham.csv", "tuketim_h1_isil.csv")

from y1_olcum import olc  # noqa: E402

r = olc("tuketim_h1_isil.csv")
print(json.dumps({k: v for k, v in r.items() if k != "kapi"}, indent=1))
print("KAPI:", json.dumps(r["kapi"]))

json.dump(
    dict(
        marjinal_Q=Q_marj,
        marjinal_kurtoz=float((z**4).mean()),
        l_kolon=L_KOL,
        nan_egitim=nan_tr,
        nan_test=nan_te,
        tur=[TUR_HUB, TUR_L1],
        ozellik=int(Xtr.shape[1]),
        temizle=rap_t,
        olcum=r,
    ),
    open(os.path.join(BURA, "h1_isil.json"), "w"),
    indent=1,
)
print(f"YAZILDI h1_isil.json ({time.time() - t0:.0f}s)")
