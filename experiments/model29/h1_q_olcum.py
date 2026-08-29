"""H1 ADIM 1 -- HAFTA GUNU yonunun Q'sunu OLC (aday uretmeden once).

Soru: m30_ozellik.kur() zaten `hgun` / `hafta_sonu` uretiyor. Bu kolonlari
ATARSAK tahminler ne kadar degisiyor?  Q = ort( (koy - at)^2 ) log1p uzayinda.

KAPI: Q >= 0,01 degilse yon LB'de gurultude bogulur -> aday URETME.

Ayrica PLASEBO olcumu: ayni ozellik setiyle, sadece TOHUM degisince olusan
fark. Sinyalin Q'su plasebonun Q'suna yakinsa "fark" model kararsizligidir,
hafta gunu bilgisi degil.

Kolonlar yerinde SABITLENEREK (0.0) atilir -- sabit kolon LightGBM'de hicbir
bolme uretmez, yani dusurmekle esdegerdir; ama 3,3M satirlik matrisin kopyasi
cikarilmadigi icin bellek patlamaz.

Kollar:
  koy_a   : tam matris, tohum (7,17)
  koy_b   : tam matris, tohum (37,47)      <- PLASEBO esi
  at      : hgun+hafta_sonu sabit, tohum (7,17)
  attak   : hgun+hafta_sonu+tatil+ayin_gunu sabit, tohum (7,17)
  arti    : tam matris + GENIS TAKVIM demeti, tohum (7,17)

Cikti: h1_q_olcum.json + h1_p_<kol>.npy (test tahminleri, log1p uzayinda)
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
from m30_ozellik import TATIL, kur  # noqa: E402
from m33_durust import VARSAYILAN as V0  # noqa: E402
from m33_durust import hizala  # noqa: E402

KESIM = "2026-03-31"
AILE = ("A", "C", "G", "E")
TUR_HUB = 200
V = dict(V0)
V["num_threads"] = int(os.environ.get("IPLIK", "14"))
HUB = dict(objective="huber", alpha=2.0, lambda_l2=20.0)

HG = ["hgun", "hafta_sonu"]
TAKVIM_TABAN = ["hgun", "hafta_sonu", "tatil", "ayin_gunu"]

# ------------------------------------------------------------------ genis takvim

_TATIL_GUN = np.sort(np.array(sorted(TATIL), dtype="datetime64[D]"))
RAMAZAN = [("2025-03-01", "2025-03-30"), ("2026-02-18", "2026-03-19")]


def genis_takvim(tarih):
    """tk_* demeti -- yalnizca takvimden turer, sizinti yok."""
    t = pd.DatetimeIndex(tarih)
    g = t.values.astype("datetime64[D]")
    n = len(_TATIL_GUN)
    i = np.searchsorted(_TATIL_GUN, g)
    d_ileri = (_TATIL_GUN[np.minimum(i, n - 1)] - g).astype("timedelta64[D]").astype(np.int64)
    d_geri = (g - _TATIL_GUN[np.maximum(i - 1, 0)]).astype("timedelta64[D]").astype(np.int64)
    ileri = np.where(i < n, d_ileri, 999).astype(np.float32)
    geri = np.where(i > 0, d_geri, 999).astype(np.float32)
    hg = np.asarray(t.dayofweek)
    tatil = np.isin(g, _TATIL_GUN)
    ram = np.zeros(len(t), dtype=np.int8)
    for a, b in RAMAZAN:
        ram |= np.asarray((t >= a) & (t <= b)).astype(np.int8)
    ay = np.asarray(t.month)
    d = {
        "tk_yilin_gunu": np.asarray(t.dayofyear).astype(np.float32),
        "tk_hafta_no": np.asarray(t.isocalendar().week).astype(np.float32),
        "tk_ceyrek": np.asarray(t.quarter).astype(np.float32),
        "tk_pazar": (hg == 6).astype(np.int8),
        "tk_cumartesi": (hg == 5).astype(np.int8),
        "tk_pazartesi": (hg == 0).astype(np.int8),
        "tk_cuma": (hg == 4).astype(np.int8),
        "tk_tatil_ileri": np.minimum(ileri, 30.0),
        "tk_tatil_geri": np.minimum(geri, 30.0),
        "tk_tatil_mesafe": np.minimum(np.minimum(ileri, geri), 30.0),
        "tk_arefe": (ileri == 1).astype(np.int8),
        "tk_tatil_ertesi": (geri == 1).astype(np.int8),
        "tk_kopru": ((hg < 5) & ~tatil & ((ileri <= 1) | (geri <= 1))).astype(np.int8),
        "tk_ay_basi": (np.asarray(t.day) <= 3).astype(np.int8),
        "tk_ay_sonu": (np.asarray(t.day) >= np.asarray(t.days_in_month) - 2).astype(np.int8),
        "tk_ramazan": ram,
        "tk_okul": (~np.isin(ay, (7, 8))).astype(np.int8),
        "tk_uzun_tatil_bloku": ((ileri <= 2) | (geri <= 2)).astype(np.int8),
    }
    return pd.DataFrame(d)


TK_KOL = list(genis_takvim(pd.to_datetime(["2025-01-01"])).columns)

# ------------------------------------------------------------------ matris

t0 = time.time()
o = h.ortam()
tr = o["tr"]
te = o["te"]
Xs, ys, tarihler = [], [], []
for k in h.AY:
    r = h.baz(k, KESIM)
    if r is None:
        continue
    Xs.append(h.zenginlestir(r[0], r[2], r[3], AILE))
    ys.append(r[1])
    tarihler.append(r[2].tarih.to_numpy())
    print(f"  kesim {k}: {len(r[0]):,} ({time.time() - t0:.0f}s)", flush=True)
Xtr = pd.concat(Xs, ignore_index=True)
ytr = np.concatenate(ys)
TR_TARIH = np.concatenate(tarihler)
del Xs, ys, tarihler
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
Xte = h.zenginlestir(Xte, meta, egt, AILE)
Xtr, Xte = hizala(Xtr, Xte)
Xte = Xte[Xtr.columns]
TE_TARIH = te.tarih.to_numpy()
print(f"  TEST {len(Xte):,} satir, {Xte.shape[1]} ozellik ({time.time() - t0:.0f}s)", flush=True)

# ------------------------------------------------------------------ kollar

YEDEK_TR = {c: Xtr[c].copy() for c in TAKVIM_TABAN}
YEDEK_TE = {c: Xte[c].copy() for c in TAKVIM_TABAN}


def sabitle(kolonlar):
    for c in kolonlar:
        Xtr[c] = np.zeros(len(Xtr), dtype=np.int8)
        Xte[c] = np.zeros(len(Xte), dtype=np.int8)


def geri_al():
    for c in TAKVIM_TABAN:
        Xtr[c] = YEDEK_TR[c]
        Xte[c] = YEDEK_TE[c]


def tk_ekle():
    a = genis_takvim(TR_TARIH)
    b = genis_takvim(TE_TARIH)
    for c in TK_KOL:
        Xtr[c] = a[c].to_numpy()
        Xte[c] = b[c].to_numpy()
    del a, b
    gc.collect()


def tk_sil():
    for c in TK_KOL:
        del Xtr[c]
        del Xte[c]
    gc.collect()


def egit(tohumlar):
    ds = lgb.Dataset(Xtr, ytr, params={"feature_pre_filter": False})
    acc = []
    for s in tohumlar:
        p = dict(V)
        p.update(HUB)
        p.update(seed=s, bagging_seed=s + 1, feature_fraction_seed=s + 2)
        acc.append(lgb.train(p, ds, TUR_HUB).predict(Xte))
        print(f"    tohum {s} bitti ({time.time() - t0:.0f}s)", flush=True)
    del ds
    gc.collect()
    return np.mean(acc, axis=0)


KOLLAR = [
    ("koy_a", None, (7, 17)),
    ("koy_b", None, (37, 47)),
    ("at", HG, (7, 17)),
    ("attak", TAKVIM_TABAN, (7, 17)),
    ("arti", "TK", (7, 17)),
]

P = {}
for ad, kol, toh in KOLLAR:
    print(f"  == kol {ad} ({time.time() - t0:.0f}s)", flush=True)
    if kol == "TK":
        tk_ekle()
    elif kol:
        sabitle(kol)
    P[ad] = egit(toh)
    np.save(os.path.join(BURA, f"h1_p_{ad}.npy"), P[ad])
    if kol == "TK":
        tk_sil()
    elif kol:
        geri_al()

# ------------------------------------------------------------------ olcum


def istat(f):
    Q = float((f**2).mean())
    if Q <= 0:
        return dict(Q=0.0)
    z = f / np.sqrt(Q)
    return dict(
        Q=Q,
        rms=float(np.sqrt(Q)),
        kurtoz=float((z**4).mean()),
        maks=float(np.abs(f).max()),
        p999=float(np.quantile(np.abs(f), 0.999)),
        en_kotu1_pay=float((np.sort(f**2)[-len(f) // 100 :]).sum() / (f**2).sum()),
    )


FARK = [
    ("hafta_gunu_sinyali", "koy_a", "at"),
    ("takvim_taban_sinyali", "koy_a", "attak"),
    ("genis_takvim_sinyali", "arti", "koy_a"),
    ("PLASEBO_tohum", "koy_a", "koy_b"),
]
rap = {"tur": TUR_HUB, "ozellik": int(Xtr.shape[1]), "tk_kolon": len(TK_KOL), "fark": {}}
print("\n===== Q OLCUMU (log1p uzayinda) =====")
for ad, a, b in FARK:
    s = istat(P[a] - P[b])
    rap["fark"][ad] = s
    print(
        f"  {ad:24s} Q={s['Q']:.6f}  rms={s['rms']:.4f}  kurtoz={s['kurtoz']:.1f}  "
        f"maks={s['maks']:.3f}  p999={s['p999']:.3f}  enkotu%1pay={s['en_kotu1_pay']:.3f}"
    )
Qs = rap["fark"]["hafta_gunu_sinyali"]["Q"]
Qp = rap["fark"]["PLASEBO_tohum"]["Q"]
rap["sinyal_gurultu"] = float(Qs / Qp) if Qp > 0 else None
rap["kapi_Q001"] = bool(Qs >= 0.01)
print(f"\n  sinyal/plasebo Q orani = {rap['sinyal_gurultu']:.2f}")
print(f"  KAPI Q>=0,01 : {'GECTI' if rap['kapi_Q001'] else 'KALDI'}  (Q={Qs:.6f})")
json.dump(rap, open(os.path.join(BURA, "h1_q_olcum.json"), "w"), indent=1)
print(f"YAZILDI h1_q_olcum.json ({time.time() - t0:.0f}s)")
