"""YENI EKSEN AILELERI -- m121'in TARAMADIGI yonler.

m121 su donusum kumesini taradi: ham kolon, {x_sv, x_soguk, x_ufuk, x_ay}
carpimlari, ust10/ust25/alt10 esik kesitleri, kare ve 21 odak kolonunun
ikili carpimlari. Bu betik ORADA OLMAYAN aileleri uretir:

  A  TRAFO/GRUP HEDEF KODLAMASI -- artik ortalamalari DIS BLOKLARDAN
     (guz25 + kis26) hesaplanir, yaz25 blogunda ve testte YALNIZCA
     kimlik kullanilir. SIZINTI YOK: olcum blogu (yaz25) kodlayiciya
     hic girmez.
  B  LOKASYON HIYERARSISI -- il > bolge > ilce > lokasyon duzeyinde
     ve (ilce x ay), (ilce x haftanin gunu), (bolge x soguk) kirilimlari.
  C  TAKVIM -- haftanin gunu kuklalari, tatil kuklalari, ay ici gun ve
     yilin gunu harmonikleri, bunlarin sv/soguk/ufuk etkilesimleri.
  D  HAVA GECIKMELI/KUMULATIF -- (lokasyon, tarih) panelinde 1/2 gun
     gecikme, gecmis 3/7 gun ortalamasi ve ANOMALI (bugun - 7 gun ort).
     m121'de yalnizca hazir *_ort3/7/14 sutunlari vardi, gecikme ve
     anomali yoktu.
  E  UFUK -- parcali dogrusal mentese (24/48/73/98 dugumleri), kova
     kuklalari, log/kare ve etkilesimler. Test 122 gunluk bir ufuk.
  F  GUC/KAPASITE/YAS -- log guc, guc kovalari, yas x yuk faktoru ve
     m121'in ODAK listesinde OLMAYAN kolonlarin ikili carpimlari.
  G  MENTESE (ucgen/spline) -- esik yerine max(x-q,0) ve min(x-q,0).
  H  MEVCUT 40 EKSENIN BIRBIRIYLE CARPIMLARI.

KAPILAR m122'den aynen alindi:
  |rho_s| >= 0.015, Q_dik >= 0.25, rcond kararliligi (1e-5 vs 1e-6, %30),
  plasebo |z| >= 3, tavan dayanmali (|rho_cv| >= 1.95*|rho_s|).

CIKTI: yeni eksenler MEVCUT 40 eksene DIKLESTIRILEREK degerlendirilir;
kumulatif sqrt(sum rho_s^2) dogru hesaplanir. HICBIR GONDERIM YAZILMAZ.
"""

import gc
import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
BURA = os.path.dirname(os.path.abspath(__file__))
TABAN = "tuketim_m6_ikiyon.csv"
HEDEF_SOGUK, CARPAN, TAVAN = 0.222, 0.798, 1.95
RHO_S_ALT = 0.015
QS_ALT, QD_ALT = 0.02, 0.25
AZAMI_MEVCUT = 40  # m122'nin kestigi yer
AZAMI_YENI = 120  # bellek siniri; kesim asil KAPIDAN gelmeli (Kural 64)
KODLAMA_K = 20.0  # hedef kodlamasinda buzme sabiti
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0, buzmeli_r_hat  # noqa: E402

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
IDS = te.id.values


def oku(f):
    d = pd.read_csv(os.path.join(S, f))
    k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
    if not np.array_equal(d.id.values, IDS):
        if len(d) != len(IDS) or d.id.duplicated().any():
            return None
        pos = pd.Index(d.id).get_indexer(IDS)
        if (pos < 0).any():
            return None
        d = d.iloc[pos].reset_index(drop=True)
    return np.log1p(d[k].values.astype(np.float64))


a0 = oku(TABAN)
N = len(a0)
with open(os.path.join(M29, "olculmus_skorlar.json")) as fh:
    SK = json.load(fh)
with open(os.path.join(M29, "m112_durum.json")) as fh:
    DUR = json.load(fh)
V, L = [], []
for f, Pj in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is None or len(v) != N:
        continue
    dd = v - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - Pj * Pj) / 2)
for f, Lj in EK_MODEL.items():
    V.append(oku(f) - a0)
    L.append(Lj)
for o in DUR.get("olcumler", []):
    dd = oku(o["dosya"]) - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - o["skor"] ** 2) / 2)
V, L = np.array(V).T, np.array(L)
G = (V.T @ V) / N
Gi = np.linalg.pinv(G, rcond=1e-6)
GI5 = np.linalg.pinv(G, rcond=1e-5)
r_hat, gercek, kL = buzmeli_r_hat(V, L, G, N)
MSE_OPT = M0 - gercek
print(f"saf optimum {np.sqrt(MSE_OPT):.6f}   V: {V.shape}")

# ----------------------------------------------------------------- veri
tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
e = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
SIC_AILE = [(t, aa) for t in (1000, 1001, 1002) for aa in ("cat", "xgb", "lgbm")]


def blok_artik(eg, blok):
    """Blogun (satir sirasi = sicak sonra soguk) artigi ve o satirlarin indeksi."""
    blk = eg[eg._blok == blok]
    sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
    P = [
        np.load(os.path.join(AO, f"{blok}_{t}_{aa}_uretim.npy")).astype(np.float64)
        for t, aa in SIC_AILE
        if os.path.exists(os.path.join(AO, f"{blok}_{t}_{aa}_uretim.npy"))
    ]
    z = np.load(os.path.join(DN, f"soguk_tahmin_{blok}.npz"))
    idx = np.concatenate([sic.index.values, sog.index.values])
    pb = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])
    return idx, np.log1p(eg.loc[idx].tuketim.values.astype(np.float64)) - pb


idx, rb = blok_artik(e, "yaz25")
bf = e.loc[idx].copy()
sgm = bf.soguk_mu.values.astype(np.float64)
ww = np.where(sgm == 1, HEDEF_SOGUK / sgm.mean(), (1 - HEDEF_SOGUK) / (1 - sgm.mean()))
ww = ww / ww.mean()
m0b = float((ww * rb * rb).mean())
NB = len(rb)

# DIS BLOK artiklari -- hedef kodlamasinin TEK kaynagi (yaz25 girmez)
d_idx, d_r = [], []
for b in ("guz25", "kis26"):
    i2, r2 = blok_artik(e, b)
    d_idx.append(i2)
    d_r.append(r2)
DIS_R = np.concatenate(d_r)
DIS = e.loc[np.concatenate(d_idx)].copy()
DIS["_blk_kis"] = (DIS._blok == "kis26").to_numpy()
print(f"blok yaz25 {NB:,} satir | dis blok kodlayici {len(DIS_R):,} satir")
del e
gc.collect()


def st(x):
    x = np.asarray(x, dtype=np.float64).copy()
    f = np.isfinite(x)
    if not f.any():
        return None
    x[~f] = np.median(x[f])
    x -= x.mean()
    s = np.sqrt(float((x * x).mean()))
    return x / s if s > 1e-12 else None


svT, svB = st(a0), st(np.log1p(bf.tuketim.values.astype(np.float64)) - rb)
sgT = tp.soguk_mu.values.astype(np.float64)
ufTh, ufBh = tp.ufuk_gun.to_numpy().astype(np.float64), bf.ufuk_gun.to_numpy().astype(np.float64)  # noqa: E501
ufT, ufB = st(ufTh), st(ufBh)
ayT = st(pd.to_datetime(tp.tarih).dt.month.to_numpy().astype(np.float64))
ayB = st(pd.to_datetime(bf.tarih).dt.month.to_numpy().astype(np.float64))
CARP = {"x_sv": (svT, svB), "x_soguk": (sgT, sgm), "x_ufuk": (ufT, ufB), "x_ay": (ayT, ayB)}
ESIK = {"ust10": (0.9, True), "ust25": (0.75, True), "alt10": (0.1, False)}


def kur(ad):
    """m122'nin eksen kurucusu (mevcut 40 ekseni yeniden uretmek icin)."""
    if "*" in ad:
        k1, k2 = ad.split("*", 1)
        if k1 not in tp.columns or k2 not in tp.columns:
            return None, None
        a1, b1 = st(tp[k1].to_numpy()), st(bf[k1].to_numpy())
        a2, b2 = st(tp[k2].to_numpy()), st(bf[k2].to_numpy())
        if a1 is None or a2 is None or b1 is None or b2 is None:
            return None, None
        return st(a1 * a2), st(b1 * b2)
    kol, kip = ad.split(":", 1) if ":" in ad else (ad, "")
    if kol not in tp.columns or kol not in bf.columns:
        return None, None
    xt, xb = tp[kol].to_numpy(), bf[kol].to_numpy()
    if kip in CARP:
        mt, mb = CARP[kip]
        a_, b_ = st(xt), st(xb)
        return (None, None) if a_ is None or b_ is None else (st(a_ * mt), st(b_ * mb))
    if kip in ESIK:
        q, ust = ESIK[kip]
        fv = xt[np.isfinite(xt)]
        if fv.size == 0:
            return None, None
        v_ = np.quantile(fv, q)
        if ust:
            return st((xt > v_).astype(np.float64)), st((xb > v_).astype(np.float64))
        return st((xt < v_).astype(np.float64)), st((xb < v_).astype(np.float64))
    if kip == "kare":
        a_, b_ = st(xt), st(xb)
        return (None, None) if a_ is None else (st(a_**2), st(b_**2))
    return st(xt), st(xb)


# ------------------------------------------------- plasebo permutasyonlari
rng = np.random.default_rng(5)
tn = bf.tanim.values
uqn = pd.unique(tn)
gi = pd.Series(np.arange(len(uqn)), index=uqn)[tn].to_numpy()
PERM = [
    np.argsort(np.argsort(rng.permutation(len(uqn))[gi], kind="stable"), kind="stable")
    for _ in range(20)
]


def olc(xt, xb, onceki):
    """m122 kapilari. Gecerse sozluk, gecmezse (None, sebep) doner."""
    dot = float((r_hat * xt).mean())
    if abs(dot) < RHO_S_ALT * np.sqrt(QS_ALT):  # |rho_s| = |dot|/sqrt(Qs) <= |dot|/sqrt(0.02)
        return None, "dot"
    kor = float((ww * rb * xb).mean()) / np.sqrt(m0b)
    rho_cv = CARPAN * kor
    if abs(rho_cv) < TAVAN * abs(dot):  # |rho_s| >= |dot| oldugu icin gerekli sart
        return None, "tavan_on"
    gur = float(np.std([float((ww * rb[s] * xb).mean()) / np.sqrt(m0b) for s in PERM]))
    if gur < 1e-12 or abs(kor) < 3 * gur:
        return None, "plasebo"
    cc = Gi @ ((V.T @ xt) / N)
    xp0 = xt - V @ cc
    Qs = 1.0 - float((xp0 * xp0).mean())
    if Qs < QS_ALT:
        return None, "Qs"
    rho_s = dot / np.sqrt(Qs)
    if abs(rho_s) < RHO_S_ALT:
        return None, "rho_s"
    cc5 = GI5 @ ((V.T @ xt) / N)
    xp5 = xt - V @ cc5
    Qs5 = 1.0 - float((xp5 * xp5).mean())
    if Qs5 < QS_ALT:
        return None, "Qs5"
    if abs(dot / np.sqrt(Qs5) - rho_s) > 0.3 * abs(rho_s):
        return None, "rcond"
    xp = xp0
    for u in onceki:
        xp = xp - float((xp * u).mean()) * u
    Qd = float((xp * xp).mean())
    if Qd < QD_ALT:
        return None, "Qdik"
    if abs(rho_cv) < TAVAN * abs(rho_s):
        return None, "tavan"
    return (
        dict(
            rho_cv=rho_cv,
            rho_s=rho_s,
            Qd=Qd,
            z=abs(kor) / gur,
            birim=xp / np.sqrt(Qd),
            a=cc / np.sqrt(Qs),  # span-ici BIRIM yonun V-koordinati (a'Ga = 1)
        ),
        "",
    )


# ================================================== 1) MEVCUT 40 EKSEN
with open(os.path.join(M29, "m121_derin_tarama.json")) as fh:
    TARAMA = json.load(fh)
ONCEKI, MEVCUT, A_MEVCUT = [], [], []
for kayit in TARAMA:
    if len(MEVCUT) >= AZAMI_MEVCUT:
        break
    xt, xb = kur(kayit["eksen"])
    if xt is None or xb is None:
        continue
    s, _ = olc(xt, xb, ONCEKI)
    if s is None:
        continue
    ONCEKI.append(s["birim"])
    A_MEVCUT.append(s["a"])
    MEVCUT.append(dict(eksen=kayit["eksen"], rho_s=s["rho_s"], rho_cv=s["rho_cv"], Qd=s["Qd"]))
S2_MEVCUT = float(sum(m["rho_s"] ** 2 for m in MEVCUT))
print(f"\nMEVCUT {len(MEVCUT)} eksen yeniden kuruldu: sqrt(sum rho_s^2) = {np.sqrt(S2_MEVCUT):.4f}")
MEVCUT_ADLAR = [m["eksen"] for m in MEVCUT]

# ================================================== 2) YENI AILELER
ATLA = {"id", "tanim", "lokasyon", "tarih", "bolge", "_blok", "tuketim", "soguk_mu"}
KOL = [
    c
    for c in tp.columns
    if c not in ATLA and pd.api.types.is_numeric_dtype(tp[c]) and c in bf.columns
]
tarT, tarB = pd.to_datetime(tp.tarih), pd.to_datetime(bf.tarih)


def dizi(df, c):
    """Sutunu float64 numpy'a cevirir (Int64/NA guvenli)."""
    return df[c].to_numpy(dtype=np.float64, na_value=np.nan)


def esle(deger, anahtar_t, anahtar_b):
    """Grup -> deger eslemesi; gorulmeyen grup 0 (bilgi yok)."""
    m = pd.Series(deger.to_numpy(dtype=np.float64), index=deger.index)
    return (
        pd.Series(anahtar_t).map(m).fillna(0.0).to_numpy(dtype=np.float64),
        pd.Series(anahtar_b).map(m).fillna(0.0).to_numpy(dtype=np.float64),
    )


def kod(sutunlar):
    """Sutun listesinden dis/test/blok icin ortak tamsayi grup kodu."""
    kd, kt, kb = None, None, None
    for c in sutunlar:
        vd, vt, vb = DIS[c].to_numpy(), tp[c].to_numpy(), bf[c].to_numpy()
        u = pd.Index(pd.unique(np.concatenate([vd, vt, vb]).astype(str)))
        mp = pd.Series(np.arange(len(u)), index=u)
        cd = mp[pd.Index(vd.astype(str))].to_numpy()
        ct = mp[pd.Index(vt.astype(str))].to_numpy()
        cb = mp[pd.Index(vb.astype(str))].to_numpy()
        n = len(u)
        kd, kt, kb = (cd, ct, cb) if kd is None else (kd * n + cd, kt * n + ct, kb * n + cb)
    return kd, kt, kb


def hedef_kodla(kd, kt, kb, r, k=KODLAMA_K, kip="ort"):
    df = pd.DataFrame({"g": kd, "r": r})
    ag = df.groupby("g")["r"]
    if kip == "ort":
        v = ag.sum() / (ag.count() + k)
    elif kip == "std":
        v = ag.std().fillna(0.0) * (ag.count() / (ag.count() + k))
    else:  # mutlak sapma
        v = df.assign(r=df.r.abs()).groupby("g")["r"].sum() / (ag.count() + k)
    return esle(v, kt, kb)


def aile_a_trafo():
    """A: trafo duzeyi hedef kodlamasi (dis bloklardan)."""
    kd, kt, kb = kod(["tanim"])
    kis = DIS._blk_kis.to_numpy()
    tabanlar = {
        "te_trafo": hedef_kodla(kd, kt, kb, DIS_R),
        "te_trafo_std": hedef_kodla(kd, kt, kb, DIS_R, kip="std"),
        "te_trafo_mut": hedef_kodla(kd, kt, kb, DIS_R, kip="mut"),
        "te_trafo_k5": hedef_kodla(kd, kt, kb, DIS_R, k=5.0),
        "te_trafo_k100": hedef_kodla(kd, kt, kb, DIS_R, k=100.0),
    }
    # yalniz kis26 (en yakin blok) ve blok farki (kayma)
    kis_t, kis_b = hedef_kodla(kd[kis], kt, kb, DIS_R[kis])
    guz_t, guz_b = hedef_kodla(kd[~kis], kt, kb, DIS_R[~kis])
    tabanlar["te_trafo_kis"] = (kis_t, kis_b)
    tabanlar["te_trafo_kayma"] = (kis_t - guz_t, kis_b - guz_b)
    # trafo x haftanin gunu / hafta sonu
    for ek in ("tk_hafta_sonu", "tk_haftanin_gunu"):
        k2 = kod(["tanim", ek])
        tabanlar[f"te_trafo_{ek}"] = hedef_kodla(*k2, DIS_R, k=40.0)
    for ad, (xt, xb) in tabanlar.items():
        yield ad, (lambda xt=xt, xb=xb: (st(xt), st(xb)))
        for kip, (mt, mb) in CARP.items():
            a_, b_ = st(xt), st(xb)
            if a_ is None or b_ is None:
                continue
            yield f"{ad}:{kip}", (lambda a_=a_, b_=b_, mt=mt, mb=mb: (st(a_ * mt), st(b_ * mb)))
    # trafo kodlamasi x ufuk mentesesi (ufuk boyunca sonme)
    xt, xb = tabanlar["te_trafo"]
    for d in (24, 48, 73, 98):
        a_, b_ = st(xt), st(xb)
        yield (
            f"te_trafo:mentese{d}",
            lambda a_=a_, b_=b_, d=d: (
                st(a_ * np.maximum(ufTh - d, 0)),
                st(b_ * np.maximum(ufBh - d, 0)),
            ),
        )


def aile_b_lokasyon():
    """B: lokasyon hiyerarsisi (il > bolge > ilce > lokasyon) kodlamalari."""
    gruplar = {
        "il": ["il_key"],
        "bolge": ["bolge"],
        "ilce": ["ilce_key"],
        "lok": ["lokasyon"],
        "ilce_ay": ["ilce_key", "tk_ay"],
        "ilce_hg": ["ilce_key", "tk_haftanin_gunu"],
        "bolge_soguk": ["bolge", "soguk_mu"],
        "ilce_kova": ["ilce_key", "g_guc_kova"],
        "on2": ["tanim_on2"],
        "on3": ["tanim_on3"],
        "on2_ilce": ["tanim_on2", "ilce_key"],
        "kova": ["g_guc_kova"],
        "kova_ay": ["g_guc_kova", "tk_ay"],
    }
    for ad, sut in gruplar.items():
        if any(c not in tp.columns or c not in DIS.columns for c in sut):
            continue
        kd, kt, kb = kod(sut)
        xt, xb = hedef_kodla(kd, kt, kb, DIS_R, k=200.0)
        yield f"te_{ad}", (lambda xt=xt, xb=xb: (st(xt), st(xb)))
        for kip in ("x_sv", "x_ufuk", "x_soguk"):
            mt, mb = CARP[kip]
            a_, b_ = st(xt), st(xb)
            if a_ is None or b_ is None:
                continue
            yield (
                f"te_{ad}:{kip}",
                lambda a_=a_, b_=b_, mt=mt, mb=mb: (st(a_ * mt), st(b_ * mb)),
            )
        # hiyerarsi ARTIGI: ilce kodlamasi eksi il kodlamasi gibi farklar
    kd, kt, kb = kod(["ilce_key"])
    it, ib = hedef_kodla(kd, kt, kb, DIS_R, k=200.0)
    kd2, kt2, kb2 = kod(["bolge"])
    bt, bb = hedef_kodla(kd2, kt2, kb2, DIS_R, k=200.0)
    yield "te_ilce_eksi_bolge", (lambda: (st(it - bt), st(ib - bb)))
    kd3, kt3, kb3 = kod(["tanim"])
    tt, tb = hedef_kodla(kd3, kt3, kb3, DIS_R)
    yield "te_trafo_eksi_ilce", (lambda: (st(tt - it), st(tb - ib)))


def aile_c_takvim():
    """C: haftanin gunu / tatil kuklalari ve yilin gunu harmonikleri."""
    hgT, hgB = tp.tk_haftanin_gunu.to_numpy(), bf.tk_haftanin_gunu.to_numpy()
    for g in range(7):
        at, ab = (hgT == g).astype(np.float64), (hgB == g).astype(np.float64)
        yield f"hg{g}", (lambda at=at, ab=ab: (st(at), st(ab)))
        for kip in ("x_sv", "x_ufuk"):
            mt, mb = CARP[kip]
            a_, b_ = st(at), st(ab)
            if a_ is None or b_ is None:
                continue
            yield (
                f"hg{g}:{kip}",
                lambda a_=a_, b_=b_, mt=mt, mb=mb: (st(a_ * mt), st(b_ * mb)),
            )
    ikili = [
        ("tatil_mi", tp.tatil_mi.to_numpy(), bf.tatil_mi.to_numpy()),
        ("hafta_sonu", tp.tk_hafta_sonu.to_numpy(), bf.tk_hafta_sonu.to_numpy()),
        ("tatil_vh", tp.tatil_veya_haftasonu.to_numpy(), bf.tatil_veya_haftasonu.to_numpy()),
        ("tatil_yakin", tp.tatil_yakininda.to_numpy(), bf.tatil_yakininda.to_numpy()),
        ("ay_ilk", tp.tk_ayin_ilk_gunu.to_numpy(), bf.tk_ayin_ilk_gunu.to_numpy()),
        ("ay_son", tp.tk_ayin_son_gunu.to_numpy(), bf.tk_ayin_son_gunu.to_numpy()),
        ("ramazan", tp.ramazan_ayi.to_numpy(), bf.ramazan_ayi.to_numpy()),
    ]
    for ad, at, ab in ikili:
        at, ab = at.astype(np.float64), ab.astype(np.float64)
        for kip in ("x_sv", "x_soguk", "x_ufuk"):
            mt, mb = CARP[kip]
            a_, b_ = st(at), st(ab)
            if a_ is None or b_ is None:
                continue
            yield (
                f"tk_{ad}:{kip}",
                lambda a_=a_, b_=b_, mt=mt, mb=mb: (st(a_ * mt), st(b_ * mb)),
            )
    yg_t = tarT.dt.dayofyear.to_numpy().astype(np.float64)
    yg_b = tarB.dt.dayofyear.to_numpy().astype(np.float64)
    gun_t = tp.tk_gun.to_numpy().astype(np.float64)
    gun_b = bf.tk_gun.to_numpy().astype(np.float64)
    for h in (1, 2, 3):
        for f_, adf in ((np.sin, "sin"), (np.cos, "cos")):
            at = f_(2 * np.pi * h * yg_t / 365.25)
            ab = f_(2 * np.pi * h * yg_b / 365.25)
            yield f"yil_{adf}{h}", (lambda at=at, ab=ab: (st(at), st(ab)))
            yield (
                f"yil_{adf}{h}:x_sv",
                lambda at=at, ab=ab: (st(st(at) * svT), st(st(ab) * svB)),
            )
        at = np.sin(2 * np.pi * h * gun_t / 30.44)
        ab = np.sin(2 * np.pi * h * gun_b / 30.44)
        yield f"ayici_sin{h}", (lambda at=at, ab=ab: (st(at), st(ab)))
    for d in (5, 10, 15, 20, 25):
        at, ab = np.maximum(gun_t - d, 0), np.maximum(gun_b - d, 0)
        yield f"ayici_mentese{d}", (lambda at=at, ab=ab: (st(at), st(ab)))


HAVA = [
    "sicaklik_ort",
    "sicaklik_max",
    "sicaklik_min",
    "hissedilen_max",
    "nem_ort",
    "yagis_toplam",
    "ruzgar_max",
    "gunes_radyasyon",
    "gunes_ghi_gunluk",
    "vpd_ort",
    "ciy_ort",
    "toprak_nem_ort",
    "bulut_dusuk_ort",
    "cdd22",
    "cdd24",
    "et0_toplam",
]


def hava_paneli():
    """(lokasyon, tarih) panelinde gecikme / kumulatif / anomali."""
    sut = [c for c in HAVA if c in tp.columns and c in bf.columns]
    pan = pd.concat(
        [tp[["lokasyon", "tarih"] + sut], bf[["lokasyon", "tarih"] + sut]], ignore_index=True
    )
    pan = pan.drop_duplicates(["lokasyon", "tarih"]).sort_values(["lokasyon", "tarih"])
    g = pan.groupby("lokasyon", observed=True)
    yeni = {}
    for c in sut:
        s1 = g[c].shift(1)
        yeni[f"{c}_g1"] = s1
        yeni[f"{c}_g2"] = g[c].shift(2)
        o3 = s1.groupby(pan.lokasyon, observed=True).rolling(3, min_periods=2).mean()
        o7 = s1.groupby(pan.lokasyon, observed=True).rolling(7, min_periods=4).mean()
        o3 = o3.reset_index(level=0, drop=True).reindex(pan.index)
        o7 = o7.reset_index(level=0, drop=True).reindex(pan.index)
        yeni[f"{c}_go3"] = o3
        yeni[f"{c}_go7"] = o7
        yeni[f"{c}_anom7"] = pan[c] - o7
        yeni[f"{c}_ivme"] = o3 - o7
    pan = pd.concat([pan[["lokasyon", "tarih"]], pd.DataFrame(yeni, index=pan.index)], axis=1)
    return pan, [k for k in yeni]


def aile_d_hava():
    pan, yeni_sut = hava_paneli()
    at = tp[["lokasyon", "tarih"]].merge(pan, on=["lokasyon", "tarih"], how="left")
    ab = bf[["lokasyon", "tarih"]].merge(pan, on=["lokasyon", "tarih"], how="left")
    for c in yeni_sut:
        xt, xb = dizi(at, c), dizi(ab, c)
        yield f"hv_{c}", (lambda xt=xt, xb=xb: (st(xt), st(xb)))
        if c.endswith("anom7") or c.endswith("ivme"):
            for kip in ("x_sv", "x_soguk", "x_ufuk"):
                mt, mb = CARP[kip]
                a_, b_ = st(xt), st(xb)
                if a_ is None or b_ is None:
                    continue
                yield (
                    f"hv_{c}:{kip}",
                    lambda a_=a_, b_=b_, mt=mt, mb=mb: (st(a_ * mt), st(b_ * mb)),
                )
    # hazir *_ort7 sutunlarindan anomali (m121 yalniz ham hallerini gordu)
    for c, o in [
        ("sicaklik_ort", "sicaklik_ort_ort7"),
        ("cdd22", "cdd22_ort7"),
        ("cdd24", "cdd24_ort7"),
    ]:
        if o not in tp.columns:
            continue
        xt = dizi(tp, c) - dizi(tp, o)
        xb = dizi(bf, c) - dizi(bf, o)
        yield f"hz_{c}_anom", (lambda xt=xt, xb=xb: (st(xt), st(xb)))
        for kip in ("x_sv", "x_ufuk"):
            mt, mb = CARP[kip]
            a_, b_ = st(xt), st(xb)
            if a_ is None or b_ is None:
                continue
            yield (
                f"hz_{c}_anom:{kip}",
                lambda a_=a_, b_=b_, mt=mt, mb=mb: (st(a_ * mt), st(b_ * mb)),
            )


UFUK_ODAK = [
    "t_yuk_faktoru",
    "t_log_std",
    "t_sifir_orani",
    "t_trend",
    "t_son_kayit_yasi",
    "yas",
    "guc_yuzdelik",
    "t_kayma",
    "t_mevsim_genlik",
    "p_doluluk",
    "t_egim_sicaklik_ort",
    "sicaklik_ort",
]


def aile_e_ufuk():
    """E: ufuk_gun parcali dogrusal / kova / etkilesim."""
    for d in (12, 24, 36, 48, 61, 73, 86, 98, 110):
        yield (
            f"uf_mentese{d}",
            lambda d=d: (st(np.maximum(ufTh - d, 0)), st(np.maximum(ufBh - d, 0))),
        )
        yield (
            f"uf_ters{d}",
            lambda d=d: (st(np.minimum(ufTh - d, 0)), st(np.minimum(ufBh - d, 0))),
        )
    kenar = [0, 24, 48, 73, 98, 10**6]
    for i in range(len(kenar) - 1):
        at = ((ufTh > kenar[i]) & (ufTh <= kenar[i + 1])).astype(np.float64)
        ab = ((ufBh > kenar[i]) & (ufBh <= kenar[i + 1])).astype(np.float64)
        yield f"uf_kova{i}", (lambda at=at, ab=ab: (st(at), st(ab)))
        yield (
            f"uf_kova{i}:x_sv",
            lambda at=at, ab=ab: (st(st(at) * svT), st(st(ab) * svB)),
        )
    yield "uf_log", (lambda: (st(np.log1p(ufTh)), st(np.log1p(ufBh))))
    yield "uf_kare", (lambda: (st(ufTh**2), st(ufBh**2)))
    yield "uf_kok", (lambda: (st(np.sqrt(np.maximum(ufTh, 0))), st(np.sqrt(np.maximum(ufBh, 0)))))
    for c in UFUK_ODAK:
        if c not in tp.columns or c not in bf.columns:
            continue
        a_, b_ = st(tp[c].to_numpy()), st(bf[c].to_numpy())
        if a_ is None or b_ is None:
            continue
        yield (
            f"{c}:uf_log",
            lambda a_=a_, b_=b_: (st(a_ * st(np.log1p(ufTh))), st(b_ * st(np.log1p(ufBh)))),
        )
        for d in (24, 73):
            yield (
                f"{c}:uf_mentese{d}",
                lambda a_=a_, b_=b_, d=d: (
                    st(a_ * np.maximum(ufTh - d, 0)),
                    st(b_ * np.maximum(ufBh - d, 0)),
                ),
            )


YENI_ODAK = [
    "guc",
    "yas",
    "t_yuk_faktoru",
    "t_log_std",
    "t_sifir_orani",
    "t_trend",
    "t_mevsim_genlik",
    "t_hg_genligi",
    "t_son_kayit_yasi",
    "t_doluluk",
    "guc_payi",
    "guc_medyan_orani",
    "trafo_basina_nufus",
    "kva_basina_nufus",
    "ilce_nufus_yogunlugu",
    "osm_hat_yogunlugu",
    "trafo_basina_hat",
    "yerlesim_orani",
    "tarim_orani",
    "bitki_ortusu_orani",
    "ulusal_tepe_orani",
    "ulusal_yillik_buyume",
    "t_gy_sifir_orani",
    "ozet_pencere_gun",
]


def aile_f_guc():
    """F: guc/kapasite kirilimleri, yas ve yuk faktoru etkilesimleri."""
    gt, gb = dizi(tp, "guc"), dizi(bf, "guc")
    yield "guc_log", (lambda: (st(np.log1p(gt)), st(np.log1p(gb))))
    kn = np.quantile(gt[np.isfinite(gt)], [0.2, 0.4, 0.6, 0.8])
    kenar = np.concatenate([[-np.inf], kn, [np.inf]])
    for i in range(len(kenar) - 1):
        at = ((gt > kenar[i]) & (gt <= kenar[i + 1])).astype(np.float64)
        ab = ((gb > kenar[i]) & (gb <= kenar[i + 1])).astype(np.float64)
        yield f"guc_kova{i}", (lambda at=at, ab=ab: (st(at), st(ab)))
        for kip in ("x_sv", "x_ufuk", "x_soguk"):
            mt, mb = CARP[kip]
            a_, b_ = st(at), st(ab)
            if a_ is None or b_ is None:
                continue
            yield (
                f"guc_kova{i}:{kip}",
                lambda a_=a_, b_=b_, mt=mt, mb=mb: (st(a_ * mt), st(b_ * mb)),
            )
    var = [c for c in YENI_ODAK if c in tp.columns and c in bf.columns]
    hazir = {}
    for c in var:
        a_, b_ = st(tp[c].to_numpy()), st(bf[c].to_numpy())
        if a_ is not None and b_ is not None:
            hazir[c] = (a_, b_)
    ad_list = list(hazir)
    for i, c1 in enumerate(ad_list):
        a1, b1 = hazir[c1]
        for c2 in ad_list[i + 1 :]:
            a2, b2 = hazir[c2]
            yield (
                f"{c1}*{c2}",
                lambda a1=a1, a2=a2, b1=b1, b2=b2: (st(a1 * a2), st(b1 * b2)),
            )
    # ucluler: yas x yuk faktoru x guc yuzdeligi gibi
    ucler = [
        ("yas", "t_yuk_faktoru", "guc_yuzdelik"),
        ("t_yuk_faktoru", "t_log_std", "t_sifir_orani"),
        ("guc", "yas", "t_trend"),
    ]
    for c1, c2, c3 in ucler:
        if any(c not in tp.columns for c in (c1, c2, c3)):
            continue
        a1, b1 = st(tp[c1].to_numpy()), st(bf[c1].to_numpy())
        a2, b2 = st(tp[c2].to_numpy()), st(bf[c2].to_numpy())
        a3, b3 = st(tp[c3].to_numpy()), st(bf[c3].to_numpy())
        if any(v is None for v in (a1, a2, a3, b1, b2, b3)):
            continue
        yield (
            f"{c1}*{c2}*{c3}",
            lambda a1=a1, a2=a2, a3=a3, b1=b1, b2=b2, b3=b3: (
                st(a1 * a2 * a3),
                st(b1 * b2 * b3),
            ),
        )


def aile_g_mentese():
    """G: her kolonda parcali dogrusal mentese (esik kesitinin surekli hali)."""
    for kol in KOL:
        xt, xb = dizi(tp, kol), dizi(bf, kol)
        fv = xt[np.isfinite(xt)]
        if fv.size == 0:
            continue
        for q in (0.25, 0.5, 0.75):
            v_ = float(np.quantile(fv, q))
            yield (
                f"{kol}:mnt{int(q * 100)}",
                lambda xt=xt, xb=xb, v_=v_: (
                    st(np.maximum(xt - v_, 0)),
                    st(np.maximum(xb - v_, 0)),
                ),
            )
            yield (
                f"{kol}:eks{int(q * 100)}",
                lambda xt=xt, xb=xb, v_=v_: (
                    st(np.minimum(xt - v_, 0)),
                    st(np.minimum(xb - v_, 0)),
                ),
            )


def aile_h_carpimlar():
    """H: mevcut 40 eksenin BIRBIRIYLE carpimlari."""
    onbellek = []
    for ad in MEVCUT_ADLAR:
        xt, xb = kur(ad)
        if xt is None or xb is None:
            onbellek.append(None)
            continue
        onbellek.append((ad, xt.astype(np.float32), xb.astype(np.float32)))
    for i, o1 in enumerate(onbellek):
        if o1 is None:
            continue
        for o2 in onbellek[i + 1 :]:
            if o2 is None:
                continue
            yield (
                f"M[{o1[0]}]x[{o2[0]}]",
                lambda o1=o1, o2=o2: (st(o1[1] * o2[1]), st(o1[2] * o2[2])),
            )


AILELER = [
    ("A_trafo_kodlama", aile_a_trafo),
    ("B_lokasyon", aile_b_lokasyon),
    ("C_takvim", aile_c_takvim),
    ("D_hava_gecikme", aile_d_hava),
    ("E_ufuk", aile_e_ufuk),
    ("F_guc_yas", aile_f_guc),
    ("G_mentese", aile_g_mentese),
    ("H_carpim40", aile_h_carpimlar),
]

# ================================================== 3) TARAMA
print("\nAILE TARAMASI (kapilar: |rho_s|>=0.015, Q_dik>=0.25, plasebo z>=3, tavan)")
SEBEP_AD = {
    "dot": "sinyal yok",
    "tavan_on": "tavan(on)",
    "plasebo": "plasebo",
    "Qs": "Q_span",
    "rho_s": "|rho_s|<0.015",
    "Qs5": "Q_span(1e-5)",
    "rcond": "rcond kirilgan",
    "Qdik": "Q_dik<0.25",
    "tavan": "tavan dayanmiyor",
}
gecen, sayac, sebepler = [], {}, {}
for aile_ad, uret in AILELER:
    n_aday = n_gecen = 0
    sb = {}
    for ad, fn in uret():
        try:
            xt, xb = fn()
        except Exception as ex:  # noqa: BLE001  -- tek aday coktuyse tarama surmeli
            print(f"  ! {ad}: {type(ex).__name__} {ex}")
            continue
        n_aday += 1
        if xt is None or xb is None or not (np.isfinite(xt).all() and np.isfinite(xb).all()):
            sb["dejenere"] = sb.get("dejenere", 0) + 1
            continue
        s, sebep = olc(xt, xb, ONCEKI)
        if s is None:
            sb[sebep] = sb.get(sebep, 0) + 1
            continue
        n_gecen += 1
        gecen.append(
            dict(
                aile=aile_ad,
                eksen=ad,
                rho_s=s["rho_s"],
                rho_cv=s["rho_cv"],
                Qd=s["Qd"],
                z=s["z"],
                fn=fn,
            )
        )
    sayac[aile_ad] = (n_aday, n_gecen)
    sebepler[aile_ad] = sb
    ilk = sorted(sb.items(), key=lambda t: -t[1])[:3]
    nerede = ", ".join(f"{SEBEP_AD.get(k, k)} {v}" for k, v in ilk)
    print(f"  {aile_ad:>18s}: {n_aday:5d} aday -> {n_gecen:4d} GECTI   [elenme: {nerede}]")

gecen.sort(key=lambda d: -(d["rho_s"] ** 2))
print(f"\ntoplam {sum(v[0] for v in sayac.values())} aday, {len(gecen)} kapiyi gecti")


# ================================================== 4) DIK EKLEME
def dik_ekle(adaylar, taban, azami, baslik):
    """Adaylari sirayla dikleyerek ekler; kumulatif sum rho_s^2 dogru buyur."""
    onc = list(taban)
    alar = list(A_MEVCUT)
    sec, s2 = [], S2_MEVCUT
    print(f"\n--- {baslik} ---")
    print(
        f"{'eksen':>38s} {'aile':>16s} {'rho_cv':>8s} {'rho_s':>8s} {'Q_dik':>6s} "
        f"{'z':>6s} {'kum.rho':>8s}"
    )
    for kayit in adaylar:
        if len(sec) >= azami:
            print(f"  ... AZAMI {azami} sinirina dayandi (kesim kapidan gelmedi)")
            break
        xt, xb = kayit["fn"]()
        s, _sebep = olc(xt, xb, onc)  # onc buyuduyse Q_dik yeniden sinanir
        if s is None:
            continue
        onc.append(s["birim"])
        alar.append(s["a"])
        s2 += s["rho_s"] ** 2
        sec.append(
            dict(
                eksen=kayit["eksen"],
                aile=kayit["aile"],
                rho_s=s["rho_s"],
                rho_cv=s["rho_cv"],
                Qd=s["Qd"],
                z=s["z"],
                kum_rho_s=float(np.sqrt(s2)),
            )
        )
        if len(sec) <= 25 or len(sec) % 10 == 0:
            print(
                f"{kayit['eksen'][:38]:>38s} {kayit['aile'][:16]:>16s} {s['rho_cv']:+8.4f} "
                f"{s['rho_s']:+8.4f} {s['Qd']:6.3f} {s['z']:6.1f} {np.sqrt(s2):8.4f}"
            )
    return sec, float(np.sqrt(s2)), alar


RHO_ESKI = float(np.sqrt(S2_MEVCUT))
# H DISI: yeni aileler tek basina ne getiriyor (H, mevcut eksenlerin carpimi
# oldugu icin ayri raporlanir -- ayni hava sinyalinin dogrusal olmayan hali
# olabilir; Q_dik yalniz DOGRUSAL artikligi denetler).
sec_hsiz, rho_hsiz, A_HSIZ = dik_ekle(
    [g for g in gecen if g["aile"] != "H_carpim40"],
    ONCEKI,
    AZAMI_YENI,
    "H HARIC (A-G yeni aileler)",
)
gc.collect()
secilen, RHO_YENI, A_TUM = dik_ekle(gecen, ONCEKI, AZAMI_YENI, "TUM AILELER (H dahil)")

# ---------------------------------------------- KANIT TABANI TESHISI
# Her eksenin katsayisi 1.95*|rho_s| ile veriliyor ve rho_s YALNIZCA eksenin
# SPAN-ICI parcasindan olculuyor (r_hat span(V) icinde, 28 boyut). Q_dik kapisi
# eksenlerin DIK parcalarinin farkli olmasini sinar; SPAN-ICI parcalarinin
# farkli olmasini SINAMAZ. Iki eksen ayni span-ici sinyali tasiyip farkli dik
# yonlere dusuyorsa sum rho_s^2 iki kat buyur ama arkasindaki OLCUM tektir.
# Bagimsiz kanit tabani = ||P_A r_hat||, A = span-ici birim yonlerin gerdigi
# altuzay; tavani ||r_hat||.
LR = (V.T @ r_hat) / N


def kanit_tabani(alar):
    A = np.array(alar).T
    GA = A.T @ G @ A
    LA = A.T @ LR
    return float(np.sqrt(max(float(LA @ np.linalg.pinv(GA, rcond=1e-8) @ LA), 0.0)))


NRM = float(np.sqrt(float((r_hat * r_hat).mean())))
K40, KH, KT = kanit_tabani(A_MEVCUT), kanit_tabani(A_HSIZ), kanit_tabani(A_TUM)
print()
print("KANIT TABANI ||P_A r_hat||  (tavan ||r_hat||)")
print(f"  ||r_hat||                = {NRM:.4f}   <- ASILAMAZ TAVAN")
print(f"  mevcut {len(A_MEVCUT):3d} eksen         = {K40:.4f}")
print(f"  + H haric yeni  ({len(A_HSIZ):3d})    = {KH:.4f}   ({KH - K40:+.4f})")
print(f"  + tum yeni      ({len(A_TUM):3d})    = {KT:.4f}   ({KT - K40:+.4f})")
print("  sum rho_s^2 bu tavana TABI DEGIL: ayni span-ici olcum farkli dik")
print("  yonlere tekrar tekrar tasiniyor; buyumenin bir kismi bu tekrardir.")

print(f"\nMEVCUT {len(MEVCUT)} eksen              : sqrt(sum rho_s^2) = {RHO_ESKI:.4f}")
print(f"+ {len(sec_hsiz):3d} yeni eksen (H haric) : sqrt(sum rho_s^2) = {rho_hsiz:.4f}")
print(f"+ {len(secilen):3d} yeni eksen (H dahil) : sqrt(sum rho_s^2) = {RHO_YENI:.4f}")
print(f"ARTIS: {RHO_YENI - RHO_ESKI:+.4f}  ({100 * (RHO_YENI / RHO_ESKI - 1):+.1f}%)")
print(f"rho_pred = 1.95 * {RHO_YENI:.4f} = {TAVAN * RHO_YENI:.4f}  (mevcut {TAVAN * RHO_ESKI:.4f})")
for ad, h in [("3. sira", 0.99927), ("2. sira", 0.99614), ("1. sira", 0.99009)]:
    kap = float(np.sqrt(max(MSE_OPT - h * h, 1e-12)))
    print(
        f"  {ad}: gereken rho {kap:.4f}  ->  f_eski = {kap / (TAVAN * RHO_ESKI):.3f}"
        f"   f_yeni = {kap / (TAVAN * RHO_YENI):.3f}"
    )

with open(os.path.join(BURA, "m144_yeni_aileler.json"), "w") as fh:
    json.dump(
        dict(
            mevcut_eksen=len(MEVCUT),
            mevcut_rho_s=RHO_ESKI,
            yeni_eksen=len(secilen),
            birlesik_rho_s=RHO_YENI,
            yeni_eksen_h_haric=len(sec_hsiz),
            birlesik_rho_s_h_haric=rho_hsiz,
            kanit_tabani=dict(r_hat_normu=NRM, mevcut=K40, h_haric=KH, tum=KT),
            rho_pred_eski=TAVAN * RHO_ESKI,
            rho_pred_yeni=TAVAN * RHO_YENI,
            aile_sayaci={
                k: dict(aday=v[0], gecen=v[1], elenme=sebepler[k]) for k, v in sayac.items()
            },
            secilen=secilen,
            secilen_h_haric=sec_hsiz,
            kapidan_gecen=[
                dict(
                    aile=g["aile"], eksen=g["eksen"], rho_s=g["rho_s"], rho_cv=g["rho_cv"], z=g["z"]
                )
                for g in gecen[:400]
            ],
            mevcut=MEVCUT,
        ),
        fh,
        indent=1,
    )
print("\n-> m144_yeni_aileler.json yazildi (GONDERIM YOK)")
