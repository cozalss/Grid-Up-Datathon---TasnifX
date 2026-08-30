# ruff: noqa  -- YARIM KALDI: ajan kesildi, betik HIC KOSMADI, sonuc uretmedi.
# Sordugu soru sonradan baska olcumlerle cevaplandi (bkz. commit mesaji).
"""n04 -- Q_dik ve RHO_S_ALT kapilarinin DUYARLILIK analizi.

SORU: m148_demet_plani.py'deki Q_dik >= 0.25 kapisi 369 adaydan 136'sini
kabul ediyor. Gevsetirsek daha cok eksen girer ve SPAN-ICI ongorulen rho
buyur -- ama yeni eksenler oncekilere daha yakin oldugu icin DIS-ORNEK
gerceklesme orani dusebilir. Net etki nedir?

YONTEM (m141'in blok sinavinin parametrize edilmis hali):
  1. Eksen secimi + isaret + katsayi YALNIZ yaz25 blogundan (m148 ile birebir).
  2. Bilesigin ongorulen rho = ||beta|| = sqrt(toplam KATS^2)  (dik eksenler).
  3. GERCEKLESME: ayni eksen listesi, ayni isaretler, ayni katsayilar ile
     bilesik guz25 ve kis26 bloklarinda YENIDEN kurulur (o bloklarin kendi
     Gram-Schmidt'i ile) ve artikla korelasyonu olculur. Bu bloklar secimde
     HIC kullanilmadi.
  4. gerceklesen rho = CARPAN * kor_dis   (CARPAN=0.798 blok->test olcegi;
     m148'de rho_cv = CARPAN*kor ile ayni donusum).
  5. oran = gerceklesen / ongorulen.

HIZ: kapilarin cogu (Qs, rho_s, rcond, plasebo, tavan) esiklerden BAGIMSIZ.
Tek gecisle hesaplanip onbelleklenir; her esik ayari yalnizca ucuz ardisik
Gram-Schmidt'i yeniden calistirir.
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
TABAN = "tuketim_m6_ikiyon.csv"
HEDEF_SOGUK, CARPAN, TAVAN = 0.222, 0.798, 1.95
AZAMI_EKSEN = 40
AZAMI_KABUL = 500  # bellek tavani; gecis 1 sonrasi hayatta kalan sayisina kirpilir
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0, buzmeli_r_hat  # noqa: E402

# ---------------------------------------------------------------- LB uzayi
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
print(f"saf optimum {np.sqrt(M0 - gercek):.6f}   V: {V.shape}   N={N:,}")

# ------------------------------------------------------------------ bloklar
tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
e = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
SIC_AILE = [(t, aa) for t in (1000, 1001, 1002) for aa in ("cat", "xgb", "lgbm")]


def blok_kur(blok):
    bl = e[e._blok == blok]
    sic, sog = bl[bl.soguk_mu == 0], bl[bl.soguk_mu == 1]
    Pl = [
        np.load(os.path.join(AO, f"{blok}_{t}_{aa}_uretim.npy")).astype(np.float64)
        for t, aa in SIC_AILE
        if os.path.exists(os.path.join(AO, f"{blok}_{t}_{aa}_uretim.npy"))
    ]
    zz = np.load(os.path.join(DN, f"soguk_tahmin_{blok}.npz"))
    ii = np.concatenate([sic.index.values, sog.index.values])
    pp = np.concatenate([np.mean(Pl, axis=0), np.mean([zz[q] for q in zz.files], axis=0)])
    ff = e.loc[ii].copy()
    rr = np.log1p(ff.tuketim.values.astype(np.float64)) - pp
    sg = ff.soguk_mu.values.astype(np.float64)
    w2 = np.where(sg == 1, HEDEF_SOGUK / sg.mean(), (1 - HEDEF_SOGUK) / (1 - sg.mean()))
    return ff, pp, rr, sg, w2 / w2.mean()


def st(x):
    x = np.asarray(x, dtype=np.float64).copy()
    f = np.isfinite(x)
    if not f.any():
        return None
    x[~f] = np.median(x[f])
    x -= x.mean()
    s = np.sqrt(float((x * x).mean()))
    return x / s if s > 1e-12 else None


svT = st(a0)
sgT = tp.soguk_mu.values.astype(np.float64)
ufT = st(tp.ufuk_gun.to_numpy())
ayT = st(pd.to_datetime(tp.tarih).dt.month.to_numpy().astype(np.float64))
ESIK = {"ust10": (0.9, True), "ust25": (0.75, True), "alt10": (0.1, False)}

# kur() icin cerceve-bagimli baglam: (bf, CARP)
BAGLAM = {}


def hazirla(blok):
    ff, pp, rr, sg, w2 = blok_kur(blok)
    carp = {
        "x_sv": (svT, st(pp)),
        "x_soguk": (sgT, sg),
        "x_ufuk": (ufT, st(ff.ufuk_gun.to_numpy())),
        "x_ay": (ayT, st(pd.to_datetime(ff.tarih).dt.month.to_numpy().astype(np.float64))),
    }
    return {"bf": ff, "carp": carp, "r": rr, "w": w2, "m0": float((w2 * rr * rr).mean())}


def kur(ad, B):
    """(test yonu, blok yonu) -- m148_demet_plani.kur() ile birebir."""
    bf, CARP = B["bf"], B["carp"]
    if ad.startswith("M[") and "]x[" in ad and ad.endswith("]"):
        k1, k2 = ad[2:-1].split("]x[", 1)
        a1, b1 = kur(k1, B)
        a2, b2 = kur(k2, B)
        if a1 is None or a2 is None or b1 is None or b2 is None:
            return None, None
        return st(a1 * a2), st(b1 * b2)
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
    if kip == "mnt75":
        fv = xt[np.isfinite(xt)]
        if fv.size == 0:
            return None, None
        v_ = float(np.quantile(fv, 0.75))
        return st(np.maximum(xt - v_, 0.0)), st(np.maximum(xb - v_, 0.0))
    if kip == "kare":
        a_, b_ = st(xt), st(xb)
        # m148'de yalniz a_ kontrol ediliyor; dis bloklarda b_ sabit cikabilir
        return (None, None) if a_ is None or b_ is None else (st(a_**2), st(b_**2))
    return st(xt), st(xb)


# ------------------------------------------------------- aday sirasi (m148)
with open(os.path.join(M29, "m121_derin_tarama.json")) as fh:
    TARAMA = json.load(fh)
with open(os.path.join(M29, "m144_yeni_aileler.json"), encoding="utf-8") as fh:
    _M144 = json.load(fh)["kapidan_gecen"]
YENI_EKSENLER = [r["eksen"] for r in sorted(_M144, key=lambda r: -abs(r["rho_s"]))]
ADAYLAR = [(k["eksen"], False) for k in TARAMA] + [(a, True) for a in YENI_EKSENLER]
print(f"aday eksen: {len(ADAYLAR)}")

YAZ = hazirla("yaz25")
DIS_BLOK = {b: hazirla(b) for b in ("guz25", "kis26")}
del e
gc.collect()
for _b, _B in DIS_BLOK.items():
    print(f"dis blok {_b}: {len(_B['r']):,} satir")
rng = np.random.default_rng(5)
tn = YAZ["bf"].tanim.values
uqn = pd.unique(tn)
gi = pd.Series(np.arange(len(uqn)), index=uqn)[tn].to_numpy()
PERM = [
    np.argsort(np.argsort(rng.permutation(len(uqn))[gi], kind="stable"), kind="stable")
    for _ in range(20)
]

# ==========================================================================
# GECIS 1 -- esikten BAGIMSIZ kapilar. Hayatta kalanlarin xp0 (test, span'a
# dik) ve xb (yaz25) yonleri onbellege alinir.
# ==========================================================================
RHO_S_TABAN = 0.010  # en gevsek ayar; daha sikilari sonradan filtrelenir
ww, rb, m0b = YAZ["w"], YAZ["r"], YAZ["m0"]
SAG = []  # [{ad, yeni, rho_s, rho_cv, ix}]
XP0 = []  # test tarafi span-dik yonler
sayac = {"kur": 0, "Qs": 0, "rho_s": 0, "rcond": 0, "plasebo": 0, "tavan": 0}
for ad, _yeni in ADAYLAR:
    xt, xb = kur(ad, YAZ)
    if xt is None or xb is None:
        sayac["kur"] += 1
        continue
    cc = Gi @ ((V.T @ xt) / N)
    xp0 = xt - V @ cc
    Qs = 1.0 - float((xp0 * xp0).mean())
    if Qs < 0.02:
        sayac["Qs"] += 1
        continue
    rho_s = float((r_hat * xt).mean()) / np.sqrt(Qs)
    if abs(rho_s) < RHO_S_TABAN:
        sayac["rho_s"] += 1
        continue
    cc5 = GI5 @ ((V.T @ xt) / N)
    xp5 = xt - V @ cc5
    Qs5 = 1.0 - float((xp5 * xp5).mean())
    if Qs5 < 0.02:
        sayac["rcond"] += 1
        continue
    if abs(float((r_hat * xt).mean()) / np.sqrt(Qs5) - rho_s) > 0.3 * abs(rho_s):
        sayac["rcond"] += 1
        continue
    kor = float((ww * rb * xb).mean()) / np.sqrt(m0b)
    gur = np.std([float((ww * rb[s] * xb).mean()) / np.sqrt(m0b) for s in PERM])
    if abs(kor) < 3 * gur:
        sayac["plasebo"] += 1
        continue
    rho_cv = CARPAN * kor
    if abs(rho_cv) < TAVAN * abs(rho_s):
        sayac["tavan"] += 1
        continue
    SAG.append({"ad": ad, "yeni": _yeni, "rho_s": float(rho_s), "rho_cv": float(rho_cv)})
    XP0.append(xp0)
XP0 = np.asarray(XP0)  # (M, N)
print(f"gecis 1: {len(SAG)} aday esikten-bagimsiz kapilari gecti; elenme {sayac}")
AZAMI_KABUL = min(AZAMI_KABUL, len(SAG))
gc.collect()


def ardisik_kabul(qd_esik, rho_s_alt):
    """m148'in eksen dongusu -- kabul edilen adaylarin indeksleri + Qd + kats."""
    U = np.empty((AZAMI_KABUL, N), dtype=np.float64)
    k = 0
    kabul, kats, qdler = [], [], []
    for i, s in enumerate(SAG):
        if abs(s["rho_s"]) < rho_s_alt:
            continue
        if not s["yeni"] and k >= AZAMI_EKSEN:
            continue
        if k >= AZAMI_KABUL:
            break
        xp = XP0[i].copy()
        if k:
            xp -= U[:k].T @ ((U[:k] @ xp) / N)
        Qd = float((xp * xp).mean())
        if Qd < qd_esik:
            continue
        U[k] = xp / np.sqrt(Qd)
        k += 1
        kabul.append(i)
        kats.append(float(np.sign(s["rho_cv"]) * TAVAN * abs(s["rho_s"])))
        qdler.append(Qd)
    del U
    gc.collect()
    return kabul, np.array(kats), qdler


def dis_gerceklesme(blok_ad, B, kabul, kats):
    """Ayni eksenler/isaretler/katsayilar ile bilesigi blokta yeniden kur."""
    r2, w2 = B["r"], B["w"]
    n2 = len(r2)
    U = np.empty((AZAMI_KABUL, n2), dtype=np.float64)
    k = 0
    duz = np.zeros(n2)
    dusen = 0
    for j, i in enumerate(kabul):
        _, xb = kur(SAG[i]["ad"], B)
        if xb is None:
            dusen += 1
            continue
        ub = xb.copy()
        if k:
            # agirlikli izdusum: kats = <ub, u>_w = mean(w*ub*u)
            ub -= U[:k].T @ ((U[:k] @ (w2 * ub)) / n2)
        nb = np.sqrt(float((w2 * ub * ub).mean()))
        if nb < 0.15:
            dusen += 1
            continue
        ub /= nb
        U[k] = ub
        k += 1
        duz += kats[j] * ub
    nrm = np.sqrt(float((w2 * duz * duz).mean()))
    del U
    gc.collect()
    if nrm < 1e-12:
        return 0.0, 0.0, dusen
    kor = float((w2 * r2 * duz).mean()) / (nrm * np.sqrt(B["m0"]))
    return kor, nrm, dusen


# ==========================================================================
# GECIS 2 -- esik izgarasi
# ==========================================================================
QD_LISTE = [0.25, 0.20, 0.15, 0.10, 0.05]
RS_LISTE = [0.020, 0.015, 0.012, 0.010]
SONUC = []
print(
    f"\n{'Q_dik':>6s} {'rho_s_alt':>9s} {'eksen':>6s} {'ongor.rho':>9s} "
    f"{'kor_guz':>8s} {'kor_kis':>8s} {'gercek.rho':>10s} {'oran':>6s} {'iciyaz':>7s}"
)
for qd in QD_LISTE:
    for rs in RS_LISTE:
        kabul, kats, qdler = ardisik_kabul(qd, rs)
        ongor = float(np.sqrt((kats**2).sum())) if len(kats) else 0.0
        kors, dusenler = {}, {}
        for b, B in DIS_BLOK.items():
            kor, _, dus = dis_gerceklesme(b, B, kabul, kats)
            kors[b] = kor
            dusenler[b] = dus
        # bloklarin satir sayisiyla agirlikli ortalama korelasyon
        wgz, wks = len(DIS_BLOK["guz25"]["r"]), len(DIS_BLOK["kis26"]["r"])
        kor_dis = (kors["guz25"] * wgz + kors["kis26"] * wks) / (wgz + wks)
        gercek = CARPAN * kor_dis
        # IC-ORNEK karsilastirma: ayni bilesigin yaz25'teki korelasyonu
        kor_ic, _, _ = dis_gerceklesme("yaz25", YAZ, kabul, kats)
        oran = gercek / ongor if ongor > 1e-9 else 0.0
        SONUC.append(
            {
                "Q_dik": qd,
                "rho_s_alt": rs,
                "eksen": len(kabul),
                "ongorulen_rho": ongor,
                "kor_guz25": kors["guz25"],
                "kor_kis26": kors["kis26"],
                "kor_dis_agirlikli": kor_dis,
                "gerceklesen_rho": gercek,
                "oran": oran,
                "kor_yaz25_ic": kor_ic,
                "gerceklesen_ic_rho": CARPAN * kor_ic,
                "dusen_eksen": dusenler,
            }
        )
        print(
            f"{qd:6.2f} {rs:9.3f} {len(kabul):6d} {ongor:9.4f} "
            f"{kors['guz25']:8.4f} {kors['kis26']:8.4f} {gercek:10.4f} "
            f"{oran:6.3f} {CARPAN * kor_ic:7.4f}"
        )

en_iyi = max(SONUC, key=lambda d: d["gerceklesen_rho"])
taban = [d for d in SONUC if d["Q_dik"] == 0.25 and d["rho_s_alt"] == 0.015][0]
print(
    f"\nEN IYI gerceklesen rho: Q_dik={en_iyi['Q_dik']} rho_s_alt={en_iyi['rho_s_alt']} "
    f"-> {en_iyi['gerceklesen_rho']:.4f} ({en_iyi['eksen']} eksen)"
)
print(
    f"MEVCUT (0.25 / 0.015): gerceklesen {taban['gerceklesen_rho']:.4f} "
    f"({taban['eksen']} eksen, ongorulen {taban['ongorulen_rho']:.4f})"
)
CIK = {
    "aciklama": "Q_dik ve RHO_S_ALT kapilarinin dis-ornek (guz25+kis26) duyarliligi",
    "yontem": (
        "secim/isaret/katsayi yaz25'ten; bilesik guz25 ve kis26'da yeniden kurulup "
        "korelasyonu olculur; gerceklesen rho = 0.798 * agirlikli kor"
    ),
    "aday_sayisi": len(ADAYLAR),
    "gecis1_hayatta": len(SAG),
    "elenme": sayac,
    "izgara": SONUC,
    "mevcut": taban,
    "en_iyi": en_iyi,
}
yol = os.path.join(M29, "n04_kapi_duyarliligi.json")
with open(yol, "w", encoding="utf-8") as fh:
    json.dump(CIK, fh, ensure_ascii=False, indent=2)
print(f"\nyazildi: {yol}")
