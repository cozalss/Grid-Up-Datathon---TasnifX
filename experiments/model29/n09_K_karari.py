# ruff: noqa: F821  -- U ic ice fonksiyonlarda kapanis degiskeni (satir 421);
# ruff bunu goremiyor, betik sorunsuz kosup n09_K_karari.json uretti.
"""K KARARI -- m148 kac eksen kullanmali?

n01_K_asiri_uyum.py'nin diskteki JSON'u ESKI bir surumden kaldi (63 eksenlik
ortak kume, m148'inkinden FARKLI havuz ve FARKLI siralama). Bu betik olcumu
m148'in KENDI secim dongusuyle, KENDI siralamasiyla ve KENDI kapilariyla
bastan yapar.

YONTEM.
 1) m148'in secim dongusu BIREBIR kopyalanir (ayni kur(), ayni kapilar:
    Qs>=0.02, |rho_s|>=0.015, rcond kararliligi, Q_dik>=0.25, plasebo z>=3,
    TAVAN |rho_cv|>=1.95|rho_s|). Cikti: kul, KATS, U(test uzayi), aile/hava.
    ! Bu kapilarin bir kismi yaz25'e BAKARAK karar verir (plasebo, tavan,
      isaret). Dolayisiyla yaz25'te olculen her sey IC-ORNEKTIR.
 2) Ayni eksenler guz25 ve kis26 bloklarinda kurulur, AYNI SIRAYLA agirlikli
    Gram-Schmidt uygulanir, m148'in LB agirliklari (KATS) ve yaz25 isaretleri
    tasinir. Gerceklesen = agirlikli korelasyon.
 3) IKI KURULUS ayri ayri olculur:
      TEK   : tek bilesik yon (agirliklandirma tamamen tahminden)
      4BLOK : m148'in 4 blogu ayri ayri, alt uzaya izdusum
              (bloklar ARASI agirlik LB tarafindan duzeltilir)
 4) Gurultu tabani: trafo-kumeli permutasyon (m148'in PERM'i gibi) --
    "doyum" ile "gurultu" ayirt edilebiliyor mu?
 5) Guven araligi: trafo-kumeli bootstrap.

CIKTI: n09_K_karari.json
"""

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
SCR = os.environ.get("N09_SCR") or os.path.join(
    r"C:/Users/Cem/AppData/Local/Temp/claude",
    "c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX",
    "e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad",
)
os.makedirs(SCR, exist_ok=True)
TABAN = "tuketim_m6_ikiyon.csv"
HEDEF_SOGUK, CARPAN, TAVAN = 0.222, 0.798, 1.95
RHO_S_ALT = 0.015
AZAMI_EKSEN = 40
BLOK_ALT = 0.06
DEMET_HEDEF = 4
NB_ALT = 0.02
BLOKLAR = ["yaz25", "guz25", "kis26"]
K_LISTE = [5, 10, 25, 40, 50, 63, 80, 100, 120, 136]
NBOOT = 400
NPERM = 200
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0, buzmeli_r_hat  # noqa: E402

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
IDS = te.id.values


def oku(f):
    d = pd.read_csv(os.path.join(S, f))
    k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
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
TABAN_MSE = float(M0 - 2 * kL + float((r_hat * r_hat).mean()))
print(f"span V {V.shape}, TABAN_MSE {TABAN_MSE:.8f} -> saf span skoru {np.sqrt(TABAN_MSE):.5f}")

tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
e = pd.read_parquet(os.path.join(DN, "egitim.parquet"))


def st(x):
    x = np.asarray(x, dtype=np.float64).copy()
    f = np.isfinite(x)
    if not f.any():
        return None
    x[~f] = np.median(x[f])
    x -= x.mean()
    s = np.sqrt(float((x * x).mean()))
    return x / s if s > 1e-12 else None


def blok_kur(blok):
    bl = e[e._blok == blok]
    sic, sog = bl[bl.soguk_mu == 0], bl[bl.soguk_mu == 1]
    Pl = [
        np.load(os.path.join(AO, f"{blok}_{t}_{aa}_uretim.npy")).astype(np.float64)
        for t in (1000, 1001, 1002)
        for aa in ("cat", "xgb", "lgbm")
        if os.path.exists(os.path.join(AO, f"{blok}_{t}_{aa}_uretim.npy"))
    ]
    zz = np.load(os.path.join(DN, f"soguk_tahmin_{blok}.npz"))
    ii = np.concatenate([sic.index.values, sog.index.values])
    pp = np.concatenate([np.mean(Pl, axis=0), np.mean([zz[q] for q in zz.files], axis=0)])
    ff = e.loc[ii].copy()
    rr = np.log1p(ff.tuketim.values.astype(np.float64)) - pp
    sg = ff.soguk_mu.values.astype(np.float64)
    w2 = np.where(sg == 1, HEDEF_SOGUK / sg.mean(), (1 - HEDEF_SOGUK) / (1 - sg.mean()))
    return ff, pp, rr, w2 / w2.mean()


sgT = tp.soguk_mu.values.astype(np.float64)
svT = st(a0)
ufT = st(tp.ufuk_gun.to_numpy())
ayT = st(pd.to_datetime(tp.tarih).dt.month.to_numpy().astype(np.float64))
ESIK = {"ust10": (0.9, True), "ust25": (0.75, True), "alt10": (0.1, False)}


class Kur:
    """m148_demet_plani.py'deki kur() fonksiyonunun BIREBIR kopyasi,
    yalnizca blok cercevesi (bf, pb) parametrelendirildi."""

    def __init__(self, bf, pb):
        self.bf = bf
        svB = st(pb)
        sgm = bf.soguk_mu.values.astype(np.float64)
        ufB = st(bf.ufuk_gun.to_numpy())
        ayB = st(pd.to_datetime(bf.tarih).dt.month.to_numpy().astype(np.float64))
        self.CARP = {
            "x_sv": (svT, svB),
            "x_soguk": (sgT, sgm),
            "x_ufuk": (ufT, ufB),
            "x_ay": (ayT, ayB),
        }

    def __call__(self, ad):
        bf = self.bf
        if ad.startswith("M[") and "]x[" in ad and ad.endswith("]"):
            ic = ad[2:-1]
            k1, k2 = ic.split("]x[", 1)
            a1, b1 = self(k1)
            a2, b2 = self(k2)
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
        if kip in self.CARP:
            mt, mb = self.CARP[kip]
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
            # NOT: m148'in kur()'unda burada yalniz a_ kontrol ediliyor; bf=yaz25
            # oldugu icin orada patlamiyor. kis26'da b_ None olabilir (eksen o
            # blokta SABIT) -- eksen o blokta olculemez, dusurulur.
            return (None, None) if a_ is None or b_ is None else (st(a_**2), st(b_**2))
        return st(xt), st(xb)


# =============================================================== 1) SECIM
SEC_YOL = os.path.join(SCR, "n09_secim.json")
if os.path.exists(SEC_YOL):
    with open(SEC_YOL, encoding="utf-8") as fh:
        SEC = json.load(fh)
    print(f"secim onbellekten: {len(SEC['kul'])} eksen")
else:
    ff0, pb0, rb, ww = blok_kur("yaz25")
    m0b = float((ww * rb * rb).mean())
    kur0 = Kur(ff0, pb0)
    with open(os.path.join(M29, "m121_derin_tarama.json")) as fh:
        TARAMA = json.load(fh)
    with open(os.path.join(M29, "m144_yeni_aileler.json"), encoding="utf-8") as fh:
        _M144 = json.load(fh)["kapidan_gecen"]
    YENI_EKSENLER = [r["eksen"] for r in sorted(_M144, key=lambda r: -abs(r["rho_s"]))]
    YENI_AILE = {r["eksen"]: r["aile"] for r in _M144}
    rng0 = np.random.default_rng(5)
    tn = ff0.tanim.values
    uqn = pd.unique(tn)
    gi = pd.Series(np.arange(len(uqn)), index=uqn)[tn].to_numpy()
    PERM = [
        np.argsort(np.argsort(rng0.permutation(len(uqn))[gi], kind="stable"), kind="stable")
        for _ in range(20)
    ]
    kul, KAT_L, RHOS_L, RHOCV_L, AILE_L, YM_L, ONCEKI = [], [], [], [], [], [], []
    for kayit in TARAMA + [{"eksen": a, "_yeni": True} for a in YENI_EKSENLER]:
        _yeni = bool(kayit.get("_yeni"))
        if not _yeni and len(kul) >= AZAMI_EKSEN:
            continue
        ad = kayit["eksen"]
        xt, xb = kur0(ad)
        if xt is None or xb is None:
            continue
        cc = Gi @ ((V.T @ xt) / N)
        xp0 = xt - V @ cc
        Qs = 1.0 - float((xp0 * xp0).mean())
        if Qs < 0.02:
            continue
        rho_s = float((r_hat * xt).mean()) / np.sqrt(Qs)
        if abs(rho_s) < RHO_S_ALT:
            continue
        cc5 = GI5 @ ((V.T @ xt) / N)
        xp5 = xt - V @ cc5
        Qs5 = 1.0 - float((xp5 * xp5).mean())
        if Qs5 < 0.02:
            continue
        if abs(float((r_hat * xt).mean()) / np.sqrt(Qs5) - rho_s) > 0.3 * abs(rho_s):
            continue
        xp = xp0.copy()
        for u in ONCEKI:
            xp -= float((xp * u).mean()) * u
        Qd = float((xp * xp).mean())
        if Qd < 0.25:
            continue
        kor = float((ww * rb * xb).mean()) / np.sqrt(m0b)
        gur = np.std([float((ww * rb[s] * xb).mean()) / np.sqrt(m0b) for s in PERM])
        if abs(kor) < 3 * gur:
            continue
        rho_cv = CARPAN * kor
        if abs(rho_cv) < TAVAN * abs(rho_s):
            continue
        ONCEKI.append(xp / np.sqrt(Qd))
        kul.append(ad)
        KAT_L.append(float(np.sign(rho_cv) * TAVAN * abs(rho_s)))
        RHOS_L.append(float(rho_s))
        RHOCV_L.append(float(rho_cv))
        AILE_L.append(YENI_AILE.get(ad, "m121_taban"))
        YM_L.append(bool(_yeni))
    SEC = dict(
        kul=kul,
        KATS=KAT_L,
        rho_s=RHOS_L,
        rho_cv=RHOCV_L,
        aile=AILE_L,
        yeni=YM_L,
        taban_mse=TABAN_MSE,
    )
    with open(SEC_YOL, "w", encoding="utf-8") as fh:
        json.dump(SEC, fh)
    del ONCEKI
    print(f"secim: {len(kul)} eksen")

KUL = SEC["kul"]
KATS = np.array(SEC["KATS"])
RHOS = np.array(SEC["rho_s"])
AILE = np.array(SEC["aile"])
NE = len(KUL)
print(f"eksen sayisi = {NE}, ||BETA|| tum = {np.sqrt((KATS**2).sum()):.4f}")
for K in K_LISTE:
    if K <= NE:
        print(f"  P_{K:<4d} = {np.sqrt((KATS[:K] ** 2).sum()):.4f}")

# --- m148'in 4 blogunu (aile x hava/yapi) BIREBIR yeniden kur ---
HAVA = (
    "sicak",
    "cdd",
    "hdd",
    "nem",
    "vpd",
    "et0",
    "bulut",
    "yagis",
    "hissedilen",
    "asiri",
    "ruzgar",
    "guneslen",
    "gunes",
    "x_ay",
    "ay_",
    "sicaklik",
)
_HV = np.array([bool(any(h in a for h in HAVA)) for a in KUL])
_HAM = {}
for _f in sorted(set(SEC["aile"])):
    _m = np.equal(AILE, _f)
    if _f in ("m121_taban", "H_carpim40"):
        _HAM[f"{_f}/hava"] = _m & _HV
        _HAM[f"{_f}/yapi"] = _m & ~_HV
    else:
        _HAM[_f] = _m
_AG = {k: float(np.sqrt((KATS[m] ** 2).sum())) for k, m in _HAM.items() if m.sum()}
_HAM = {k: v for k, v in _HAM.items() if k in _AG}
while True:
    _kucuk = [k for k, v in _AG.items() if v < BLOK_ALT]
    if not _kucuk or len(_AG) <= 2:
        break
    _k = min(_kucuk, key=lambda k: _AG[k])
    _hedef = min((k for k in _AG if k != _k), key=lambda k: _AG[k])
    _HAM[_hedef] = _HAM[_hedef] | _HAM[_k]
    del _HAM[_k], _AG[_k]
    _AG[_hedef] = float(np.sqrt((KATS[_HAM[_hedef]] ** 2).sum()))
while len(_AG) > DEMET_HEDEF:
    _k = min(_AG, key=lambda k: _AG[k])
    _hedef = min((k for k in _AG if k != _k), key=lambda k: _AG[k])
    _HAM[_hedef] = _HAM[_hedef] | _HAM[_k]
    del _HAM[_k], _AG[_k]
    _AG[_hedef] = float(np.sqrt((KATS[_HAM[_hedef]] ** 2).sum()))
_SIRA = sorted(_AG, key=lambda k: -_AG[k])
print("\nm148 BLOKLARI (tam K):")
for k in _SIRA:
    print(f"  {k:>24s} {int(_HAM[k].sum()):4d} eksen  ||BETA_b|| {_AG[k]:.4f}")
print(f"  toplam ||BETA|| = {np.sqrt(sum(v * v for v in _AG.values())):.4f}")
MASK4 = np.array([_HAM[k] for k in _SIRA])  # (4, NE)


# =============================================== 2) BLOKLARDA GS + OLCUM
rng = np.random.default_rng(20260831)
B_LISTE = [1, 2, 3, 4, 5, 6, 8]


def kor_w(w, rr, m0, v):
    """agirlikli korelasyon <rr,v>_w / (||v||_w * sqrt(m0))"""
    n2 = float((w * v * v).mean())
    if n2 <= 0:
        return 0.0
    return float((w * rr * v).mean()) / np.sqrt(n2 * m0)


def maske_B(B, idx):
    """m148'in 4 temel blogundan B blok uret.
    B<4: en kucuk ikisini birlestir (m148 kurali).
    B>4: en buyugu, ICINDEKI |rho_s| medyaninden ikiye bol.
    idx: bu K'da ayakta olan eksenlerin indeks dizisi. Maskeler idx UZERINDE.
    """
    kk = np.abs(KATS[idx])
    rs = np.abs(RHOS[idx])

    def ag(m):
        return float(np.sqrt((kk[m] ** 2).sum()))

    cur = []
    for b in range(len(MASK4)):
        m = MASK4[b][idx]
        if m.sum():
            cur.append([_SIRA[b], m.copy()])
    while len(cur) > B and len(cur) > 1:
        i = min(range(len(cur)), key=lambda i: ag(cur[i][1]))
        j = min((j for j in range(len(cur)) if j != i), key=lambda j: ag(cur[j][1]))
        cur[j] = [cur[j][0] + "+" + cur[i][0], cur[j][1] | cur[i][1]]
        cur.pop(i)
    while len(cur) < B:
        aday = [i for i in range(len(cur)) if int(cur[i][1].sum()) >= 2]
        if not aday:
            break
        i = max(aday, key=lambda i: ag(cur[i][1]))
        ad, m = cur[i]
        pos = np.flatnonzero(m)
        med = float(np.median(rs[pos]))
        m1 = np.zeros(len(m), dtype=bool)
        m1[pos[rs[pos] >= med]] = True
        m2 = m & ~m1
        if m1.sum() == 0 or m2.sum() == 0:
            # medyan tum degerleri bir yana atti -> ikiye ortadan bol
            m1[:] = False
            m1[pos[: len(pos) // 2]] = True
            m2 = m & ~m1
            if m1.sum() == 0 or m2.sum() == 0:
                break
        cur[i] = [ad + "|ust", m1]
        cur.append([ad + "|alt", m2])
    cur.sort(key=lambda t: -ag(t[1]))
    return cur


SONUC = {}
for blok in [b for b in BLOKLAR if os.environ.get("N09_ATLA_BLOK") != "1"]:
    ff, pb, rr, w = blok_kur(blok)
    m0 = float((w * rr * rr).mean())
    kur = Kur(ff, pb)
    nrow = len(rr)
    print(f"\n=== {blok}: {nrow:,} satir, ||rr||_w = {np.sqrt(m0):.5f}")
    U = np.zeros((NE, nrow), dtype=np.float64)
    canli = np.zeros(NE, dtype=bool)
    for i, ad in enumerate(KUL):
        _, xb = kur(ad)
        if xb is None or not np.isfinite(xb).all():
            continue
        u = xb.astype(np.float64)
        for j in range(i):
            if canli[j]:
                u -= float((w * u * U[j]).mean()) * U[j]
        n1 = np.sqrt(float((w * u * u).mean()))
        if n1 < NB_ALT:
            continue
        U[i] = u / n1
        canli[i] = True
    print(f"  {int(canli.sum())}/{NE} eksen blokta ayakta (GS sonrasi)")
    tn = ff.tanim.values
    uq, gidx = np.unique(tn, return_inverse=True)
    ng = len(uq)
    rngp = np.random.default_rng(7)
    PERMS = [
        np.argsort(np.argsort(rngp.permutation(ng)[gidx], kind="stable"), kind="stable")
        for _ in range(NPERM)
    ]
    s3 = np.bincount(gidx, weights=w * rr * rr, minlength=ng)
    # ortak bootstrap agirliklari: B'ler arasi FARK esli olsun
    BW = np.array(
        [rng.multinomial(ng, np.full(ng, 1.0 / ng)).astype(np.float64) for _ in range(NBOOT)]
    )

    def _rho2_vek(Gd, PW):
        """Gd yonleri icin (nboot,) sum corr^2 dizisi + tam ornek degeri."""
        if not Gd:
            return 0.0, np.zeros(len(PW))
        S1 = np.array([np.bincount(gidx, weights=w * rr * g, minlength=ng) for g in Gd])
        S2 = np.array([np.bincount(gidx, weights=w * g * g, minlength=ng) for g in Gd])
        one = np.ones(ng)
        t1, t2, t3 = S1 @ one, S2 @ one, float(one @ s3)
        tam = float((np.where(t2 > 0, t1 / np.sqrt(np.maximum(t2, 1e-300) * t3), 0.0) ** 2).sum())
        T1, T2 = S1 @ PW.T, S2 @ PW.T  # (nG, nboot)
        T3 = PW @ s3
        rr2 = np.where(
            (T2 > 0) & (T3 > 0), T1 / np.sqrt(np.maximum(T2, 1e-300) * np.maximum(T3, 1e-300)), 0.0
        )
        return tam, (rr2**2).sum(axis=0)

    def _gs_blok(cur, idx):
        Gd = []
        for _ad, m in cur:
            sel = idx[m]
            if len(sel) == 0:
                continue
            v = KATS[sel] @ U[sel]
            n0 = np.sqrt(float((w * v * v).mean()))
            if n0 < 1e-12:
                continue
            v = v / n0
            for g in Gd:
                v = v - float((w * v * g).mean()) * g
            n1 = np.sqrt(float((w * v * v).mean()))
            if n1 < 0.05:
                continue
            Gd.append(v / n1)
        return Gd

    def olc(K):
        m = canli.copy()
        m[K:] = False
        idx = np.flatnonzero(m)
        P = float(np.sqrt((KATS[idx] ** 2).sum()))
        out = dict(K=K, n=int(len(idx)), P=P)
        # --- TAVAN: eksen uzayina OPTIMAL izdusum = sqrt(sum rho_i^2)
        Ui = U[idx]
        ri = (Ui @ (w * rr)) / (nrow * np.sqrt(m0))
        tav2 = float((ri**2).sum())
        tav_null = float(
            np.mean(
                [
                    float((((Ui @ (w * rr[s])) / (nrow * np.sqrt(m0))) ** 2).sum())
                    for s in PERMS[:30]
                ]
            )
        )
        out["tavan"] = float(np.sqrt(tav2))
        out["tavan_null"] = float(np.sqrt(tav_null))
        out["tavan_duz"] = float(np.sqrt(max(tav2 - tav_null, 0.0)))
        # --- B BLOK taramasi
        out["B"] = {}
        boot = {}
        for B in B_LISTE:
            cur = maske_B(B, idx)
            Gd = _gs_blok(cur, idx)
            tam, bs = _rho2_vek(Gd, BW)
            nul = float(np.mean([sum(kor_w(w, rr[s], m0, g) ** 2 for g in Gd) for s in PERMS[:40]]))
            boot[B] = bs
            out["B"][str(B)] = dict(
                nB=len(Gd),
                rho2=tam,
                rho=float(np.sqrt(tam)),
                null2=nul,
                rho_duz=float(np.sqrt(max(tam - nul, 0.0))),
                lo=float(np.sqrt(max(np.quantile(bs, 0.05), 0.0))),
                hi=float(np.sqrt(max(np.quantile(bs, 0.95), 0.0))),
            )
        # B=4 -> 5 esli fark (rho^2 biriminde)
        d = boot[5] - boot[4]
        out["B45_fark_rho2"] = float(out["B"]["5"]["rho2"] - out["B"]["4"]["rho2"])
        out["B45_lo"] = float(np.quantile(d, 0.05))
        out["B45_hi"] = float(np.quantile(d, 0.95))
        d6 = boot[6] - boot[4]
        out["B46_fark_rho2"] = float(out["B"]["6"]["rho2"] - out["B"]["4"]["rho2"])
        out["B46_lo"] = float(np.quantile(d6, 0.05))
        out["B46_hi"] = float(np.quantile(d6, 0.95))
        # TEK = B=1 (isaretli)
        duz = KATS[idx] @ U[idx]
        out["tek_isaretli"] = kor_w(w, rr, m0, duz)
        return out, boot

    rows, BOOTK = [], {}
    for K in [K for K in K_LISTE if K <= NE]:
        _o, _b = olc(K)
        rows.append(_o)
        BOOTK[K] = _b
    # K'lar arasi ESLI karsilastirma (ayni bootstrap agirliklari kullanildi)
    KL = [K for K in K_LISTE if K <= NE]
    for r, K in zip(rows, KL):
        r["K_esli"] = {}
        for K2 in KL:
            if K2 == K:
                continue
            d = BOOTK[K2][4] - BOOTK[K][4]
            r["K_esli"][str(K2)] = dict(
                fark_rho2=float(
                    BOOTK[K2][4].mean() * 0
                    + rows[KL.index(K2)]["B"]["4"]["rho2"]
                    - r["B"]["4"]["rho2"]
                ),
                lo=float(np.quantile(d, 0.05)),
                hi=float(np.quantile(d, 0.95)),
                p_pozitif=float((d > 0).mean()),
            )
    SONUC[blok] = rows
    ic = "IC-ORNEK (secim burada yapildi)" if blok == "yaz25" else "DIS-ORNEK"
    print(f"  --- {ic} ---")
    bas = (
        f"{'K':>5s} {'P_K':>7s} {'TEK(B=1)':>9s} |"
        + "".join(f"{'B=' + str(B):>8s}" for B in B_LISTE)
        + f" | {'TAVAN':>7s} {'TAVANduz':>9s}"
    )
    print(bas)
    for r in rows:
        print(
            f"{r['K']:5d} {r['P']:7.4f} {r['tek_isaretli']:+9.4f} |"
            + "".join(f"{r['B'][str(B)]['rho']:8.4f}" for B in B_LISTE)
            + f" | {r['tavan']:7.4f} {r['tavan_duz']:9.4f}"
        )
    print(
        f"  {'':5s} {'':7s} {'GURULTU ':>9s} |"
        + "".join(f"{np.sqrt(rows[-1]['B'][str(B)]['null2']):8.4f}" for B in B_LISTE)
        + f" | {rows[-1]['tavan_null']:7.4f}   (K={rows[-1]['K']} bos deneme)"
    )
    print(
        f"  B=4 -> B=5 kazanc (rho^2), K={rows[-1]['K']}: "
        f"{rows[-1]['B45_fark_rho2']:+.6f} [%90 GA {rows[-1]['B45_lo']:+.6f}, "
        f"{rows[-1]['B45_hi']:+.6f}]"
    )
    del U
    import gc

    gc.collect()

if not SONUC:  # N09_ATLA_BLOK=1 -> onceki kosunun sonuclarini oku
    with open(os.path.join(M29, "n09_K_karari.json"), encoding="utf-8") as fh:
        SONUC = json.load(fh)["olcum"]

# ------------------------------------------------- DIS-ORNEK BIRLESIK OZET
print("\n\n############ DIS-ORNEK BIRLESIK (guz25 + kis26, satir-agirlikli) ############")
AG_B = {"guz25": 319732.0, "kis26": 444076.0}
TOP = sum(AG_B.values())
BIRL = []
for i, K in enumerate([K for K in K_LISTE if K <= NE]):
    d = dict(K=K, P=SONUC["guz25"][i]["P"])
    for alan in ("tavan", "tavan_duz", "tek_isaretli"):
        d[alan] = sum(AG_B[b] * SONUC[b][i][alan] for b in ("guz25", "kis26")) / TOP
    d["B"] = {}
    for B in B_LISTE:
        d["B"][str(B)] = {
            k: sum(AG_B[b] * SONUC[b][i]["B"][str(B)][k] for b in ("guz25", "kis26")) / TOP
            for k in ("rho", "rho_duz", "lo", "hi")
        }
    for alan in ("B45_fark_rho2", "B45_lo", "B45_hi", "B46_fark_rho2"):
        d[alan] = sum(AG_B[b] * SONUC[b][i][alan] for b in ("guz25", "kis26")) / TOP
    d["K_esli"] = {
        k2: {
            f: sum(AG_B[b] * SONUC[b][i]["K_esli"][k2][f] for b in ("guz25", "kis26")) / TOP
            for f in ("fark_rho2", "lo", "hi", "p_pozitif")
        }
        for k2 in SONUC["guz25"][i]["K_esli"]
    }
    BIRL.append(d)
print(
    f"{'K':>5s} {'P_K(ongoru)':>12s} {'TEK':>8s} |"
    + "".join(f"{'B=' + str(B):>8s}" for B in B_LISTE)
    + f" | {'TAVAN':>7s} {'TAVANduz':>9s}"
)
for d in BIRL:
    print(
        f"{d['K']:5d} {d['P']:12.4f} {d['tek_isaretli']:+8.4f} |"
        + "".join(f"{d['B'][str(B)]['rho']:8.4f}" for B in B_LISTE)
        + f" | {d['tavan']:7.4f} {d['tavan_duz']:9.4f}"
    )

print("\nK'lar arasi ESLI FARK, B=4, dis-ornek (rho^2 birimi; + = K2 daha iyi)")
print("  taban K -> K2 :  fark  [%90 GA]  P(K2 daha iyi)")
for tK in (40, 50, 63):
    d0 = next(d for d in BIRL if d["K"] == tK)
    for k2 in sorted(d0["K_esli"], key=int):
        q = d0["K_esli"][k2]
        print(
            f"  K={tK:3d} -> {int(k2):3d}: {q['fark_rho2']:+.6f} "
            f"[{q['lo']:+.6f},{q['hi']:+.6f}]  P={q['p_pozitif']:.2f}"
        )

print("\nB=4 -> B=5 KAZANC (rho^2), dis-ornek birlesik:")
for d in BIRL:
    print(
        f"  K={d['K']:4d}: {d['B45_fark_rho2']:+.6f}  [%90 GA {d['B45_lo']:+.6f}, "
        f"{d['B45_hi']:+.6f}]   (B=6 farki {d['B46_fark_rho2']:+.6f})"
    )

with open(os.path.join(M29, "n09_K_karari.json"), "w", encoding="utf-8") as fh:
    json.dump(
        dict(
            taban_mse=TABAN_MSE,
            n_eksen=NE,
            eksenler=KUL,
            KATS=[float(x) for x in KATS],
            rho_s=[float(x) for x in RHOS],
            aile=SEC["aile"],
            blok_adlari=_SIRA,
            blok_agirlik={k: _AG[k] for k in _SIRA},
            K_liste=[K for K in K_LISTE if K <= NE],
            B_liste=B_LISTE,
            olcum=SONUC,
            dis_ornek_birlesik=BIRL,
        ),
        fh,
        indent=1,
    )
print("\nyazildi: n09_K_karari.json")

# ===========================================================================
# 3) YAZ25 ICI ZAMAN BOLMESI -- MEVSIM ESLESEN VEKIL
#
# Koordinator uyarisi (dogru): test ufku Nis-Tem 2026, yaz25 Nis-Tem 2025.
# guz25 (Agu-Kas) ve kis26 (Ara-Mar) FARKLI MEVSIMLERDIR; yukaridaki
# blok-disi olcumler mevsim kaymasini da iceriyor, bu yuzden bu bolumdeki
# yaz25-ici zaman bolmesi ASIL vekildir.
#
#   A yarisi = Nis-May 2025   B yarisi = Haz-Tem 2025   (iki yon de olculur)
#   ISARET  : yalniz FIT yarisindan (m148'in kor'u gibi, HAM eksen uzerinde)
#   GENLIK  : 1.95*|rho_s| -- LB'den gelir, yaridan BAGIMSIZ
#   OLCUM   : olcum yarisinda agirlikli GS + korelasyon
#
# KALAN SIZINTI (durust olmak icin): eksenlerin SECIMI (plasebo z>=3 ve
# TAVAN |rho_cv|>=1.95|rho_s| kapilari) TUM yaz25'e bakarak yapildi, yani
# olcum yarisi da secimde kullanildi. Bu, asagidaki sayilari YUKARI
# yanlilastirir; K buyudukce yanlilik BUYUR, dolayisiyla K egrisinin
# dususu ancak DAHA GERCEK olabilir, daha az degil.
# ===========================================================================
print("\n\n############ YAZ25 ICI ZAMAN BOLMESI (mevsim eslesen vekil) ############")
K_BOL = [10, 17, 25, 40, 63, 100, 136]
ff, pb, rr_t, w_t = blok_kur("yaz25")
kurY = Kur(ff, pb)
ay = pd.to_datetime(ff.tarih).dt.month.to_numpy()
YARI = {"NisMay": np.isin(ay, [4, 5]), "HazTem": np.isin(ay, [6, 7])}
print({k: int(v.sum()) for k, v in YARI.items()})

# eksenleri BIR kez kur (tum yaz25), sonra yarilara dilimle
XY = np.zeros((NE, len(rr_t)), dtype=np.float32)
XOK = np.zeros(NE, dtype=bool)
for i, ad in enumerate(KUL):
    _, xb = kurY(ad)
    if xb is None or not np.isfinite(xb).all():
        continue
    XY[i] = xb.astype(np.float32)
    XOK[i] = True
print(f"  {int(XOK.sum())}/{NE} eksen yaz25'te kuruldu")

BOL_SONUC = {}
for FIT, OLC in (("NisMay", "HazTem"), ("HazTem", "NisMay")):
    mf, mo = YARI[FIT], YARI[OLC]

    def hazirla(m):
        sg = ff.soguk_mu.values.astype(np.float64)[m]
        w = np.where(sg == 1, HEDEF_SOGUK / sg.mean(), (1 - HEDEF_SOGUK) / (1 - sg.mean()))
        w = w / w.mean()
        r = rr_t[m]
        return w, r, float((w * r * r).mean())

    wf, rf, m0f = hazirla(mf)
    wo, ro, m0o = hazirla(mo)
    # --- ISARET: yalniz FIT yarisindan, HAM eksen uzerinde (m148'in kor'u)
    Xf = XY[:, mf].astype(np.float64)
    korf = (Xf @ (wf * rf)) / (mf.sum() * np.sqrt(m0f))
    del Xf
    ISR_F = np.sign(korf)
    ISR_F[ISR_F == 0] = 1.0
    KATS_F = ISR_F * TAVAN * np.abs(RHOS)
    tam_isr = np.sign(KATS)
    uyum = float((ISR_F[XOK] == tam_isr[XOK]).mean())
    # --- OLCUM yarisinda GS
    nrow = int(mo.sum())
    Uo = np.zeros((NE, nrow), dtype=np.float64)
    canli = np.zeros(NE, dtype=bool)
    for i in range(NE):
        if not XOK[i]:
            continue
        u = XY[i, mo].astype(np.float64)
        s = np.sqrt(float((wo * (u - u.mean()) ** 2).mean()))
        u = u - u.mean()
        if s < 1e-12:
            continue
        u = u / s
        for j in range(i):
            if canli[j]:
                u -= float((wo * u * Uo[j]).mean()) * Uo[j]
        n1 = np.sqrt(float((wo * u * u).mean()))
        if n1 < NB_ALT:
            continue
        Uo[i] = u / n1
        canli[i] = True
    tn = ff.tanim.values[mo]
    uq, gidx = np.unique(tn, return_inverse=True)
    ng = len(uq)
    rngp = np.random.default_rng(11)
    PERMS = [
        np.argsort(np.argsort(rngp.permutation(ng)[gidx], kind="stable"), kind="stable")
        for _ in range(60)
    ]
    s3 = np.bincount(gidx, weights=wo * ro * ro, minlength=ng)
    BW = np.array(
        [rng.multinomial(ng, np.full(ng, 1.0 / ng)).astype(np.float64) for _ in range(NBOOT)]
    )
    print(
        f"\n--- FIT {FIT} -> OLCUM {OLC}: {nrow:,} satir, "
        f"{int(canli.sum())}/{NE} eksen ayakta, isaret uyumu(tam yaz25 ile) {uyum:.3f}"
    )

    def _rho2(Gd):
        if not Gd:
            return 0.0, np.zeros(NBOOT)
        S1 = np.array([np.bincount(gidx, weights=wo * ro * g, minlength=ng) for g in Gd])
        S2 = np.array([np.bincount(gidx, weights=wo * g * g, minlength=ng) for g in Gd])
        one = np.ones(ng)
        t1, t2, t3 = S1 @ one, S2 @ one, float(one @ s3)
        tam = float((np.where(t2 > 0, t1 / np.sqrt(np.maximum(t2, 1e-300) * t3), 0.0) ** 2).sum())
        T1, T2, T3 = S1 @ BW.T, S2 @ BW.T, BW @ s3
        q = np.where(
            (T2 > 0) & (T3 > 0), T1 / np.sqrt(np.maximum(T2, 1e-300) * np.maximum(T3, 1e-300)), 0.0
        )
        return tam, (q**2).sum(axis=0)

    rows, BK = [], {}
    for K in K_BOL:
        m = canli.copy()
        m[K:] = False
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        Ui = Uo[idx]
        ri = (Ui @ (wo * ro)) / (nrow * np.sqrt(m0o))
        tav2 = float((ri**2).sum())
        tavnull = float(
            np.mean(
                [
                    float((((Ui @ (wo * ro[s])) / (nrow * np.sqrt(m0o))) ** 2).sum())
                    for s in PERMS[:30]
                ]
            )
        )
        out = dict(
            K=K,
            n=len(idx),
            P=float(np.sqrt((KATS_F[idx] ** 2).sum())),
            tavan=float(np.sqrt(tav2)),
            tavan_null=float(np.sqrt(tavnull)),
            tavan_duz=float(np.sqrt(max(tav2 - tavnull, 0.0))),
            B={},
        )
        boot = {}
        for B in B_LISTE:
            cur = maske_B(B, idx)
            Gd = []
            for _ad, mm in cur:
                sel = idx[mm]
                if len(sel) == 0:
                    continue
                v = KATS_F[sel] @ Uo[sel]
                n0 = np.sqrt(float((wo * v * v).mean()))
                if n0 < 1e-12:
                    continue
                v = v / n0
                for g in Gd:
                    v = v - float((wo * v * g).mean()) * g
                n1 = np.sqrt(float((wo * v * v).mean()))
                if n1 < 0.05:
                    continue
                Gd.append(v / n1)
            tam, bs = _rho2(Gd)
            nul = float(
                np.mean([sum(kor_w(wo, ro[s], m0o, g) ** 2 for g in Gd) for s in PERMS[:40]])
            )
            boot[B] = bs
            out["B"][str(B)] = dict(
                nB=len(Gd),
                rho2=tam,
                rho=float(np.sqrt(tam)),
                null2=nul,
                rho_duz=float(np.sqrt(max(tam - nul, 0.0))),
                lo=float(np.sqrt(max(np.quantile(bs, 0.05), 0.0))),
                hi=float(np.sqrt(max(np.quantile(bs, 0.95), 0.0))),
            )
        # KARAR ESIGI: LB'de 0.001 rho^2 kazanmak, blok korelasyon biriminde
        # 0.001 / CARPAN^2 = 0.00157 demektir (rho_LB = CARPAN * kor_blok).
        ESIK_BLOK = 0.001 / (CARPAN**2)
        for Bx in (5, 6):
            d = boot[Bx] - boot[4]
            out[f"B4{Bx}_fark_rho2"] = float(out["B"][str(Bx)]["rho2"] - out["B"]["4"]["rho2"])
            out[f"B4{Bx}_lo"] = float(np.quantile(d, 0.05))
            out[f"B4{Bx}_hi"] = float(np.quantile(d, 0.95))
            out[f"B4{Bx}_P"] = float((d > 0).mean())
            out[f"B4{Bx}_P_esik"] = float((d > ESIK_BLOK).mean())
            out[f"B4{Bx}_medyan"] = float(np.median(d))
        duz = KATS_F[idx] @ Uo[idx]
        out["tek_isaretli"] = kor_w(wo, ro, m0o, duz)
        BK[K] = boot
        rows.append(out)
    KLb = [r["K"] for r in rows]
    for r in rows:
        r["K_esli"] = {}
        for K2 in KLb:
            if r["K"] == K2:
                continue
            d = BK[K2][4] - BK[r["K"]][4]
            r["K_esli"][str(K2)] = dict(
                fark_rho2=float(
                    next(q for q in rows if q["K"] == K2)["B"]["4"]["rho2"] - r["B"]["4"]["rho2"]
                ),
                lo=float(np.quantile(d, 0.05)),
                hi=float(np.quantile(d, 0.95)),
                p_pozitif=float((d > 0).mean()),
            )
    BOL_SONUC[f"{FIT}->{OLC}"] = dict(isaret_uyumu=uyum, n=nrow, satir=rows)
    print(
        f"{'K':>5s} {'P_K':>7s} {'TEK':>8s} |"
        + "".join(f"{'B=' + str(B):>8s}" for B in B_LISTE)
        + f" | {'TAVAN':>7s} {'TAVANduz':>9s}"
    )
    for r in rows:
        print(
            f"{r['K']:5d} {r['P']:7.4f} {r['tek_isaretli']:+8.4f} |"
            + "".join(f"{r['B'][str(B)]['rho']:8.4f}" for B in B_LISTE)
            + f" | {r['tavan']:7.4f} {r['tavan_duz']:9.4f}"
        )
    print(
        f"{'':5s} {'':7s} {'GURULTU':>8s} |"
        + "".join(f"{np.sqrt(rows[-1]['B'][str(B)]['null2']):8.4f}" for B in B_LISTE)
        + f" | {rows[-1]['tavan_null']:7.4f}"
    )
    del Uo
    import gc

    gc.collect()

# ---- iki yonun satir-agirlikli ortalamasi
print("\n### YAZ25 ZAMAN BOLMESI -- IKI YONUN ORTALAMASI ###")
YY = list(BOL_SONUC)
AGY = {k: float(BOL_SONUC[k]["n"]) for k in YY}
TY = sum(AGY.values())
ORT = []
for i, K in enumerate([r["K"] for r in BOL_SONUC[YY[0]]["satir"]]):
    d = dict(K=K)
    for alan in ("P", "tavan", "tavan_duz", "tek_isaretli"):
        d[alan] = sum(AGY[y] * BOL_SONUC[y]["satir"][i][alan] for y in YY) / TY
    d["B"] = {
        str(B): {
            k: sum(AGY[y] * BOL_SONUC[y]["satir"][i]["B"][str(B)][k] for y in YY) / TY
            for k in ("rho", "rho_duz", "rho2", "lo", "hi")
        }
        for B in B_LISTE
    }
    for alan in (
        "B45_fark_rho2",
        "B45_lo",
        "B45_hi",
        "B45_P",
        "B45_P_esik",
        "B45_medyan",
        "B46_fark_rho2",
        "B46_lo",
        "B46_hi",
        "B46_P",
        "B46_P_esik",
        "B46_medyan",
    ):
        d[alan] = sum(AGY[y] * BOL_SONUC[y]["satir"][i][alan] for y in YY) / TY
    d["K_esli"] = {
        k2: {
            f: sum(AGY[y] * BOL_SONUC[y]["satir"][i]["K_esli"][k2][f] for y in YY) / TY
            for f in ("fark_rho2", "lo", "hi", "p_pozitif")
        }
        for k2 in BOL_SONUC[YY[0]]["satir"][i]["K_esli"]
    }
    ORT.append(d)
print(
    f"{'K':>5s} {'P_K':>7s} {'TEK':>8s} |"
    + "".join(f"{'B=' + str(B):>8s}" for B in B_LISTE)
    + f" | {'TAVAN':>7s} {'TAVANduz':>9s}"
)
for d in ORT:
    print(
        f"{d['K']:5d} {d['P']:7.4f} {d['tek_isaretli']:+8.4f} |"
        + "".join(f"{d['B'][str(B)]['rho']:8.4f}" for B in B_LISTE)
        + f" | {d['tavan']:7.4f} {d['tavan_duz']:9.4f}"
    )

print("\nK=25 TABANLI ESLI FARKLAR (B=4, rho^2; + = K2 daha iyi):")
d25 = next(d for d in ORT if d["K"] == 25)
for k2 in sorted(d25["K_esli"], key=int):
    q = d25["K_esli"][k2]
    print(
        f"  K=25 -> {int(k2):4d}: {q['fark_rho2']:+.6f} [{q['lo']:+.6f},{q['hi']:+.6f}] "
        f"P(K2 iyi)={q['p_pozitif']:.2f}"
    )

print("\nB=4 -> B=5 KAZANC (rho^2), yaz25 zaman bolmesi:")
for d in ORT:
    print(
        f"  K={d['K']:4d}: {d['B45_fark_rho2']:+.6f} [%90 GA {d['B45_lo']:+.6f},"
        f"{d['B45_hi']:+.6f}]  medyan {d['B45_medyan']:+.6f}  P(>0)={d['B45_P']:.2f} "
        f"P(>LB 0.001)={d['B45_P_esik']:.2f}\n          "
        f"(B=6: {d['B46_fark_rho2']:+.6f} [{d['B46_lo']:+.6f},{d['B46_hi']:+.6f}] "
        f"P(>LB 0.001)={d['B46_P_esik']:.2f})"
    )

with open(os.path.join(M29, "n09_K_karari.json"), encoding="utf-8") as fh:
    _J = json.load(fh)
_J["yaz25_zaman_bolmesi"] = dict(
    yonler=BOL_SONUC,
    ortalama=ORT,
    K_liste=K_BOL,
    uyari="guz25/kis26 mevsim eslesmiyor; ASIL vekil budur",
)
with open(os.path.join(M29, "n09_K_karari.json"), "w", encoding="utf-8") as fh:
    json.dump(_J, fh, indent=1)
print("\nyazildi: n09_K_karari.json (+ yaz25_zaman_bolmesi)")
