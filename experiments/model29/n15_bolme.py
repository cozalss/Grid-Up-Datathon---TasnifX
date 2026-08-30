# ruff: noqa: F821  -- kapanis degiskenleri; betikler kosup json uretti
"""DORT BLOK DAHA IYI SECILEBILIR MI?  (K=25 eksen, yaz25-ici zaman vekili)

SORU. m148 25 dik eksenden BETA = toplam KATS[i]*U[i] kuruyor ve BETA'yi
DORT DIK BLOGA boluyor. Nihai skor^2 = 1.00202690 - toplam_k rho_k^2.
toplam_k rho_k^2 = ||P_A r||^2, A = bloklarin gerdigi 4 boyutlu ALT UZAY.
Yani onemli olan blok SAYISI degil, alt uzayin NEREYE baktigi.

GEOMETRI (hesabi 25 boyuta indiren gozlem).
  Olcum yarisinda eksenler agirlikli Gram-Schmidt ile ortonormal U yapilir.
  Bir blok yonu  v_b = toplam_{i in b} KATS[i] U[i]  yani KOORDINATTA
  c_b vektorudur (blok disinda sifir). Bloklar AYRIK oldugu icin c_b'ler
  BIRBIRINE DIKTIR (tam ornekte U ortonormal). Dolayisiyla GS hicbir sey
  yapmaz, sadece normalize eder ve

      toplam_k rho_k^2 = toplam_b (c_b . ri)^2 / ||c_b||^2 ,
      ri[i] = <w r, U_i> / (n sqrt(m0))     (25 boyutlu "gercek yon")

  Bu m148/n09'un olctugu seyin BIREBIR aynisidir, ama 25 boyutta doner --
  boylece BINLERCE rastgele bolme taranabilir.

VEKIL (n09'dan AYNEN). yaz25 icinde zaman bolmesi:
  A = Nis-May 2025, B = Haz-Tem 2025; iki yon de olculur.
  ISARET yalniz FIT yarisindan; GENLIK 1.95*|rho_s| (LB'den, yaridan bagimsiz);
  OLCUM yalniz olcum yarisinda. guz25/kis26 mevsim eslesmedigi icin
  KULLANILMAZ.

ADAYLAR (hepsi 4 blok, hepsi ayni vekil, ESLI onyukleme):
  a  MEVCUT (m148, BLOK_KIP=oran): {hava,yapi} x {|rho_cv|/|KATS| yuksek,dusuk}
     a_fit : oran FIT yarisindan (durust)   a_tam : oran tum yaz25'ten (m148 birebir)
  a9 n09'un "aile" kipi bolmesi (0.1814 sayisinin geldigi bolme)
  b  {hava,yapi} x {|rho_s| yuksek,dusuk}
  c  FIT yariminda optimal agirlik w*; (w*_i / KATS_i) oranina gore 4 esit parca
  d  eksenlerin FIT yarisindaki korelasyon yapisinin PCA'si; her eksen en cok
     yuklendigi ana yone atanir
  e  |R| uzerinde |KATS| agirlikli k-ortalamalar, 4 kume
  g  DOGRUDAN ARAMA: FIT yariminda ||P_A r_fit||^2'yi maksimize eden bolme
     (cok baslangicli yerel arama), OLCUM yarisinda olculur
  f  RASTGELE 4000 bolme -- SANS TABANI. Kazanan bunun kuyrugundan
     ayirt edilemiyorsa kazanc YOKTUR.

CIKTI: experiments/model29/n15_bolme.json
HICBIR GONDERIM YAZILMAZ. m148 DEGISTIRILMEZ.
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
SCR = os.path.join(
    r"C:/Users/Cem/AppData/Local/Temp/claude",
    "c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX",
    "e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad",
)
TABAN = "tuketim_m6_ikiyon.csv"
HEDEF_SOGUK, CARPAN, TAVAN = 0.222, 0.798, 1.95
NB_ALT = 0.02
K = 25
NBOOT = 2000
NPERM = 200
NRAST = 4000
BLOK_ALT = 0.06
DEMET_HEDEF = 4

# --- secim onbellegi (n09_K_karari.py'nin urettigi, BIREBIR ayni kume) -----
SEC_YOL = os.path.join(SCR, "n09_secim.json")
if not os.path.exists(SEC_YOL):
    raise SystemExit(f"DUR: {SEC_YOL} yok. once n09_K_karari.py kosulmali.")
with open(SEC_YOL, encoding="utf-8") as fh:
    SEC = json.load(fh)
KUL = SEC["kul"]
KATS_TAM = np.array(SEC["KATS"])
RHOS = np.array(SEC["rho_s"])
RHOCV_TAM = np.array(SEC["rho_cv"])
AILE = np.array(SEC["aile"])
NE = len(KUL)
print(f"secim onbellekten: {NE} eksen, K={K} kullanilacak")

# --- eksen kurulusu (n09'un Kur sinifi, BIREBIR) ---------------------------
tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
e = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
_d = pd.read_csv(os.path.join(S, TABAN))
_k = "tuketim" if "tuketim" in _d.columns else _d.columns[-1]
a0 = np.log1p(_d[_k].values.astype(np.float64))
del _d


def st(x):
    x = np.asarray(x, dtype=np.float64).copy()
    f = np.isfinite(x)
    if not f.any():
        return None
    x[~f] = np.median(x[f])
    x -= x.mean()
    s = np.sqrt(float((x * x).mean()))
    return x / s if s > 1e-12 else None


sgT = tp.soguk_mu.values.astype(np.float64)
svT = st(a0)
ufT = st(tp.ufuk_gun.to_numpy())
ayT = st(pd.to_datetime(tp.tarih).dt.month.to_numpy().astype(np.float64))
ESIK = {"ust10": (0.9, True), "ust25": (0.75, True), "alt10": (0.1, False)}


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


class Kur:
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
            return (None, None) if a_ is None or b_ is None else (st(a_**2), st(b_**2))
        return st(xt), st(xb)


# ---------------------------------------------------------------- yaz25 kur
ff, pb, rr_t, _w_t = blok_kur("yaz25")
kurY = Kur(ff, pb)
ay = pd.to_datetime(ff.tarih).dt.month.to_numpy()
YARI = {"NisMay": np.isin(ay, [4, 5]), "HazTem": np.isin(ay, [6, 7])}
print({k: int(v.sum()) for k, v in YARI.items()})

XY = np.zeros((K, len(rr_t)), dtype=np.float32)
XOK = np.zeros(K, dtype=bool)
for i, ad in enumerate(KUL[:K]):
    _, xb = kurY(ad)
    if xb is None or not np.isfinite(xb).all():
        continue
    XY[i] = xb.astype(np.float32)
    XOK[i] = True
print(f"  {int(XOK.sum())}/{K} eksen yaz25'te kuruldu")

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
HV = np.array([bool(any(h in a for h in HAVA)) for a in KUL[:K]])
print(f"  hava eksenleri: {int(HV.sum())}, yapi: {int((~HV).sum())}")


# ======================================================= BOLME URETICILERI
def _birlestir_4(HAM, kk):
    """m148/n09'un blok birlestirme kurali: kucukleri kat, 4'e indir."""
    HAM = {k: v.copy() for k, v in HAM.items() if v.sum()}
    AG = {k: float(np.sqrt((kk[m] ** 2).sum())) for k, m in HAM.items()}
    while True:
        kucuk = [k for k, v in AG.items() if v < BLOK_ALT]
        if not kucuk or len(AG) <= DEMET_HEDEF:
            break
        k1 = min(kucuk, key=lambda k: AG[k])
        hed = min((k for k in AG if k != k1), key=lambda k: AG[k])
        HAM[hed] = HAM[hed] | HAM[k1]
        del HAM[k1], AG[k1]
        AG[hed] = float(np.sqrt((kk[HAM[hed]] ** 2).sum()))
    while len(AG) > DEMET_HEDEF:
        k1 = min(AG, key=lambda k: AG[k])
        hed = min((k for k in AG if k != k1), key=lambda k: AG[k])
        HAM[hed] = HAM[hed] | HAM[k1]
        del HAM[k1], AG[k1]
        AG[hed] = float(np.sqrt((kk[HAM[hed]] ** 2).sum()))
    sira = sorted(AG, key=lambda k: -AG[k])
    return [(k, HAM[k]) for k in sira]


def bol_oran(hv, oran, kk):
    """m148 BLOK_KIP=oran: {hava,yapi} x {oran yuksek, dusuk}, GRUP ICI medyan."""
    yuk = np.zeros(len(hv), dtype=bool)
    for msk in (hv, ~hv):
        if msk.sum() >= 2:
            yuk |= msk & (oran > float(np.median(oran[msk])))
    HAM = {
        "hava/oran-yuksek": hv & yuk,
        "hava/oran-dusuk": hv & ~yuk,
        "yapi/oran-yuksek": (~hv) & yuk,
        "yapi/oran-dusuk": (~hv) & ~yuk,
    }
    return _birlestir_4(HAM, kk)


def bol_etiket(et, kk, on=""):
    """etiket dizisinden (0..3) bolme uret."""
    HAM = {}
    for v in sorted(set(et.tolist())):
        HAM[f"{on}k{v}"] = np.equal(et, v)
    return _birlestir_4(HAM, kk)


def bol_dortlu_sirali(skor, kk, on="q"):
    """skora gore sirala, DORT ESIT PARCAYA bol."""
    n = len(skor)
    sr = np.argsort(skor, kind="stable")
    et = np.zeros(n, dtype=int)
    kes = [int(round(n * t / 4)) for t in range(5)]
    for t in range(4):
        et[sr[kes[t] : kes[t + 1]]] = t
    return bol_etiket(et, kk, on)


def kmeans_ag(Xf, wgt, kkume, rng, nbas=20, nit=60):
    """agirlikli k-ortalamalar, cok baslangicli."""
    n = len(Xf)
    en_iyi, en_maliyet = None, np.inf
    for _ in range(nbas):
        c = Xf[rng.choice(n, kkume, replace=False)].copy()
        et = np.zeros(n, dtype=int)
        for _t in range(nit):
            d = ((Xf[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
            et_y = d.argmin(axis=1)
            if _t and (et_y == et).all():
                break
            et = et_y
            for q in range(kkume):
                m = et == q
                if m.sum():
                    c[q] = (wgt[m, None] * Xf[m]).sum(axis=0) / max(wgt[m].sum(), 1e-12)
        d = ((Xf[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
        mal = float((wgt * d[np.arange(n), et]).sum())
        if mal < en_maliyet:
            en_maliyet, en_iyi = mal, et.copy()
    return en_iyi


def rastgele_etiket(n, rng):
    """4 dolu kumeye rastgele atama."""
    while True:
        et = rng.integers(0, 4, size=n)
        if len(np.unique(et)) == 4:
            return et


def deger(et, kk, ri):
    """toplam_b (c_b.ri)^2/||c_b||^2 -- bloklar AYRIK oldugu icin tam izdusum."""
    s = 0.0
    for q in np.unique(et):
        m = et == q
        c = kk[m]
        nn = float((c * c).sum())
        if nn <= 0:
            continue
        s += float((c @ ri[m]) ** 2) / nn
    return s


def yerel_arama(kk, ri, rng, nbas=60, nit=200):
    """FIT yarisinda deger()'i maksimize eden bolme (cok baslangicli tepe tirmanma)."""
    n = len(kk)
    en_et, en_d = None, -1.0
    for _ in range(nbas):
        et = rastgele_etiket(n, rng)
        d = deger(et, kk, ri)
        for _t in range(nit):
            gelisti = False
            for i in rng.permutation(n):
                eski = et[i]
                for q in range(4):
                    if q == eski:
                        continue
                    et[i] = q
                    if len(np.unique(et)) < 4:
                        et[i] = eski
                        continue
                    d2 = deger(et, kk, ri)
                    if d2 > d + 1e-15:
                        d, eski, gelisti = d2, q, True
                    else:
                        et[i] = eski
                et[i] = eski
            if not gelisti:
                break
        if d > en_d:
            en_d, en_et = d, et.copy()
    return en_et, en_d


# ============================================================ ANA DONGU
rng = np.random.default_rng(20260831)
SONUC = {}
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

    # --- ISARET yalniz FIT yarisindan (m148'in kor'u, HAM eksen uzerinde)
    Xf = XY[:, mf].astype(np.float64)
    korf = (Xf @ (wf * rf)) / (int(mf.sum()) * np.sqrt(m0f))
    ISR_F = np.sign(korf)
    ISR_F[ISR_F == 0] = 1.0
    KATS_F = ISR_F * TAVAN * np.abs(RHOS[:K])
    RHOCV_F = CARPAN * korf
    uyum = float((ISR_F[XOK] == np.sign(KATS_TAM[:K])[XOK]).mean())

    # --- FIT yarisinda ortonormallestirme (w*, PCA, k-ort, arama icin)
    nf = int(mf.sum())
    Uf = np.zeros((K, nf))
    canf = np.zeros(K, dtype=bool)
    for i in range(K):
        if not XOK[i]:
            continue
        u = Xf[i] - Xf[i].mean()
        s = np.sqrt(float((wf * u * u).mean()))
        if s < 1e-12:
            continue
        u = u / s
        for j in range(i):
            if canf[j]:
                u -= float((wf * u * Uf[j]).mean()) * Uf[j]
        n1 = np.sqrt(float((wf * u * u).mean()))
        if n1 < NB_ALT:
            continue
        Uf[i] = u / n1
        canf[i] = True
    ri_f = (Uf @ (wf * rf)) / (nf * np.sqrt(m0f))
    # ham eksenlerin FIT yarisindaki agirlikli korelasyon matrisi (PCA/k-ort)
    Xc = Xf - (wf * Xf).mean(axis=1, keepdims=True) / wf.mean()
    sd = np.sqrt((wf * Xc * Xc).mean(axis=1))
    sd[sd < 1e-12] = 1.0
    Xn = Xc / sd[:, None]
    R = (Xn * wf) @ Xn.T / nf
    del Xf, Xc, Xn

    # --- OLCUM yarisinda GS (n09 ile BIREBIR)
    no = int(mo.sum())
    Uo = np.zeros((K, no))
    canli = np.zeros(K, dtype=bool)
    for i in range(K):
        if not XOK[i]:
            continue
        u = XY[i, mo].astype(np.float64)
        u = u - u.mean()
        s = np.sqrt(float((wo * u * u).mean()))
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
    idx = np.flatnonzero(canli)
    nk = len(idx)
    print(
        f"\n=== FIT {FIT} -> OLCUM {OLC}: {no:,} satir, {nk}/{K} eksen ayakta, "
        f"isaret uyumu(tam yaz25) {uyum:.3f}"
    )

    Ui = Uo[idx]
    ri = (Ui @ (wo * ro)) / (no * np.sqrt(m0o))
    tavan2 = float((ri**2).sum())

    # --- DIKLIK DENETIMI: c_b'lerin dikligi varsayimi gecerli mi?
    Gchk = (Ui * wo) @ Ui.T / no
    sap = float(np.abs(Gchk - np.eye(nk)).max())
    print(f"  U ortonormallik sapmasi: {sap:.2e}")
    if sap > 1e-5:
        raise SystemExit(f"DUR: olcum yarisinda U ortonormal degil (sapma {sap:.2e})")
    del Gchk

    # --- onyukleme / permutasyon altyapisi (trafo kumeleri)
    tn = ff.tanim.values[mo]
    uq, gidx = np.unique(tn, return_inverse=True)
    ng = len(uq)
    A1 = np.array([np.bincount(gidx, weights=wo * ro * u, minlength=ng) for u in Ui])
    A2 = np.array(
        [
            [np.bincount(gidx, weights=wo * Ui[a] * Ui[b], minlength=ng) for b in range(nk)]
            for a in range(nk)
        ]
    )
    s3 = np.bincount(gidx, weights=wo * ro * ro, minlength=ng)
    BW = np.array(
        [rng.multinomial(ng, np.full(ng, 1.0 / ng)).astype(np.float64) for _ in range(NBOOT)]
    )
    B1 = A1 @ BW.T  # (nk, NBOOT)
    B2 = np.tensordot(A2, BW, axes=([2], [1]))  # (nk, nk, NBOOT)
    B3 = BW @ s3  # (NBOOT,)
    rngp = np.random.default_rng(11)
    PERM_RI = []
    for _ in range(NPERM):
        s = np.argsort(np.argsort(rngp.permutation(ng)[gidx], kind="stable"), kind="stable")
        PERM_RI.append((Ui @ (wo * ro[s])) / (no * np.sqrt(m0o)))
    PERM_RI = np.array(PERM_RI)
    del A1, A2

    kk_o = KATS_F[idx]
    hv_o = HV[idx]
    ri_f_o = ri_f[idx]
    R_o = R[np.ix_(idx, idx)]

    def olc_boot(et):
        """(tam deger, (NBOOT,) onyukleme dizisi, permutasyon ortalamasi)"""
        tam = deger(et, kk_o, ri)
        acc = np.zeros(NBOOT)
        for q in np.unique(et):
            m = et == q
            c = np.zeros(nk)
            c[m] = kk_o[m]
            nn = np.sqrt(float((c * c).sum()))
            if nn <= 0:
                continue
            c = c / nn
            t1 = c @ B1
            t2 = np.einsum("a,abp,b->p", c, B2, c)
            gec = (t2 > 0) & (B3 > 0)
            v = np.zeros(NBOOT)
            v[gec] = t1[gec] / np.sqrt(t2[gec] * B3[gec])
            acc += v**2
        nul = float(np.mean([deger(et, kk_o, p) for p in PERM_RI]))
        return tam, acc, nul

    # ---------------------------------------------------- ADAYLAR
    ADAY = {}

    def kaydet(ad, cur):
        et = np.full(nk, -1, dtype=int)
        for q, (_nm, m) in enumerate(cur):
            et[m] = q
        assert (et >= 0).all(), f"{ad}: atanmamis eksen var"
        ADAY[ad] = (et, [nm for nm, _ in cur])

    oran_fit = np.abs(RHOCV_F[idx]) / np.maximum(np.abs(kk_o), 1e-12)
    kaydet("a_fit", bol_oran(hv_o, oran_fit, kk_o))
    oran_tam = np.abs(RHOCV_TAM[:K][idx]) / np.maximum(np.abs(KATS_TAM[:K][idx]), 1e-12)
    kaydet("a_tam", bol_oran(hv_o, oran_tam, kk_o))

    # a9: n09'un aile kipi
    HAM9 = {}
    AIL = AILE[:K][idx]
    for f_ in sorted(set(AIL.tolist())):
        m_ = np.equal(AIL, f_)
        if f_ in ("m121_taban", "H_carpim40"):
            HAM9[f"{f_}/hava"] = m_ & hv_o
            HAM9[f"{f_}/yapi"] = m_ & ~hv_o
        else:
            HAM9[f_] = m_
    kaydet("a9_aile", _birlestir_4(HAM9, kk_o))

    # b: {hava,yapi} x {|rho_s| yuksek,dusuk}  (grup ici medyan)
    rs = np.abs(RHOS[:K][idx])
    yuk = np.zeros(nk, dtype=bool)
    for msk in (hv_o, ~hv_o):
        if msk.sum() >= 2:
            yuk |= msk & (rs > float(np.median(rs[msk])))
    kaydet(
        "b_rhos",
        _birlestir_4(
            {
                "hava/rhos-yuksek": hv_o & yuk,
                "hava/rhos-dusuk": hv_o & ~yuk,
                "yapi/rhos-yuksek": (~hv_o) & yuk,
                "yapi/rhos-dusuk": (~hv_o) & ~yuk,
            },
            kk_o,
        ),
    )

    # c: w* / KATS oranina gore dort esit parca (w* FIT yarisindan)
    kaydet(
        "c_wyildiz",
        bol_dortlu_sirali(ri_f_o / np.where(np.abs(kk_o) > 1e-12, kk_o, 1e-12), kk_o, "c"),
    )

    # d: FIT yarisindaki eksen korelasyon yapisinin PCA'si
    ev, EV = np.linalg.eigh(R_o)
    V4 = EV[:, np.argsort(-ev)[:4]]
    kaydet("d_pca", bol_etiket(np.abs(V4).argmax(axis=1), kk_o, "pca"))

    # e: |R| satirlari uzerinde |KATS| agirlikli k-ortalamalar
    kaydet(
        "e_kort",
        bol_etiket(kmeans_ag(np.abs(R_o), np.abs(kk_o), 4, np.random.default_rng(3)), kk_o, "km"),
    )

    # g: FIT yarisinda dogrudan arama
    et_g, dg_fit = yerel_arama(kk_o, ri_f_o, np.random.default_rng(7))
    ADAY["g_arama"] = (et_g, [f"arama{q}" for q in range(4)])
    print(
        f"  g_arama: FIT yarisinda ||P r_fit||^2 = {dg_fit:.6f} "
        f"(FIT tavani {float((ri_f_o**2).sum()):.6f})"
    )

    # --- olcum
    ADLAR_S = list(ADAY)
    OUT, BOOT = {}, {}
    for ad, (et, adlar) in ADAY.items():
        tam, bs, nul = olc_boot(et)
        OUT[ad] = dict(
            rho2=tam,
            rho=float(np.sqrt(max(tam, 0.0))),
            null2=nul,
            rho_duz=float(np.sqrt(max(tam - nul, 0.0))),
            lo=float(np.sqrt(max(np.quantile(bs, 0.05), 0.0))),
            hi=float(np.sqrt(max(np.quantile(bs, 0.95), 0.0))),
            bloklar=adlar,
            etiket=et.tolist(),
            eksenler={
                str(q): [KUL[int(idx[t])] for t in np.flatnonzero(et == q)] for q in range(4)
            },
        )
        BOOT[ad] = bs

    # --- f: rastgele bolmeler (SANS TABANI)
    rr_rng = np.random.default_rng(101)
    RAST = np.array([deger(rastgele_etiket(nk, rr_rng), kk_o, ri) for _ in range(NRAST)])

    TB = "a_fit"
    for ad in OUT:
        d = BOOT[ad] - BOOT[TB]
        OUT[ad]["fark_vs_a"] = float(OUT[ad]["rho2"] - OUT[TB]["rho2"])
        OUT[ad]["fark_lo"] = float(np.quantile(d, 0.05))
        OUT[ad]["fark_hi"] = float(np.quantile(d, 0.95))
        OUT[ad]["P_iyi"] = float((d > 0).mean())
        OUT[ad]["rast_yuzdelik"] = float((OUT[ad]["rho2"] > RAST).mean())

    SONUC[f"{FIT}->{OLC}"] = dict(
        n=no,
        isaret_uyumu=uyum,
        nk=nk,
        tavan=float(np.sqrt(tavan2)),
        tavan2=tavan2,
        aday=OUT,
        rastgele=dict(
            ort=float(RAST.mean()),
            sd=float(RAST.std()),
            q50=float(np.quantile(RAST, 0.5)),
            q90=float(np.quantile(RAST, 0.9)),
            q99=float(np.quantile(RAST, 0.99)),
            maks=float(RAST.max()),
            n=NRAST,
        ),
        eksen_idx=idx.tolist(),
        eksen_ad=[KUL[int(i)] for i in idx],
        KATS_F=kk_o.tolist(),
    )

    print(f"  TAVAN (25 boyut optimal) = {np.sqrt(tavan2):.4f}")
    print(
        f"  {'aday':>12s} {'rho':>7s} {'rho_duz':>8s} {'%90 GA':>17s} "
        f"{'fark(rho2)':>11s} {'P>a':>5s} {'rast%':>6s}"
    )
    for ad in OUT:
        o = OUT[ad]
        print(
            f"  {ad:>12s} {o['rho']:7.4f} {o['rho_duz']:8.4f} "
            f"[{o['lo']:.4f},{o['hi']:.4f}] {o['fark_vs_a']:+11.6f} "
            f"{o['P_iyi']:5.2f} {100 * o['rast_yuzdelik']:6.1f}"
        )
    print(
        f"  RASTGELE {NRAST} bolme (rho biriminde): ort {np.sqrt(RAST.mean()):.4f} "
        f"medyan {np.sqrt(np.quantile(RAST, 0.5)):.4f} "
        f"%90 {np.sqrt(np.quantile(RAST, 0.9)):.4f} "
        f"%99 {np.sqrt(np.quantile(RAST, 0.99)):.4f} "
        f"maks {np.sqrt(RAST.max()):.4f}"
    )
    np.savez(
        os.path.join(SCR, f"n15_yon_{FIT}.npz"),
        ri=ri,
        kk=kk_o,
        idx=idx,
        hv=hv_o,
        ri_f=ri_f_o,
        R=R_o,
        oran_fit=oran_fit,
        rs=rs,
        B1=B1,
        B2=B2,
        B3=B3,
        PERM_RI=PERM_RI,
        no=no,
        et_ler=np.array([ADAY[a][0] for a in ADLAR_S]),
        adlar=np.array(ADLAR_S),
    )
    del Uo, Ui, B1, B2, PERM_RI
    import gc

    gc.collect()

# ---------------------------------------------------- IKI YONUN ORTALAMASI
YY = list(SONUC)
AG = {k: float(SONUC[k]["n"]) for k in YY}
TY = sum(AG.values())
ORT = {}
ADLAR = list(SONUC[YY[0]]["aday"])
for ad in ADLAR:
    ORT[ad] = {
        f: sum(AG[y] * SONUC[y]["aday"][ad][f] for y in YY) / TY
        for f in (
            "rho",
            "rho2",
            "rho_duz",
            "lo",
            "hi",
            "fark_vs_a",
            "fark_lo",
            "fark_hi",
            "P_iyi",
            "rast_yuzdelik",
        )
    }
ORT_TAVAN = sum(AG[y] * SONUC[y]["tavan"] for y in YY) / TY
ORT_RAST = {
    f: sum(AG[y] * SONUC[y]["rastgele"][f] for y in YY) / TY
    for f in ("ort", "q50", "q90", "q99", "maks")
}

print("\n\n############ IKI YONUN SATIR-AGIRLIKLI ORTALAMASI ############")
print(f"TAVAN (25 boyut, optimal agirlik -- ULASILAMAZ) = {ORT_TAVAN:.4f}")
print(
    f"{'aday':>12s} {'rho':>7s} {'rho_duz':>8s} {'%90 GA':>17s} "
    f"{'fark(rho2)':>11s} {'%90 GA fark':>21s} {'P>a':>5s} {'rast%':>6s}"
)
for ad in sorted(ADLAR, key=lambda a: -ORT[a]["rho2"]):
    o = ORT[ad]
    print(
        f"{ad:>12s} {o['rho']:7.4f} {o['rho_duz']:8.4f} [{o['lo']:.4f},{o['hi']:.4f}] "
        f"{o['fark_vs_a']:+11.6f} [{o['fark_lo']:+.6f},{o['fark_hi']:+.6f}] "
        f"{o['P_iyi']:5.2f} {100 * o['rast_yuzdelik']:6.1f}"
    )
print(
    f"\nRASTGELE bolme dagilimi (rho): ort {np.sqrt(ORT_RAST['ort']):.4f} "
    f"medyan {np.sqrt(ORT_RAST['q50']):.4f} %90 {np.sqrt(ORT_RAST['q90']):.4f} "
    f"%99 {np.sqrt(ORT_RAST['q99']):.4f} maks({NRAST}) {np.sqrt(ORT_RAST['maks']):.4f}"
)


# ---- KAZANAN: iki yonde de a'yi gecmeli, sans kuyrugunun ustunde olmali
def _kazanan():
    en, ei = None, 0.0
    for ad in ADLAR:
        if ad in ("a_fit", "a_tam", "a9_aile"):
            continue
        o = ORT[ad]
        her_yon = all(SONUC[y]["aday"][ad]["fark_vs_a"] > 0 for y in YY)
        if her_yon and o["fark_lo"] > 0 and o["rast_yuzdelik"] > 0.99 and o["fark_vs_a"] > ei:
            en, ei = ad, o["fark_vs_a"]
    return en


KAZ = _kazanan()
print(f"\nKAZANAN: {KAZ if KAZ else 'YOK -- STATUKOYU KORU'}")

J = dict(
    K=K,
    yonler=SONUC,
    ortalama=ORT,
    tavan=ORT_TAVAN,
    rastgele=ORT_RAST,
    taban_aday="a_fit",
    kazanan=KAZ,
    NBOOT=NBOOT,
    NRAST=NRAST,
    not_="vekil: yaz25-ici Nis-May <-> Haz-Tem; guz25/kis26 KULLANILMADI",
)
if KAZ:
    ky = max(YY, key=lambda y: SONUC[y]["n"])
    et = np.array(SONUC[ky]["aday"][KAZ]["etiket"])
    J["m148_takilabilir"] = dict(
        yon=ky,
        blok_adlari=SONUC[ky]["aday"][KAZ]["bloklar"],
        eksen_indeksleri={
            str(q): [int(SONUC[ky]["eksen_idx"][t]) for t in np.flatnonzero(et == q)]
            for q in range(4)
        },
        eksen_adlari=SONUC[ky]["aday"][KAZ]["eksenler"],
    )
with open(os.path.join(M29, "n15_bolme.json"), "w", encoding="utf-8") as fh:
    json.dump(J, fh, indent=1, ensure_ascii=False)
print("\nyazildi: n15_bolme.json")
sys.stdout.flush()
