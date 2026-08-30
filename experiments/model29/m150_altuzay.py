"""ALT UZAY SECIMI -- 6 gonderim hakkiyla olculecek k boyut nasil secilir.

SORU. Bilinmeyen artik r'nin, 40 ortonormal eksenin gerdigi dik uzaydaki
izdusumu rho in R^40. Ortonormal yonlerde

    skor^2 = TABAN_MSE - toplam_k rho_k^2 = TABAN_MSE - ||P_S rho||^2

yani sonuc YALNIZCA secilen S alt uzayina baglidir. Elimizde en fazla 4-5
DIK yonu olcecek hak var. HANGI k boyutlu S, E||P_S rho||^2'yi buyutur?

CEVAP. E||P_S rho||^2 = iz(W Sigma W'), Sigma = E[rho rho'] . Ortonormal W
icin bunu enbuyukleyen S, Sigma'nin en buyuk k ozvektorunun gerdigi uzaydir
(Ky Fan). Yani "alt uzay secimi" = "onsel kovaryansin PCA'i". Butun mesele
Sigma'yi durustce modellemek.

ONSEL MODEL -- DORT DAL (hepsi ayni T = E||rho||^2 enerjisine normalize).
  A) LB dali   : rho_i = c*s_i*eps_i*|rho_s_i| , c ~ (0.7, 0.30)      [m145]
     eps_i = +-1, P(eps=-1) = 0.125 (m142: iki kaynak 35/40 uyusuyor)
  B) CV dali   : rho, rho_cv YONUNDE (buyuklugu DEGIL -- ||rho_cv||=0.69
     olsaydi skor 0.90 olurdu, imkansiz; m148 de bu yuzden kesiyor)
  D) Aile dali : hava ve yapisal ailelerin carpani BAGIMSIZ           [m141]
  C) Hicbiri   : izotropik, modellenmemis.

  DIKKAT: isaret belirsizligi Sigma'ya diag(v_i^2) terimi ekler; bu terim
  izotropik DEGILDIR ve buyuk |rho_s| eksenlerini tek baslarina bir yon
  yapmayi odullendirir. Mevcut kurulusun sinandigi yer tam burasidir.

HEDEF (kullanici, 30 Agu): once 1. SIRA, sonra 2. sira. Bu yuzden yalnizca
E[yakalanan] degil P(1.sira) ve P(2.sira) de Monte Carlo ile raporlanir.

UC BULGU (ayrinti asagidaki bolumlerde):
  * Alt uzay secimi ONEMSIZ: mevcut 4 yon, en iyi alt uzayin 0.00005
    skor yakinindadir. DEGISTIRMEYE DEGMEZ.
  * KAPPA hedge yonlerinde 4 KAT BUYUK: 0.0517 yerine ~0.0125.
  * m148'in rho_k COZUM FORMULU 2. sondadan itibaren YANLIS (bolum 6b).

HICBIR GONDERIM YAPILMAZ, submissions/ altina YAZILMAZ.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")
ONBELLEK = os.path.join(M29, "m150_eksenler.json")
YUV = 5e-6 / np.sqrt(3.0)  # LB 5 hane yuvarlamasinin standart sapmasi

HEDEF = [("1. SIRA", 0.99009), ("2. SIRA", 0.99614), ("3. sira", 0.99927)]


def sira(sk):
    for ad, e in HEDEF:
        if sk < e:
            return ad
    return "4.+"


# ---------------------------------------------------------------------------
# 1) 40 EKSENI YENIDEN KUR -- m148_demet_plani.py'deki secim dongusunun
#    aynisi; TEK FARK: hicbir dosya uretilmez. Sonuc onbellege yazilir.
# ---------------------------------------------------------------------------
def eksenleri_kur():
    DN = os.path.join(KOK, "data/interim/deney")
    AO = os.path.join(KOK, "data/interim/aile_onbellek")
    S = os.path.join(KOK, "submissions")
    TABAN = "tuketim_m6_ikiyon.csv"
    HEDEF_SOGUK, CARPAN, TAVAN = 0.222, 0.798, 1.95
    RHO_S_ALT, AZAMI_EKSEN = 0.015, 40
    sys.path.insert(0, M29)
    from m112_kalibre import EK_MODEL, M0, buzmeli_r_hat

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
    GG = (V.T @ V) / N
    Gi = np.linalg.pinv(GG, rcond=1e-6)
    GI5 = np.linalg.pinv(GG, rcond=1e-5)
    r_hat, _gercek, kL = buzmeli_r_hat(V, L, GG, N)

    tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
    e = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
    blk = e[e._blok == "yaz25"]
    sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
    P = [
        np.load(os.path.join(AO, f"yaz25_{t}_{aa}_uretim.npy")).astype(np.float64)
        for t in (1000, 1001, 1002)
        for aa in ("cat", "xgb", "lgbm")
        if os.path.exists(os.path.join(AO, f"yaz25_{t}_{aa}_uretim.npy"))
    ]
    z = np.load(os.path.join(DN, "soguk_tahmin_yaz25.npz"))
    idx = np.concatenate([sic.index.values, sog.index.values])
    pb = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])
    bf = e.loc[idx].copy()
    rb = np.log1p(bf.tuketim.values.astype(np.float64)) - pb
    sgm = bf.soguk_mu.values.astype(np.float64)
    ww = np.where(sgm == 1, HEDEF_SOGUK / sgm.mean(), (1 - HEDEF_SOGUK) / (1 - sgm.mean()))
    ww = ww / ww.mean()
    m0b = float((ww * rb * rb).mean())

    def st(x):
        x = np.asarray(x, dtype=np.float64).copy()
        f = np.isfinite(x)
        if not f.any():
            return None
        x[~f] = np.median(x[f])
        x -= x.mean()
        s = np.sqrt(float((x * x).mean()))
        return x / s if s > 1e-12 else None

    svT, svB = st(a0), st(pb)
    sgT = tp.soguk_mu.values.astype(np.float64)
    ufT, ufB = st(tp.ufuk_gun.to_numpy()), st(bf.ufuk_gun.to_numpy())
    ayT = st(pd.to_datetime(tp.tarih).dt.month.to_numpy().astype(np.float64))
    ayB = st(pd.to_datetime(bf.tarih).dt.month.to_numpy().astype(np.float64))
    CARP = {"x_sv": (svT, svB), "x_soguk": (sgT, sgm), "x_ufuk": (ufT, ufB), "x_ay": (ayT, ayB)}
    ESIK = {"ust10": (0.9, True), "ust25": (0.75, True), "alt10": (0.1, False)}

    def kur(ad):
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

    with open(os.path.join(M29, "m121_derin_tarama.json")) as fh:
        TARAMA = json.load(fh)
    rng = np.random.default_rng(5)
    tn = bf.tanim.values
    uqn = pd.unique(tn)
    gi = pd.Series(np.arange(len(uqn)), index=uqn)[tn].to_numpy()
    PERM = [
        np.argsort(np.argsort(rng.permutation(len(uqn))[gi], kind="stable"), kind="stable")
        for _ in range(20)
    ]

    kul, ONCEKI, KAT, RCV, RS = [], [], [], [], []
    for kayit in TARAMA:
        if len(kul) >= AZAMI_EKSEN:
            break
        ad = kayit["eksen"]
        xt, xb = kur(ad)
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
        KAT.append(float(np.sign(rho_cv) * TAVAN * abs(rho_s)))
        RCV.append(float(rho_cv))
        RS.append(float(rho_s))
    U = np.array(ONCEKI)
    dik = float(np.abs(U @ U.T / N - np.eye(len(U))).max())
    return dict(
        eksen=kul,
        kats=KAT,
        rho_cv=RCV,
        rho_s=RS,
        taban_mse=float(M0 - 2 * kL + float((r_hat * r_hat).mean())),
        diklik=dik,
    )


if os.path.exists(ONBELLEK):
    with open(ONBELLEK) as fh:
        D = json.load(fh)
    print(f"onbellek okundu: {os.path.basename(ONBELLEK)}")
else:
    D = eksenleri_kur()
    with open(ONBELLEK, "w") as fh:
        json.dump(D, fh, indent=1)
    print(f"onbellek yazildi: {os.path.basename(ONBELLEK)}")

AD = D["eksen"]
KATS = np.array(D["kats"])  # isaret * 1.95 * |rho_s|
RHO_CV = np.array(D["rho_cv"])
RHO_S = np.array(D["rho_s"])
TABAN_MSE = float(D["taban_mse"])
n = len(AD)
RHO = float(np.sqrt((KATS**2).sum()))
print(f"{n} eksen, U diklik sapmasi {D['diklik']:.1e}, TABAN_MSE {TABAN_MSE:.8f}")
print(f"||1.95*rho_s|| = {RHO:.4f}   ||rho_cv|| = {np.linalg.norm(RHO_CV):.4f}")
print(f"hicbir yon olculmezse skor {np.sqrt(TABAN_MSE):.5f}")

# ---------------------------------------------------------------------------
# 2) ONSEL KOVARYANS
# ---------------------------------------------------------------------------
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
HAVA_MU = np.array([1.0 if any(h in a for h in HAVA) else 0.0 for a in AD])
ISR = np.sign(KATS)
SIRA_RS = np.argsort(-np.abs(RHO_S))
V_LB = ISR * np.abs(RHO_S)  # c'nin carptigi vektor (1.95 ICINDE DEGIL)
P_ISARET = 5.0 / 40.0  # m142: iki bagimsiz kaynak 35/40 uyusuyor
Q_IS = 1.0 - 2 * P_ISARET
C_ORT, C_SS = 0.7, 0.30  # m145: |c| ~ 0.7, %90 araligi [0.3, 1.3]

# OLCEK. rho_cv'yi OLDUGU GIBI onsel ortalama saymak SACMADIR: ||rho_cv|| =
# 0.69, bu da skor 0.90 demek -- liderin 0.99009'undan cok uzak, imkansiz.
# m148 zaten bu yuzden buyuklugu 1.95|rho_s| ile KESIYOR. Dogru kurulus:
# rho_cv'nin YONUNE guven, BUYUKLUGUNE guvenme. Butun dallari ayni toplam
# sinyal enerjisine normalize ederiz:
#     T = E||rho||^2 = E[c^2] * ||rho_s||^2
# Bu T, LB'ye capali TEK olcek tahminidir; alt uzay secimi zaten T'den
# BAGIMSIZDIR (T sadece skora cevirirken kullanilir).
EC2 = C_ORT**2 + C_SS**2
T_SIN = EC2 * float(V_LB @ V_LB)
W_CV = RHO_CV / np.linalg.norm(RHO_CV) * np.linalg.norm(V_LB)  # yon: rho_cv
V_HAVA = V_LB * HAVA_MU
V_YAPI = V_LB * (1.0 - HAVA_MU)
MU_OLCEK = C_ORT * Q_IS  # ortalamanin enerjiye orani (dallarda ortak)
# dal olasiliklari: A = LB agirligi, B = CV yonu, D = aile yapisi, C = hicbiri
PA, PB, PD = 0.45, 0.25, 0.15


def sigma_kur(pa=PA, pb=PB, pd_=PD, q=Q_IS, c_ort=C_ORT, c_ss=C_SS):
    """Sigma = E[rho rho'] ve mu = E[rho]. Butun dallar T_SIN enerjili."""
    pc = max(1.0 - pa - pb - pd_, 0.0)
    nv2 = float(V_LB @ V_LB)
    # A: tek kuresel c, eksen basina isaret hatasi -> diag(v^2) terimi
    SA = (c_ort**2 + c_ss**2) * (q * q * np.outer(V_LB, V_LB) + (1 - q * q) * np.diag(V_LB**2))
    # B: rho_cv YONU (buyuklugu T_SIN'e normalize)
    SB = T_SIN * np.outer(W_CV, W_CV) / float(W_CV @ W_CV)
    # D: aile basina bagimsiz carpan (m141: hava ailesi bloklar arasi doniyor)
    SD = T_SIN * (np.outer(V_HAVA, V_HAVA) + np.outer(V_YAPI, V_YAPI)) / nv2
    # C: modellenmemis, izotropik
    SC = T_SIN * np.eye(n) / n
    S = pa * SA + pb * SB + pd_ * SD + pc * SC
    mu = MU_OLCEK * (pa * V_LB + pb * W_CV + pd_ * V_LB)
    return S, mu


print(f"onsel toplam sinyal enerjisi T = E||rho||^2 = {T_SIN:.6f}")
print(
    f"  (T tamamen yakalanirsa skor {np.sqrt(TABAN_MSE - T_SIN):.5f} -> {sira(np.sqrt(TABAN_MSE - T_SIN))})"
)


# ---------------------------------------------------------------------------
# 3) ADAY ALT UZAYLAR. Hepsi R^40'ta agirlik vektorudur; U ortonormal oldugu
#    icin R^40'taki Gram-Schmidt, N boyuttaki yonlerin dikligiyle BIREBIR
#    ayni sonucu verir (m148 de zaten ayni yonleri boyle kuruyor).
# ---------------------------------------------------------------------------
def gs(vekler, kmax):
    cik = []
    for v in vekler:
        v = np.asarray(v, dtype=np.float64).copy()
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        v = v / nv
        for g in cik:
            v = v - (v @ g) * g
        nv = np.linalg.norm(v)
        if nv < 0.05:
            continue
        cik.append(v / nv)
        if len(cik) >= kmax:
            break
    return np.array(cik)


def tekil(j):
    m = np.zeros(n)
    m[j] = 1.0
    return m


def adaylar(kmax, Sig0):
    """kmax boyutlu aday alt uzaylar -> {ad: W (k x 40)}."""
    A = {}
    # (a) MEVCUT (m148): H1 1.95|rho_s|, H2 rho_cv, H3 hava, H4 yapisal, H5 esit
    hip = [np.abs(KATS), np.abs(RHO_CV), HAVA_MU, 1.0 - HAVA_MU, np.ones(n)]
    A["(a) mevcut H1..H4"] = gs([ISR * h for h in hip], kmax)
    # (b) rho_cv yonu + Sigma PCA
    EV = np.linalg.eigh(Sig0)[1][:, ::-1].T
    A["(b) rho_cv + PCA"] = gs([W_CV] + list(EV), kmax)
    # (c) Sigma ozvektorleri -- KY FAN OPTIMUMU (beklenen sinyal icin)
    A["(c) Sigma ozvektor"] = gs(list(EV), kmax)
    # (d) esit bolme: eksenleri kmax esit gruba ayir
    A["(d) esit bolme"] = gs([ISR * (np.arange(n) % kmax == j) for j in range(kmax)], kmax)
    # (e) en buyuk |rho_s| olan k eksen TEK BASINA
    A["(e) en buyuk k eksen"] = gs([tekil(j) for j in SIRA_RS[:kmax]], kmax)
    # (f) H1 + en buyuk |rho_s| tekil eksenler
    A["(f) H1 + tekil eksen"] = gs([V_LB] + [tekil(j) for j in SIRA_RS], kmax)
    # (g) rastgele k boyut (taban cizgi)
    rg = np.random.default_rng(11)
    A["(g) rastgele"] = gs(list(rg.normal(size=(kmax + 3, n))), kmax)
    # (h) YENI: aile yonleri |rho_s| AGIRLIKLI (m148 esit agirlik kullaniyor),
    #     sonra rho_cv, sonra tekil eksenler. span{v_H, v_Y} zaten H1'i icerir.
    A["(h) aile-agirlikli"] = gs([V_HAVA, V_YAPI, W_CV] + [tekil(j) for j in SIRA_RS], kmax)
    # (i) YENI: H1 + rho_cv + tekil eksenler (aile hipotezi atilir)
    A["(i) H1+cv+tekil"] = gs([V_LB, W_CV] + [tekil(j) for j in SIRA_RS], kmax)
    return A


def yakalanan(W, Sig):
    return float(np.trace(W @ Sig @ W.T))


# ---------------------------------------------------------------------------
# MONTE CARLO -- P(1. sira) ve P(2. sira). Beklenen sinyal TEK BASINA yetmez:
# 1. sira 0.02175, yani T'nin 2.24 KATI sinyal ister. Yuksek esikte VARYANS
# degerlidir; beklentiyi enbuyuten alt uzay P(1. sira)'yi enbuyutmeyebilir.
# ---------------------------------------------------------------------------
ESIK1 = TABAN_MSE - 0.99009**2  # 1. sira icin gereken toplam rho^2
ESIK2 = TABAN_MSE - 0.99614**2
ESIK3 = TABAN_MSE - 0.99927**2
print(f"gereken toplam rho^2:  1.sira {ESIK1:.5f}   2.sira {ESIK2:.5f}   3.sira {ESIK3:.5f}")

MC = 40000
rg = np.random.default_rng(2026)


def ornekle(pa=PA, pb=PB, pd_=PD, m=MC, c_ort=C_ORT, c_ss=C_SS, p_is=P_ISARET):
    """Onselden rho ornekleri (m x 40)."""
    pc = max(1.0 - pa - pb - pd_, 0.0)
    dal = rg.choice(4, size=m, p=[pa, pb, pd_, pc])
    R = np.zeros((m, n))
    q = 1.0 - 2 * p_is
    ec2 = c_ort**2 + c_ss**2
    g_ort, g_ss = c_ort * q, np.sqrt(max(ec2 - (c_ort * q) ** 2, 1e-12))
    iA = dal == 0
    if iA.any():
        c = rg.normal(c_ort, c_ss, size=(iA.sum(), 1))
        eps = np.where(rg.random((iA.sum(), n)) < p_is, -1.0, 1.0)
        R[iA] = c * eps * V_LB
    iB = dal == 1
    if iB.any():
        R[iB] = rg.normal(g_ort, g_ss, size=(iB.sum(), 1)) * W_CV
    iD = dal == 2
    if iD.any():
        a_ = rg.normal(g_ort, g_ss, size=(iD.sum(), 1))
        b_ = rg.normal(g_ort, g_ss, size=(iD.sum(), 1))
        R[iD] = a_ * V_HAVA + b_ * V_YAPI
    iC = dal == 3
    if iC.any():
        R[iC] = rg.normal(0.0, np.sqrt(T_SIN / n), size=(iC.sum(), n))
    return R


def olasilik(W, R):
    """P(1.sira), P(2.sira), E[yakalanan], ortanca skor."""
    y = ((R @ W.T) ** 2).sum(axis=1)
    sk = np.sqrt(np.maximum(TABAN_MSE - y, 1e-9))
    return (
        float((y >= ESIK1).mean()),
        float((y >= ESIK2).mean()),
        float(y.mean()),
        float(np.median(sk)),
    )


R_ORN = ornekle()
print(f"MC {MC} ornek. E||rho||^2 (tum 40 boyut) = {(R_ORN**2).sum(axis=1).mean():.5f}")
_yt = (R_ORN**2).sum(axis=1)
print(
    f"  TAM ORAKL (40 boyutun hepsi olculebilseydi): P(1.sira)={float((_yt >= ESIK1).mean()):.3f}"
    f"  P(2.sira)={float((_yt >= ESIK2).mean()):.3f}"
)

print("\n" + "=" * 92)
print("3) ALT UZAY KARSILASTIRMASI -- E[yakalanan sinyal] ve SIRA OLASILIKLARI")
print("=" * 92)
SG0 = sigma_kur()[0]
SONUC = {}
for k in (3, 4, 5, 6):
    print(f"\n--- k = {k} boyut ---")
    print(
        f"{'aday':>22s} {'E[yakalanan]':>13s} {'ortanca skor':>13s} "
        f"{'P(1.sira)':>10s} {'P(2.sira)':>10s}"
    )
    A = adaylar(k, SG0)
    satirlar = []
    for ad_, W in A.items():
        p1, p2, ey, os_ = olasilik(W, R_ORN)
        SONUC[(k, ad_)] = (p1, p2, ey, os_)
        satirlar.append((p1, p2, f"{ad_:>22s} {ey:13.5f} {os_:13.5f} {p1:10.3f} {p2:10.3f}"))
    for _, _, s in sorted(satirlar, reverse=True):
        print(s)

# ---------------------------------------------------------------------------
# 4) DUYARLILIK -- dal olasiliklari degisince siralama degisiyor mu?
# ---------------------------------------------------------------------------
print("\n" + "=" * 92)
print("4) DUYARLILIK (k=4): dal olasiliklari degisirse hangi aday kazanir?")
print("=" * 92)
SENARYO = [
    ("taban  A.45 B.25 D.15", 0.45, 0.25, 0.15),
    ("LB agir A.80 B.10 D.05", 0.80, 0.10, 0.05),
    ("CV agir A.20 B.55 D.10", 0.20, 0.55, 0.10),
    ("aile agir A.25 B.15 D.50", 0.25, 0.15, 0.50),
    ("kotumser A.30 B.15 D.10", 0.30, 0.15, 0.10),
    ("saf LB   A1.0", 1.00, 0.00, 0.00),
]
ADAY_LISTE = [
    "(a) mevcut H1..H4",
    "(c) Sigma ozvektor",
    "(h) aile-agirlikli",
    "(i) H1+cv+tekil",
    "(f) H1 + tekil eksen",
    "(e) en buyuk k eksen",
]
print(f"{'senaryo':>24s}" + "".join(f"{a[:18]:>19s}" for a in ADAY_LISTE))
for ad_s, pa, pb, pd_ in SENARYO:
    Rs = ornekle(pa=pa, pb=pb, pd_=pd_)
    A = adaylar(4, sigma_kur(pa=pa, pb=pb, pd_=pd_)[0])
    satir = f"{ad_s:>24s}"
    for a_ in ADAY_LISTE:
        p1, p2, ey, _ = olasilik(A[a_], Rs)
        satir += f"  {ey:.5f} {p1:.2f}/{p2:.2f}"
    print(satir)
print("(hucre: E[yakalanan]  P(1.sira)/P(2.sira))")

# ---------------------------------------------------------------------------
# 5) KAC YON OLCULMELI
# ---------------------------------------------------------------------------
print("\n" + "=" * 92)
print("5) KAC YON OLCULMELI -- her ek boyutun marjinal katkisi")
print("=" * 92)
print(
    f"{'aday':>22s} {'k':>2s} {'E[yakalanan]':>13s} {'marj.katki':>11s} "
    f"{'ortanca skor':>13s} {'P(1.sira)':>10s} {'P(2.sira)':>10s}"
)
for ad_ in ("(a) mevcut H1..H4", "(c) Sigma ozvektor", "(h) aile-agirlikli", "(i) H1+cv+tekil"):
    onc = 0.0
    for k in (1, 2, 3, 4, 5, 6):
        W = adaylar(k, SG0)[ad_]
        p1, p2, ey, os_ = olasilik(W, R_ORN)
        print(f"{ad_:>22s} {k:2d} {ey:13.5f} {ey - onc:11.5f} {os_:13.5f} {p1:10.3f} {p2:10.3f}")
        onc = ey
    print()

# ---------------------------------------------------------------------------
# 6) KAPPA SECIMI
#   Olcum:  rho_hat = rho + eps,  eps = YUV / kappa_etkin.
#   NIHAI dosyanin kaybi         = eps^2 = (YUV/kappa)^2     (kappa ile AZALIR)
#   SONDA dosyasinin kendi kaybi = kappa^2 - 2*kappa*mu_k    (kappa ile ARTAR)
#     -- ikincisi YALNIZCA o sonda elde kalan SON dosya olursa odenir (q_dur).
#   Beklenen ceza = (YUV/kappa)^2 + q_dur*(kappa^2 - 2*kappa*mu_k)
# ---------------------------------------------------------------------------
print("\n" + "=" * 92)
print("6) KAPPA SECIMI -- olcum hassasiyeti ile 'hak biterse elde kalan dosya' dengesi")
print("=" * 92)
SG, MU = sigma_kur()
KAPPA_SIMDI = (0.8 / 1.95) * RHO / np.sqrt(4)
print(f"su anki kappa (dort yonde de TEKDUZE) = {KAPPA_SIMDI:.5f}")
# OLCUM HATASI: docs/72 "5.6e-05" diyor, bu YALNIZCA LB yuvarlamasidir ve
# EKSIKTIR. rho_k = (sabit - P^2)/(2*kappa) ; sabit = M0 - 2*kL + Q(d) icindeki
# (M0 - 2*kL) OLCULMUS DEGIL KALIBRE edilmis bir sabittir. m112'nin kendi
# notu: 27 gercek yonde leave-one-out ort |hata| = 1.72e-04 (P^2 olceginde).
# Bu hata da 1/(2*kappa) ile buyur ve YUVARLAMADAN 30 KAT BUYUKTUR.
SIG_C = 1.72e-4  # m112_kalibre.py sat.52: LOO ort |hata| (P^2 olceginde)
SIG_P2 = 2.0 * YUV  # P^2'nin yuvarlama gurultusu (P ~ 1)


def olcum_hatasi(kappa):
    return np.sqrt(SIG_C**2 + SIG_P2**2) / (2 * kappa)


print(
    f"olcum hatasi bu kappa'da: yalnizca yuvarlama {SIG_P2 / (2 * KAPPA_SIMDI):.2e}, "
    f"kalibrasyon DAHIL {olcum_hatasi(KAPPA_SIMDI):.2e}  "
    f"(nihaiye maliyet {olcum_hatasi(KAPPA_SIMDI) ** 2:.2e})\n"
)
KS = np.geomspace(1e-4, 0.4, 40001)


def kappa_opt(mu, q_dur):
    ceza = olcum_hatasi(KS) ** 2 + q_dur * (KS**2 - 2 * KS * mu)
    j = int(np.argmin(ceza))
    return float(KS[j]), float(ceza[j])


for ad_ in ("(a) mevcut H1..H4", "(h) aile-agirlikli"):
    W = adaylar(4, SG0)[ad_]
    print(f"--- {ad_} ---")
    print(
        f"{'yon':>4s} {'mu_k=E[rho_k]':>14s} {'sd_k':>8s} {'kappa*':>9s} "
        f"{'simdiki ceza':>13s} {'kappa* cezasi':>14s} {'skor farki':>11s}"
    )
    for i in range(W.shape[0]):
        g = W[i]
        mu_k = float(g @ MU)
        if mu_k < 0:
            g, mu_k = -g, -mu_k
        sd = float(np.sqrt(max(g @ SG @ g - mu_k**2, 0.0)))
        kst, cst = kappa_opt(mu_k, 0.3)
        c_simdi = olcum_hatasi(KAPPA_SIMDI) ** 2 + 0.3 * (KAPPA_SIMDI**2 - 2 * KAPPA_SIMDI * mu_k)
        print(
            f"{i + 1:4d} {mu_k:14.5f} {sd:8.5f} {kst:9.5f} {c_simdi:13.6f} {cst:14.6f} "
            f"{(c_simdi - cst) / (2 * np.sqrt(TABAN_MSE)):11.5f}"
        )
    print()

print(
    f"{'kappa':>8s} {'olcum hatasi':>13s} {'nihaiye maliyet':>16s} "
    f"{'sonda kaybi (mu=0)':>19s} {'toplam (q=0.3)':>15s}"
)
for kp in (0.0517, 0.03, 0.02, 0.0125, 0.01, 0.005, 0.002):
    eps = olcum_hatasi(kp)
    print(f"{kp:8.4f} {eps:13.2e} {eps**2:16.2e} {kp**2:19.6f} {eps**2 + 0.3 * kp**2:15.6f}")
print("\nYORUM. Yuvarlama tek basina kappa'yi 25 kat kucultmeye izin verirdi ama")
print("KALIBRASYON hatasi (1.72e-4) da 1/kappa ile buyudugu icin izin vermez.")
print("Ikisi birlikte mu_k=0 olan HEDGE yonlerinde kappa* ~ 0.012 verir; su anki")
print("0.0517 ise sonda dosyasina kappa^2 = 0.00268'lik bir bedel yukler ve bu")
print("bedel 2. sira icin gereken 0.00973'un DORTTE BIRIDIR.")

# ---------------------------------------------------------------------------
# 6b) COZUM FORMULUNUN SINAVI -- m148'in rho_k cozumu 2. sondadan itibaren
#     DOGRU MU? Sentetik bir gercekle bire bir sinanir (cebir, veri degil).
#
#       P^2 = M0 - 2*L.d + Q(d),   d = r_hat + toplam_{j<k} rho_j g_j + kappa g_k
#       L.g_j = rho_j  (tanim),    L.r_hat = kL
#     =>  P^2 = [M0 - 2kL + Q(d)] - 2*toplam_{j<k} rho_j^2 - 2*kappa*rho_k
#              = sabit            - 2*S_onceki                - 2*kappa*rho_k
#     m148 ise  rho_k = (sabit - P^2) / (2*kappa)  kullaniyor; S_onceki terimi
#     DUSMUS. 1. sondada S_onceki = 0 oldugu icin D1 DOGRUDUR; 2. sondadan
#     itibaren rho_k, S_onceki/kappa kadar SISER.
# ---------------------------------------------------------------------------
print("\n" + "=" * 92)
print("6b) m148 COZUM FORMULU SINAVI (sentetik gercek, tam cebir)")
print("=" * 92)
rg2 = np.random.default_rng(7)
h_span, l_span = 0.0611, 0.0500  # r_hat ve L'nin span bilesenleri (temsili)
RHO_GER = np.array([0.0600, 0.0200, -0.0100, 0.0050])  # sentetik GERCEK rho_k
kap = KAPPA_SIMDI
M0s = l_span**2 + float((RHO_GER**2).sum()) + 0.9  # kalan (olculemez) artik
kLs = l_span * h_span
TABAN_s = M0s - 2 * kLs + h_span**2
print(
    f"{'sonda':>6s} {'gercek rho_k':>13s} {'m148 cozumu':>12s} {'sapma':>10s} {'duzeltilmis':>12s}"
)
for k in range(4):
    S_onc = float((RHO_GER[:k] ** 2).sum())
    Qd = h_span**2 + S_onc + kap**2
    P2 = M0s - 2 * (kLs + S_onc + kap * RHO_GER[k]) + Qd
    sabit = M0s - 2 * kLs + Qd
    m148 = (sabit - P2) / (2 * kap)
    duz = (sabit - 2 * S_onc - P2) / (2 * kap)
    print(f"{k + 1:6d} {RHO_GER[k]:13.5f} {m148:12.5f} {m148 - RHO_GER[k]:10.5f} {duz:12.5f}")
print("\nSONUC: D1 DOGRU. 2.-4. sondalarda m148'in cozumu S_onceki/kappa kadar")
print("sisiyor (yukarida ~0.070). Sisen rho nihai dosyaya uygulanirsa kazanc")
print("NEGATIFE doner. m148_demet_plani.py'de sabit_k satirina '- 2*S_onceki'")
print("eklenmelidir (ya da esdeger olarak COZUM formulune).")

# ---------------------------------------------------------------------------
# 7) OZET / ONERI
# ---------------------------------------------------------------------------
print("\n" + "=" * 92)
print("7) OZET")
print("=" * 92)
for k in (4, 5):
    p1a, p2a, eya, osa = SONUC[(k, "(a) mevcut H1..H4")]
    en = max(
        ((SONUC[(k, a_)], a_) for a_ in ADAY_LISTE + ["(b) rho_cv + PCA", "(h) aile-agirlikli"]),
        key=lambda t: t[0][2],
    )
    (p1b, p2b, eyb, osb), adb = en
    print(
        f"k={k}: mevcut(a) E={eya:.5f} P1={p1a:.3f} P2={p2a:.3f} ortanca {osa:.5f}   |   "
        f"en iyi {adb} E={eyb:.5f} P1={p1b:.3f} P2={p2b:.3f} ortanca {osb:.5f}   "
        f"kazanc {osa - osb:+.5f}"
    )
print("\nHICBIR GONDERIM YAPILMADI, submissions/ altina YAZILMADI.")
