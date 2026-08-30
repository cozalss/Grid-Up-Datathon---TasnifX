"""EKSEN KUMESININ GOZDEN GECIRILMESI -- olcut: DIK UZAYDA GERCEK SINYAL.

m148 bilesigi 40 eksenden kuruyor ve bu eksenleri 4 dik yone (H1..H4)
donusturuyor. Bu betik eksen KUMESINI sinar. Olcut rho_pred DEGIL (m144
onun sisirilebilir oldugunu gosterdi): olcut, yaz25 blogunda ZAMAN ve
TRAFO bolmeli, yonleri kurarken GORULMEYEN parcada olculen gercek
yakalama (kapsam R2).

GEOMETRI. Her eksenin span(V)'ye dik parcasi
    xp0_i = x_i - V c_i,   c_i = G^+ (V'x_i / N)
Bunlarin Gram matrisi ANALITIK olarak
    H = X X'/N - B' G^+ B,   B = V'X/N
ile bulunur; N-boyutlu vektorleri saklamaya gerek yoktur. Gram-Schmidt
katsayilari T (u_i = sum_j T_ij xp0_j) yalniz H'den cikar. Bir demet yonu
    v_k = sum_i a_ki u_i  ->  ham eksen katsayilari  (a_k T)
olur; blok tarafinda ayni katsayilarla xb'ler birlestirilir. Boylece
yonlerin kurulusu bloktan HIC BESLENMEZ (isaret ve H2 agirligi haric,
onlar da bolmeli olcumde yalniz KURMA parcasindan alinir).

BOLUMLER
  1  5 ters-isaretli eksenin (m142) demet yonlerine katkisi ve atilma etkisi
  2  m144'un A-G ailesinden 22 yeni eksen eklenirse yonler nasil degisir
  3  dik parcalarin TEKRARLILIGI (etkin boyut) ve 4 yonun kapsama payi
  4  eksen basina plasebo/kararlilik (trafo-bolmeli, zaman-bolmeli)
  5  NIHAI ONERI

HICBIR GONDERIM YAPILMAZ, submissions/ altina YAZILMAZ.
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
RHO_S_ALT, QS_ALT, QD_ALT = 0.015, 0.02, 0.25
AZAMI_EKSEN = 40
KODLAMA_K = 200.0
#: Buyuk ara sonuclar burada onbelleklenir (depo disi, gecici dizin).
ONB_DIZIN = os.environ.get("M155_ONBELLEK", os.path.join(os.environ.get("TEMP", "."), "m155"))
os.makedirs(ONB_DIZIN, exist_ok=True)
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
ONB_V = os.path.join(ONB_DIZIN, "m155_V.npz")
if os.path.exists(ONB_V):
    _z = np.load(ONB_V)
    V, L = _z["V"], _z["L"]
else:
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
    np.savez(ONB_V, V=V, L=L)
GV = (V.T @ V) / N
Gi = np.linalg.pinv(GV, rcond=1e-6)
GI5 = np.linalg.pinv(GV, rcond=1e-5)
r_hat, gercek, kL = buzmeli_r_hat(V, L, GV, N)
MSE_OPT = M0 - gercek
TABAN_MSE = float(M0 - 2 * kL + float((r_hat * r_hat).mean()))
print(f"V: {V.shape[1]} gonderim yonu, N={N:,}")
print(f"saf optimum {np.sqrt(MSE_OPT):.6f}   TABAN_MSE {TABAN_MSE:.7f}")

# ------------------------------------------------------------------ blok
tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
e = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
SIC_AILE = [(t, aa) for t in (1000, 1001, 1002) for aa in ("cat", "xgb", "lgbm")]


def blok_artik(eg, blok):
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

d_idx, d_r = [], []
for b in ("guz25", "kis26"):
    i2, r2 = blok_artik(e, b)
    d_idx.append(i2)
    d_r.append(r2)
DIS_R = np.concatenate(d_r)
DIS = e.loc[np.concatenate(d_idx)].copy()
print(f"blok yaz25 {NB:,} satir | dis blok (te kodlayici) {len(DIS_R):,} satir")
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
ufTh = tp.ufuk_gun.to_numpy().astype(np.float64)
ufBh = bf.ufuk_gun.to_numpy().astype(np.float64)
ufT, ufB = st(ufTh), st(ufBh)
tarT, tarB = pd.to_datetime(tp.tarih), pd.to_datetime(bf.tarih)
ayT = st(tarT.dt.month.to_numpy().astype(np.float64))
ayB = st(tarB.dt.month.to_numpy().astype(np.float64))
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


# -------------------------------------------------- plasebo permutasyonlari
rng = np.random.default_rng(5)
tn = bf.tanim.values
uqn = pd.unique(tn)
gi = pd.Series(np.arange(len(uqn)), index=uqn)[tn].to_numpy()
PERM = [
    np.argsort(np.argsort(rng.permutation(len(uqn))[gi], kind="stable"), kind="stable")
    for _ in range(20)
]

# =================================================== 1) MEVCUT 40 EKSEN
with open(os.path.join(M29, "m121_derin_tarama.json")) as fh:
    TARAMA = json.load(fh)

ADLAR, XT, XB, RHO_S, RHO_CV, ZPL, ONCEKI = [], [], [], [], [], [], []
for kayit in TARAMA:
    if len(ADLAR) >= AZAMI_EKSEN:
        break
    ad = kayit["eksen"]
    xt, xb = kur(ad)
    if xt is None or xb is None:
        continue
    cc = Gi @ ((V.T @ xt) / N)
    xp0 = xt - V @ cc
    Qs = 1.0 - float((xp0 * xp0).mean())
    if Qs < QS_ALT:
        continue
    rho_s = float((r_hat * xt).mean()) / np.sqrt(Qs)
    if abs(rho_s) < RHO_S_ALT:
        continue
    cc5 = GI5 @ ((V.T @ xt) / N)
    xp5 = xt - V @ cc5
    Qs5 = 1.0 - float((xp5 * xp5).mean())
    if Qs5 < QS_ALT:
        continue
    if abs(float((r_hat * xt).mean()) / np.sqrt(Qs5) - rho_s) > 0.3 * abs(rho_s):
        continue
    xp = xp0.copy()
    for u in ONCEKI:
        xp -= float((xp * u).mean()) * u
    Qd = float((xp * xp).mean())
    if Qd < QD_ALT:
        continue
    kor = float((ww * rb * xb).mean()) / np.sqrt(m0b)
    gur = float(np.std([float((ww * rb[s] * xb).mean()) / np.sqrt(m0b) for s in PERM]))
    if abs(kor) < 3 * gur:
        continue
    rho_cv = CARPAN * kor
    if abs(rho_cv) < TAVAN * abs(rho_s):
        continue
    ONCEKI.append(xp / np.sqrt(Qd))
    ADLAR.append(ad)
    XT.append(xt)
    XB.append(xb)
    RHO_S.append(rho_s)
    RHO_CV.append(rho_cv)
    ZPL.append(abs(kor) / max(gur, 1e-12))
N40 = len(ADLAR)
DIK_BIRIM = ONCEKI
print(f"\nMEVCUT eksen kumesi yeniden kuruldu: {N40} eksen")
assert N40 == 40, "m148'in 40 ekseni yeniden uretilemedi"

# ============================================ 2) M144'UN A-G YENI EKSENLERI
# m144'un secilen_h_haric listesindeki 22 eksen. Uretecleri m144'ten aynen
# alindi (yalniz gereken adlar icin). H ailesi (mevcut 40'in carpimlari)
# BILINCLI olarak DISARIDA: ayni sinyalin dogrusal olmayan tekrari.
with open(os.path.join(M29, "m144_yeni_aileler.json")) as fh:
    M144 = json.load(fh)
YENI_KAYIT = M144["secilen_h_haric"]


def kod(sutunlar):
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


def hedef_kodla(sutunlar, k=KODLAMA_K):
    kd, kt, kb = kod(sutunlar)
    df = pd.DataFrame({"g": kd, "r": DIS_R})
    ag = df.groupby("g")["r"]
    v = ag.sum() / (ag.count() + k)
    m = pd.Series(v.to_numpy(dtype=np.float64), index=v.index)
    return (
        pd.Series(kt).map(m).fillna(0.0).to_numpy(dtype=np.float64),
        pd.Series(kb).map(m).fillna(0.0).to_numpy(dtype=np.float64),
    )


HV_KOL = ["nem_ort", "hissedilen_max", "yagis_toplam"]


def hava_paneli():
    pan = pd.concat(
        [tp[["lokasyon", "tarih"] + HV_KOL], bf[["lokasyon", "tarih"] + HV_KOL]],
        ignore_index=True,
    )
    pan = pan.drop_duplicates(["lokasyon", "tarih"]).sort_values(["lokasyon", "tarih"])
    g = pan.groupby("lokasyon", observed=True)
    yeni = {}
    for c in HV_KOL:
        s1 = g[c].shift(1)
        yeni[f"{c}_g1"] = s1
        yeni[f"{c}_g2"] = g[c].shift(2)
        o7 = s1.groupby(pan.lokasyon, observed=True).rolling(7, min_periods=4).mean()
        yeni[f"{c}_go7"] = o7.reset_index(level=0, drop=True).reindex(pan.index)
    return pd.concat([pan[["lokasyon", "tarih"]], pd.DataFrame(yeni, index=pan.index)], axis=1)


PAN = hava_paneli()
HVT = tp[["lokasyon", "tarih"]].merge(PAN, on=["lokasyon", "tarih"], how="left")
HVB = bf[["lokasyon", "tarih"]].merge(PAN, on=["lokasyon", "tarih"], how="left")
del PAN
gc.collect()
TE_GRUP = {"te_il": ["il_key"], "te_on2": ["tanim_on2"], "te_bolge_soguk": ["bolge", "soguk_mu"]}
UF_LOG_T, UF_LOG_B = st(np.log1p(ufTh)), st(np.log1p(ufBh))


def kur_yeni(ad):
    """m144'un A-G ureteclerinden yalniz secilen 22 eksen icin."""
    if ad.startswith("hv_"):
        c = ad[3:]
        if c not in HVT.columns:
            return None, None
        return (
            st(HVT[c].to_numpy(dtype=np.float64, na_value=np.nan)),
            st(HVB[c].to_numpy(dtype=np.float64, na_value=np.nan)),
        )
    if ad.startswith("uf_mentese"):
        d = float(ad[len("uf_mentese") :])
        return st(np.maximum(ufTh - d, 0)), st(np.maximum(ufBh - d, 0))
    if ad.startswith("yil_cos") or ad.startswith("yil_sin"):
        gov, kip = (ad.split(":", 1) + [""])[:2]
        h = int(gov[-1])
        f_ = np.cos if "cos" in gov else np.sin
        at = f_(2 * np.pi * h * tarT.dt.dayofyear.to_numpy().astype(np.float64) / 365.25)
        ab = f_(2 * np.pi * h * tarB.dt.dayofyear.to_numpy().astype(np.float64) / 365.25)
        if kip == "x_sv":
            return st(st(at) * svT), st(st(ab) * svB)
        return st(at), st(ab)
    if ":" in ad:
        kol, kip = ad.split(":", 1)
        if kol in TE_GRUP:
            xt, xb = hedef_kodla(TE_GRUP[kol])
            if kip in CARP:
                mt, mb = CARP[kip]
                return st(st(xt) * mt), st(st(xb) * mb)
            return st(xt), st(xb)
        if kol not in tp.columns or kol not in bf.columns:
            return None, None
        xt = tp[kol].to_numpy(dtype=np.float64, na_value=np.nan)
        xb = bf[kol].to_numpy(dtype=np.float64, na_value=np.nan)
        if kip.startswith("mnt"):
            q = int(kip[3:]) / 100.0
            fv = xt[np.isfinite(xt)]
            v_ = float(np.quantile(fv, q))
            return st(np.maximum(xt - v_, 0)), st(np.maximum(xb - v_, 0))
        if kip == "uf_log":
            a_, b_ = st(xt), st(xb)
            return st(a_ * UF_LOG_T), st(b_ * UF_LOG_B)
    return kur(ad)


print("\nA-G YENI EKSENLER (m144, H haric 22 aday) -- kapilar yeniden sinaniyor")
print(f"{'eksen':>34s} {'aile':>16s} {'rho_cv':>8s} {'rho_s':>8s} {'Q_dik':>6s} {'z':>6s}  durum")
YENI_ADLAR = []
for kayit in YENI_KAYIT:
    ad = kayit["eksen"]
    xt, xb = kur_yeni(ad)
    if xt is None or xb is None:
        print(f"{ad[:34]:>34s} {kayit['aile'][:16]:>16s} {'':>8s} {'':>8s} -- KURULAMADI")
        continue
    cc = Gi @ ((V.T @ xt) / N)
    xp0 = xt - V @ cc
    Qs = 1.0 - float((xp0 * xp0).mean())
    rho_s = float((r_hat * xt).mean()) / np.sqrt(Qs) if Qs > QS_ALT else 0.0
    kor = float((ww * rb * xb).mean()) / np.sqrt(m0b)
    gur = float(np.std([float((ww * rb[s] * xb).mean()) / np.sqrt(m0b) for s in PERM]))
    rho_cv = CARPAN * kor
    xp = xp0.copy()
    for u in DIK_BIRIM:
        xp -= float((xp * u).mean()) * u
    Qd = float((xp * xp).mean())
    tamam = (
        Qs > QS_ALT
        and abs(rho_s) >= RHO_S_ALT
        and Qd >= QD_ALT
        and abs(kor) >= 3 * gur
        and abs(rho_cv) >= TAVAN * abs(rho_s)
    )
    print(
        f"{ad[:34]:>34s} {kayit['aile'][:16]:>16s} {rho_cv:+8.4f} {rho_s:+8.4f} "
        f"{Qd:6.3f} {abs(kor) / max(gur, 1e-12):6.1f}  {'GECTI' if tamam else 'ELENDI'}"
    )
    if not tamam:
        continue
    DIK_BIRIM.append(xp / np.sqrt(Qd))
    ADLAR.append(ad)
    XT.append(xt)
    XB.append(xb)
    RHO_S.append(rho_s)
    RHO_CV.append(rho_cv)
    ZPL.append(abs(kor) / max(gur, 1e-12))
    YENI_ADLAR.append(ad)
NTUM = len(ADLAR)
print(f"\n{NTUM - N40} yeni eksen dogrulandi (m144: {len(YENI_KAYIT)})   toplam {NTUM}")
del DIK_BIRIM
gc.collect()

# =================================================== 3) GRAM MATRISI (H)
XTm = np.array(XT)
del XT
gc.collect()
Xtt = (XTm @ XTm.T) / N
Bm = (V.T @ XTm.T) / N
H = Xtt - Bm.T @ (Gi @ Bm)
H = (H + H.T) / 2
LR_HAT = (XTm @ r_hat) / N  # <r_hat, x_i>/N  (dik parcaya dik oldugu icin span-ici)
del XTm
gc.collect()
XBm = np.array(XB)
del XB
gc.collect()
RHO_S = np.array(RHO_S)
RHO_CV = np.array(RHO_CV)
ZPL = np.array(ZPL)
AILE_HAVA = np.array(
    [
        1.0
        if any(
            h in a
            for h in (
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
        )
        else 0.0
        for a in ADLAR
    ]
)
print(f"H kuruldu: {H.shape}, kosegen [{np.diag(H).min():.3f}, {np.diag(H).max():.3f}]")

# ================================================ 4) ARACLAR
IX40 = list(range(N40))
IXTUM = list(range(NTUM))
gun = (tarB - tarB.min()).dt.days.to_numpy()
GRUP = gi  # trafo grup kodu (satir basina)
BOL_TRAFO = (GRUP % 2).astype(int)
BOL_ZAMAN = (gun > np.median(gun)).astype(int)
FOLDLAR = [
    ("trafo A->B", BOL_TRAFO == 0, BOL_TRAFO == 1),
    ("trafo B->A", BOL_TRAFO == 1, BOL_TRAFO == 0),
    ("zaman erken->gec", BOL_ZAMAN == 0, BOL_ZAMAN == 1),
    ("zaman gec->erken", BOL_ZAMAN == 1, BOL_ZAMAN == 0),
]
NULL_RNG = np.random.default_rng(11)


def perm_maske(mask, n=12):
    """Maskedeki satirlari TRAFO GRUBU butun kalacak sekilde karistirir."""
    g = GRUP[mask]
    u = np.unique(g)
    yer = np.searchsorted(u, g)
    out = []
    for _ in range(n):
        anahtar = NULL_RNG.permutation(len(u))[yer]
        out.append(np.argsort(np.argsort(anahtar, kind="stable"), kind="stable"))
    return out


def kor_blok(xb, mask=None):
    """Agirlikli korelasyon: <r, x> / sqrt(<r,r>) (x zaten birim varyansa yakin)."""
    if mask is None:
        w, y, x = ww, rb, xb
    else:
        w, y, x = ww[mask], rb[mask], xb[mask]
    w = w / w.mean()
    x = x - float((w * x).mean())
    sx = np.sqrt(float((w * x * x).mean()))
    return float((w * y * x).mean()) / (np.sqrt(float((w * y * y).mean())) * max(sx, 1e-12))


def gs(ix, Hm=None):
    """Sirali Gram-Schmidt; Q_dik kapisini yeniden uygular."""
    Hs = (H if Hm is None else Hm)[np.ix_(ix, ix)]
    T, tut = [], []
    for i in range(len(ix)):
        t = np.zeros(len(ix))
        t[i] = 1.0
        for tj in T:
            t = t - float(Hs[i] @ tj) * tj
        q = float(t @ Hs @ t)
        if q < QD_ALT:
            continue
        T.append(t / np.sqrt(q))
        tut.append(i)
    return np.array(T), [ix[i] for i in tut], np.array(tut, dtype=int)


def demet(ix, rcv_kaynak=None, ekstra=None, ctx=None):
    """m148'in demet yonlerini kurar. rcv_kaynak: isaret/H2 icin rho_cv dizisi.

    ctx verilirse (H, rs, rcv, hava, XB) genel kume yerine o baglam kullanilir;
    boylece bolme-ici YENIDEN SECILMIS kumeler de ayni kodla degerlendirilir.
    """
    c_H, c_rs, c_rcv, c_hava, c_XB = (
        (H, RHO_S, RHO_CV, AILE_HAVA, XBm)
        if ctx is None
        else (ctx["H"], ctx["rs"], ctx["rcv"], ctx["hava"], ctx["XB"])
    )
    T, ix2, tut = gs(ix, c_H)
    rcv = (c_rcv if rcv_kaynak is None else np.asarray(rcv_kaynak))[ix2]
    kats = np.sign(rcv) * TAVAN * np.abs(c_rs[ix2])
    hava = c_hava[ix2]
    isr = np.sign(kats)
    HIP = [
        ("H1 1.95|rho_s|", np.abs(kats), True),
        ("H2 rho_cv", np.abs(rcv), True),
        ("H3 hava/mevsim", hava, True),
        ("H4 trafo/yapisal", 1.0 - hava, True),
        ("H5 esit", np.ones(len(ix2)), True),
    ]
    if ekstra:
        HIP = list(ekstra(ix2, T)) + HIP
    A, et = [], []
    for ad, w, imzali in HIP:
        a = (isr * np.asarray(w, dtype=np.float64)) if imzali else np.asarray(w, dtype=np.float64)
        n0 = float(np.linalg.norm(a))
        if n0 < 1e-12:
            continue
        a = a / n0
        for g in A:
            a = a - float(a @ g) * g
        n1 = float(np.linalg.norm(a))
        if n1 < 0.05:
            continue
        A.append(a / n1)
        et.append(ad)
    A = np.array(A)
    C = A @ T  # ham xp0 tabaninda katsayilar
    Ctam = np.zeros((len(A), len(c_rs)))
    Ctam[:, ix2] = C
    D = C @ c_XB[ix2]  # blok tarafinda yon vektorleri (K x NB)
    return dict(A=A, C=C, Ctam=Ctam, T=T, ix=ix2, et=et, kats=kats, D=D, XB=c_XB)


def kapsam(D, mask, y_blok=None):
    """Eval parcasinda K boyutlu alt uzayin yakaladigi agirlikli R2."""
    y, w = (rb if y_blok is None else y_blok)[mask], ww[mask]
    w = w / w.mean()
    X = D[:, mask].T
    X = X - (w @ X) / len(y)
    Aq = (X.T * w) @ X / len(y)
    bq = (X.T * w) @ y / len(y)
    beta = np.linalg.pinv(Aq, rcond=1e-10) @ bq
    return float(beta @ bq / float((w * y * y).mean()))


def kapsam_null(D, mask, n=12):
    """Ayni olcum, artik trafo-butun permutasyonla karistirilmis."""
    y0, w = rb[mask], ww[mask]
    w = w / w.mean()
    X = D[:, mask].T
    X = X - (w @ X) / len(y0)
    Aq = (X.T * w) @ X / len(y0)
    Ap = np.linalg.pinv(Aq, rcond=1e-10)
    out = []
    for p in perm_maske(mask, n):
        y = y0[p]
        bq = (X.T * w) @ y / len(y)
        out.append(float(bq @ Ap @ bq / float((w * y * y).mean())))
    return float(np.mean(out)), float(np.std(out))


def rho2_ve_skor(r2):
    """Blok R2 -> test rho^2 tahmini (CARPAN olcegi) -> nihai skor."""
    rho2 = (CARPAN**2) * max(r2, 0.0)
    return rho2, float(np.sqrt(max(TABAN_MSE - rho2, 1e-9)))


def sira_adi(sk):
    for h, ad in [
        (0.99009, "1. SIRA"),
        (0.99614, "2. SIRA"),
        (0.99927, "3. sira"),
        (1.00115, "4-6"),
    ]:
        if sk < h:
            return ad
    return "7.+"


GEREKEN = {
    "1. sira (0.99009)": TABAN_MSE - 0.99009**2,
    "2. sira (0.99614)": TABAN_MSE - 0.99614**2,
}
print("\nGEREKEN SINYAL (TABAN_MSE = %.7f)" % TABAN_MSE)
for ad, v in GEREKEN.items():
    print(
        f"  {ad}: toplam rho^2 = {v:.5f}  ->  blok R2 = {v / CARPAN**2:.5f}  "
        f"(kor = {np.sqrt(v / CARPAN**2):.4f})"
    )


# ------------------------------------------------- KUME DEGERLENDIRME
def rho_cv_parca(mask):
    return np.array([CARPAN * kor_blok(XBm[i], mask) for i in range(NTUM)])


RCV_PARCA = {}
for ad, kur_m, _ev in FOLDLAR:
    anahtar = ad.split("->")[0]
    if anahtar not in RCV_PARCA:
        RCV_PARCA[anahtar] = rho_cv_parca(kur_m)


KUR_MASKE = np.ones(NB, dtype=bool)
HB_XB, HB_Y = None, None  # hb_yonu'nun baglami (None -> genel kume)


def hb_yonu(ix2, T):
    """EKSTRA HIPOTEZ: yonu blogun KENDISI soylesin (yalniz KURMA parcasindan).

    u yonlerinin blok karsiliklarina agirlikli en kucuk kareler; katsayi vektoru
    dogrudan bir yon verir. Hipotez tahmin etmiyoruz, blok neyi goruyorsa onu
    olcmeye gidiyoruz.
    """
    Ub = T @ (XBm if HB_XB is None else HB_XB)[ix2]
    m_ = KUR_MASKE
    y, w = (rb if HB_Y is None else HB_Y)[m_], ww[m_]
    w = w / w.mean()
    X = Ub[:, m_].T
    X = X - (w @ X) / len(y)
    Aq = (X.T * w) @ X / len(y)
    bq = (X.T * w) @ y / len(y)
    beta = np.linalg.pinv(Aq, rcond=1e-8) @ bq
    n_ = float(np.linalg.norm(beta))
    return [("HB blok-en-iyi", beta / n_, False)] if n_ > 1e-12 else []


def degerlendir(ix, baslik, ayrinti=False, ekstra=None):
    """Bir eksen kumesini bolmeli olarak degerlendirir; ortalama kapsami doner."""
    global KUR_MASKE
    KUR_MASKE = np.ones(NB, dtype=bool)
    d_tum = demet(ix, ekstra=ekstra)
    K = len(d_tum["A"])
    satir = []
    for ad, kur_m, ev_m in FOLDLAR:
        rcv = RCV_PARCA[ad.split("->")[0]]
        KUR_MASKE = kur_m
        d = demet(ix, rcv_kaynak=rcv, ekstra=ekstra)
        r2 = kapsam(d["D"], ev_m)
        n0, ns = kapsam_null(d["D"], ev_m)
        # ayni kumenin TUM dik uzayi (ust sinir)
        Ttam = d["T"] @ XBm[d["ix"]]
        r2t = kapsam(Ttam, ev_m)
        satir.append((ad, r2, n0, ns, r2t, len(d["A"])))
    r2ort = float(np.mean([s[1] for s in satir]))
    n0ort = float(np.mean([s[2] for s in satir]))
    r2tort = float(np.mean([s[4] for s in satir]))
    if ayrinti:
        print(f"\n  {baslik}  ({len(d_tum['ix'])} eksen, {K} yon)")
        print(
            f"  {'bolme':>18s} {'K-yon R2':>9s} {'null':>8s} {'z':>6s} "
            f"{'tum-dik R2':>11s} {'K/tum':>6s}"
        )
        for ad, r2, n0, ns, r2t, kk in satir:
            print(
                f"  {ad:>18s} {r2:9.5f} {n0:8.5f} {(r2 - n0) / max(ns, 1e-12):6.1f} "
                f"{r2t:11.5f} {r2 / max(r2t, 1e-12):6.2f}"
            )
    rho2, sk = rho2_ve_skor(max(r2ort - n0ort, 0.0))
    zam = float(np.mean([s[1] - s[2] for s in satir[2:]]))
    return dict(
        baslik=baslik,
        n=len(d_tum["ix"]),
        K=K,
        r2=r2ort,
        null=n0ort,
        net_zaman=zam,
        r2_tum=r2tort,
        rho2=rho2,
        skor=sk,
        demet=d_tum,
    )


# ==================================================================
# BOLUM 1 -- 5 TERS-ISARETLI EKSEN
# ==================================================================
print("\n" + "=" * 78)
print("BOLUM 1 -- 5 TERS-ISARETLI EKSEN (m142)")
print("=" * 78)
TERS = [i for i in range(N40) if np.sign(RHO_CV[i]) != np.sign(RHO_S[i])]
print("CV isareti ile LB isareti celisen eksenler:")
for i in TERS:
    print(
        f"  [{i:2d}] {ADLAR[i]:<34s} rho_cv={RHO_CV[i]:+.4f}  rho_s={RHO_S[i]:+.4f}  z={ZPL[i]:5.1f}"
    )

D40 = demet(IX40)
print(f"\nmevcut demet: {len(D40['A'])} yon -> {D40['et']}")
print("\n5 eksenin demet yonlerine KATKISI (|<G_k, u_i>|, yonun enerjisinin payi %)")
sira_ix = {a: j for j, a in enumerate(D40["ix"])}
print(f"{'eksen':>34s} " + " ".join(f"{e.split()[0]:>8s}" for e in D40["et"]) + f" {'toplam%':>8s}")
for i in TERS:
    j = sira_ix.get(i)
    if j is None:
        continue
    pay = D40["A"][:, j] ** 2
    print(
        f"{ADLAR[i][:34]:>34s} "
        + " ".join(f"{100 * p:8.2f}" for p in pay)
        + f" {100 * float(pay.mean()):8.2f}"
    )
tp5 = sum(float((D40["A"][:, sira_ix[i]] ** 2).sum()) for i in TERS if i in sira_ix)
print(f"5 eksenin 4 yonun TOPLAM enerjisindeki payi: {100 * tp5 / len(D40['A']):.2f}%")

IX35 = [i for i in IX40 if i not in TERS]
D35 = demet(IX35)
M = D40["Ctam"] @ H @ D35["Ctam"].T
sv = np.linalg.svd(M, compute_uv=False)
print(f"\n40-yonlu ve 35-yonlu alt uzaylar arasi ana aci kosinusleri: {np.round(sv, 4)}")
print(f"  en kucuk kosinus {sv.min():.4f} -> alt uzaylar {'AYNI' if sv.min() > 0.95 else 'FARKLI'}")

R40 = degerlendir(IX40, "S1  mevcut 40 eksen", ayrinti=True)
R35 = degerlendir(IX35, "S2  40 - 5 ters isaretli (35)", ayrinti=True)

# ==================================================================
# BOLUM 2 -- M144'UN A-G AILELERI
# ==================================================================
print("\n" + "=" * 78)
print("BOLUM 2 -- A-G AILELERINDEN 22 YENI EKSEN")
print("=" * 78)
RTUM = degerlendir(IXTUM, f"S3  40 + {NTUM - N40} yeni ({NTUM})", ayrinti=True)
IX57 = [i for i in IXTUM if i not in TERS]
R57 = degerlendir(IX57, f"S4  35 + {NTUM - N40} yeni ({len(IX57)})", ayrinti=True)
D62 = RTUM["demet"]
M2 = D40["Ctam"] @ H @ D62["Ctam"].T
sv2 = np.linalg.svd(M2, compute_uv=False)
print(f"\n40-yon vs 62-yon ana aci kosinusleri: {np.round(sv2, 4)}")
print("\nHIPOTEZ YONLERININ ongorulen rho_k'lari (m148 olcegi):")
print(f"{'yon':>18s} {'40 eksen':>10s} {'62 eksen':>10s}")
for k, e2 in enumerate(D40["et"]):
    b40 = abs(float(D40["kats"] @ D40["A"][k]))
    b62 = abs(float(D62["kats"] @ D62["A"][k])) if k < len(D62["A"]) else float("nan")
    print(f"{e2:>18s} {b40:10.4f} {b62:10.4f}")

# ==================================================================
# BOLUM 3 -- DIK PARCALARIN TEKRARLILIGI
# ==================================================================
print("\n" + "=" * 78)
print("BOLUM 3 -- DIK PARCALAR NE KADAR BAGIMSIZ?")
print("=" * 78)


def etkin_boyut(Kmat):
    d = np.sqrt(np.clip(np.diag(Kmat), 1e-300, None))
    C_ = Kmat / np.outer(d, d)
    lam = np.clip(np.linalg.eigvalsh(C_)[::-1], 0, None)
    pay = lam / lam.sum()
    pr = float(lam.sum() ** 2 / (lam**2).sum())
    ent = float(np.exp(-(pay[pay > 0] * np.log(pay[pay > 0])).sum()))
    k90 = int(np.searchsorted(np.cumsum(pay), 0.90) + 1)
    return lam, pr, ent, k90


for ad, ix in [("40 mevcut", IX40), (f"{NTUM} (40+yeni)", IXTUM)]:
    lam, pr, ent, k90 = etkin_boyut(H[np.ix_(ix, ix)])
    print(f"\n{ad}: xp0 (span'a dik ham parca) korelasyon matrisi ozdegerleri")
    print("  ilk 10: " + " ".join(f"{v:.2f}" for v in lam[:10]))
    print(f"  katilim orani (PR) = {pr:.1f}   entropi boyutu = {ent:.1f}   %90 icin {k90} boyut")
    T_, ix2_, _ = gs(ix)
    Ub = T_ @ XBm[ix2_]
    w = ww / ww.mean()
    Kb = (Ub * w) @ Ub.T / NB
    lam2, pr2, ent2, k902 = etkin_boyut(Kb)
    print("  ayni yonlerin BLOK tarafindaki korelasyon ozdegerleri (ilk 10):")
    print("  " + " ".join(f"{v:.2f}" for v in lam2[:10]))
    print(f"  PR = {pr2:.1f}   entropi = {ent2:.1f}   %90 icin {k902} boyut")


def rastgele_kapsam(ix, K, tekrar=12, tohum=3):
    """Ayni dik uzaydan RASTGELE K boyutlu alt uzaylarin kapsami (taban cizgi)."""
    T_, ix2_, _ = gs(ix)
    Ub = T_ @ XBm[ix2_]
    rr = np.random.default_rng(tohum)
    out = []
    for _ in range(tekrar):
        A_ = np.linalg.qr(rr.standard_normal((len(ix2_), K)))[0].T
        Dr = A_ @ Ub
        out.append(float(np.mean([kapsam(Dr, ev) for _a, _k, ev in FOLDLAR])))
    return float(np.mean(out)), float(np.std(out))


for ad, ix in [("40 mevcut", IX40), (f"{NTUM} (40+yeni)", IXTUM)]:
    m_ = len(gs(ix)[1])
    rk, rs = rastgele_kapsam(ix, 4)
    tum = float(np.mean([kapsam(gs(ix)[0] @ XBm[gs(ix)[1]], ev) for _a, _k, ev in FOLDLAR]))
    print(
        f"\n{ad}: rastgele 4 boyut kapsami {rk:.5f} +- {rs:.5f}   "
        f"tum {m_} boyut {tum:.5f}   oran {rk / max(tum, 1e-12):.2f}"
    )
    print(f"  (4/{m_} = {4 / m_:.3f}; rastgele oran bundan BUYUKSE dik parcalar yiginlasmis)")

# ---- PROB BUTCESI: 4 yon yeterli mi?
# Alt uzay ne kadar buyukse toplam(rho_k^2) o kadar buyur; ama her yon bir
# GONDERIM HAKKI demek. Burada "kurma parcasinda en guclu K yonu olc"
# stratejisinin gorulmeyen parcadaki kapsami olculur.
print("\nPROB BUTCESI -- kurma parcasinda en guclu K dik yonu olcmek")
print(
    f"{'kume':>14s} {'K':>4s} {'net R2':>9s} {'zaman':>9s} {'rho^2':>8s} {'skor':>8s} {'sira':>8s}"
)
for ad, ix in [("40 mevcut", IX40), (f"{NTUM} (40+yeni)", IXTUM)]:
    T_, ix2_, _ = gs(ix)
    Ub = T_ @ XBm[ix2_]
    for K in (1, 2, 4, 6, 8, 12, 20, len(ix2_)):
        vals, nuls = [], []
        for _a, kur_m, ev_m in FOLDLAR:
            w = ww[kur_m] / ww[kur_m].mean()
            c = np.abs((Ub[:, kur_m] * w) @ rb[kur_m] / int(kur_m.sum()))
            top = np.argsort(-c)[:K]
            vals.append(kapsam(Ub[top], ev_m))
            nuls.append(kapsam_null(Ub[top], ev_m, 6)[0])
        net = float(np.mean(vals) - np.mean(nuls))
        zam = float(np.mean(vals[2:]) - np.mean(nuls[2:]))
        rho2, sk = rho2_ve_skor(net)
        _r2z, skz = rho2_ve_skor(max(zam, 0.0))
        print(
            f"{ad:>14s} {K:4d} {net:9.5f} {zam:9.5f} {rho2:8.5f} {sk:8.5f} "
            f"{sira_adi(sk):>8s} (zaman {skz:.5f} {sira_adi(skz)})"
        )

# ==================================================================
# BOLUM 4 -- EKSEN BASINA PLASEBO VE KARARLILIK
# ==================================================================
print("\n" + "=" * 78)
print("BOLUM 4 -- EKSEN BASINA KARARLILIK (trafo-bolmeli, zaman-bolmeli)")
print("=" * 78)
_qg = np.quantile(gun, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
_qg[-1] += 1
ZAMAN_P = [(gun >= _qg[k]) & (gun < _qg[k + 1]) for k in range(5)]
TRAFO_P = [k == (GRUP % 4) for k in range(4)]
KOR_Z = np.array([[kor_blok(XBm[i], m_) for m_ in ZAMAN_P] for i in range(NTUM)])
KOR_T = np.array([[kor_blok(XBm[i], m_) for m_ in TRAFO_P] for i in range(NTUM)])
KOR_ALL = np.array([kor_blok(XBm[i]) for i in range(NTUM)])
sz = KOR_Z.std(axis=1)
stf = KOR_T.std(axis=1)
isz = (np.sign(KOR_Z) != np.sign(KOR_ALL)[:, None]).sum(axis=1)
ist = (np.sign(KOR_T) != np.sign(KOR_ALL)[:, None]).sum(axis=1)
KARARSIZ = isz + ist + (sz + stf) / np.maximum(np.abs(KOR_ALL), 1e-9)
sirali = np.argsort(-KARARSIZ)
print(
    f"{'eksen':>34s} {'kor':>7s} {'sd_zmn':>7s} {'sd_trf':>7s} {'don_z':>5s} {'don_t':>5s} {'skor':>6s}"
)
for i in sirali[:12]:
    ek = " (yeni)" if i >= N40 else (" [TERS]" if i in TERS else "")
    print(
        f"{(ADLAR[i][:27] + ek)[:34]:>34s} {KOR_ALL[i]:+7.4f} {sz[i]:7.4f} {stf[i]:7.4f} "
        f"{isz[i]:5d} {ist[i]:5d} {KARARSIZ[i]:6.2f}"
    )
KARARSIZ5 = [int(i) for i in sirali if i < N40][:5]
print(f"\nEN KARARSIZ 5 (mevcut 40 icinden): {[ADLAR[i] for i in KARARSIZ5]}")
IX35K = [i for i in IX40 if i not in KARARSIZ5]
R35K = degerlendir(IX35K, "S5  40 - 5 en kararsiz (35)", ayrinti=True)
IXTK = [i for i in IXTUM if i not in KARARSIZ5]
RTK = degerlendir(IXTK, f"S6  {len(IXTK)} (40+yeni - 5 kararsiz)", ayrinti=True)

# ==================================================================
# BOLUM 5 -- NIHAI KARSILASTIRMA VE ONERI
# ==================================================================
print("\n" + "=" * 78)
print("BOLUM 5 -- KUME KARSILASTIRMASI (bolme-disi kapsam, 4 fold ortalamasi)")
print("=" * 78)
R40B = degerlendir(IX40, "S7  40 eksen + HB yonu", ayrinti=True, ekstra=hb_yonu)
RTUMB = degerlendir(IXTUM, f"S8  {NTUM} eksen + HB yonu", ayrinti=True, ekstra=hb_yonu)
HEPSI = [R40, R35, RTUM, R57, R35K, RTK, R40B, RTUMB]
print(
    f"\n{'kume':>34s} {'n':>4s} {'K':>3s} {'net R2':>8s} {'zaman':>8s} "
    f"{'rho^2':>8s} {'skor':>8s} {'sira':>8s} {'zaman-skor':>11s}"
)
for r in HEPSI:
    net = max(r["r2"] - r["null"], 0.0)
    _rz, skz = rho2_ve_skor(max(r["net_zaman"], 0.0))
    print(
        f"{r['baslik'][:34]:>34s} {r['n']:4d} {r['K']:3d} {net:8.5f} {r['net_zaman']:8.5f} "
        f"{r['rho2']:8.5f} {r['skor']:8.5f} {sira_adi(r['skor']):>8s} {skz:11.5f}"
    )
print("\nZAMAN sutunu yalniz zaman-bolmeli iki foldun ortalamasidir. Test 122 gunluk")
print("bir UFUK oldugu icin ihtiyatli okuma budur; trafo-bolmeli foldlar daha")
print("yuksek cikiyor cunku trafo duzeyi yapisi bolmeler arasinda tasiniyor.")
print("\nNOT: 'skor' blok->test tasimasinin TAM oldugunu varsayar (CARPAN=0.798).")
print("m141: blok korelasyonu yaz25'e ozgu; bu sutun UST SINIR gibi okunmalidir.")
print("Kumeler arasi FARK, mutlak duzeyden daha guvenilirdir.")

EN_IYI = max(HEPSI, key=lambda r: r["r2"] - r["null"])
print(f"\nEN YUKSEK NET KAPSAM: {EN_IYI['baslik']}")
for ad, v in GEREKEN.items():
    ger_r2 = v / CARPAN**2
    net = max(EN_IYI["r2"] - EN_IYI["null"], 0.0)
    print(f"  {ad}: gereken R2 {ger_r2:.5f}, olculen {net:.5f}, oran {net / ger_r2:.2f}")


# ==================================================================
# BOLUM 6 -- SIZINTISIZ UCTAN UCA SINAV
#
# Yukaridaki tum kapsam olcumlerinde bir ORTAK KUSUR var: 40 eksen zaten
# TUM yaz25 blogu gorulerek secildi (plasebo kapisi ve tavan kapisi bloktan
# besleniyor). Bolme yalniz ISARET ve H2 agirligini korudu. Bu bolum kusuru
# kapatir: eksen SECIMI de yalnizca KURMA parcasindan yapilir, kapsam
# GORULMEMIS parcada olculur. Ayrica boru hattinin tamamini permute artikla
# tekrarlayarak SECIM SIZINTISININ kendi tabanini olcer.
# ==================================================================
print("\n" + "=" * 78)
print("BOLUM 6 -- SECIM DE BOLMELI: sizintisiz uctan uca sinav")
print("=" * 78)


def sec_kume(y_blok, mask=None, azami=AZAMI_EKSEN, yeni_de=False):
    """m122/m148 secim dongusunun aynisi; blok kapilari yalniz mask'ten beslenir."""
    if mask is None:
        mask = np.ones(NB, dtype=bool)
    w = ww[mask]
    w = w / w.mean()
    y = y_blok[mask]
    m0 = float((w * y * y).mean())
    perm = perm_maske(mask, 20)
    adl, xts, xbs, rss, rcvs, hava = [], [], [], [], [], []
    onc = []
    kaynaklar = [(k["eksen"], False) for k in TARAMA]
    if yeni_de:
        kaynaklar += [(k["eksen"], True) for k in YENI_KAYIT]
    for ad, yeni in kaynaklar:
        if not yeni and len(adl) >= azami:
            continue
        xt, xb = kur_yeni(ad) if yeni else kur(ad)
        if xt is None or xb is None:
            continue
        cc = Gi @ ((V.T @ xt) / N)
        xp0 = xt - V @ cc
        Qs = 1.0 - float((xp0 * xp0).mean())
        if Qs < QS_ALT:
            continue
        rho_s = float((r_hat * xt).mean()) / np.sqrt(Qs)
        if abs(rho_s) < RHO_S_ALT:
            continue
        cc5 = GI5 @ ((V.T @ xt) / N)
        xp5 = xt - V @ cc5
        Qs5 = 1.0 - float((xp5 * xp5).mean())
        if Qs5 < QS_ALT:
            continue
        if abs(float((r_hat * xt).mean()) / np.sqrt(Qs5) - rho_s) > 0.3 * abs(rho_s):
            continue
        xp = xp0.copy()
        for u in onc:
            xp -= float((xp * u).mean()) * u
        Qd = float((xp * xp).mean())
        if Qd < QD_ALT:
            continue
        xm = xb[mask]
        kor = float((w * y * xm).mean()) / np.sqrt(m0)
        gur = float(np.std([float((w * y[p] * xm).mean()) / np.sqrt(m0) for p in perm]))
        if abs(kor) < 3 * gur:
            continue
        rho_cv = CARPAN * kor
        if abs(rho_cv) < TAVAN * abs(rho_s):
            continue
        onc.append(xp / np.sqrt(Qd))
        adl.append(ad)
        xts.append(xt)
        xbs.append(xb)
        rss.append(rho_s)
        rcvs.append(rho_cv)
        hava.append(
            1.0
            if any(
                h in ad
                for h in (
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
            )
            else 0.0
        )
    Xl = np.array(xts)
    del xts, onc
    gc.collect()
    Bl = (V.T @ Xl.T) / N
    Hl = (Xl @ Xl.T) / N - Bl.T @ (Gi @ Bl)
    del Xl
    gc.collect()
    return dict(
        ad=adl,
        H=(Hl + Hl.T) / 2,
        rs=np.array(rss),
        rcv=np.array(rcvs),
        hava=np.array(hava),
        XB=np.array(xbs),
    )


def uctan_uca(baslik, yeni_de, hb=False):
    global KUR_MASKE, HB_XB, HB_Y
    print(f"\n  {baslik}")
    print(f"  {'bolme':>18s} {'eksen':>6s} {'yon':>4s} {'R2':>8s} {'null':>8s} {'ortak':>6s}")
    r2l, n0l, ort = [], [], []
    for ad, kur_m, ev_m in FOLDLAR:
        ctx = sec_kume(rb, mask=kur_m, yeni_de=yeni_de)
        ix = list(range(len(ctx["ad"])))
        KUR_MASKE, HB_XB, HB_Y = kur_m, ctx["XB"], rb
        d = demet(ix, ekstra=hb_yonu if hb else None, ctx=ctx)
        r2 = kapsam(d["D"], ev_m)
        n0, _ns = kapsam_null(d["D"], ev_m)
        ortak = len(set(ctx["ad"]) & set(ADLAR)) / max(len(ctx["ad"]), 1)
        r2l.append(r2)
        n0l.append(n0)
        ort.append(ortak)
        print(f"  {ad:>18s} {len(ctx['ad']):6d} {len(d['A']):4d} {r2:8.5f} {n0:8.5f} {ortak:6.2f}")
    KUR_MASKE, HB_XB, HB_Y = np.ones(NB, dtype=bool), None, None
    net = float(np.mean(r2l) - np.mean(n0l))
    z_net = float(np.mean(r2l[2:]) - np.mean(n0l[2:]))
    rho2, sk = rho2_ve_skor(net)
    print(f"  ORTALAMA net R2 = {net:.5f}  (yalniz zaman bolmeleri {z_net:.5f})")
    print(f"  -> rho^2 {rho2:.5f}   skor {sk:.5f}   {sira_adi(sk)}")
    return dict(
        baslik=baslik,
        r2=float(np.mean(r2l)),
        null=float(np.mean(n0l)),
        net=net,
        net_zaman=z_net,
        rho2=rho2,
        skor=sk,
        ortak=float(np.mean(ort)),
    )


U40 = uctan_uca("40 eksen, secim de bolmeli", yeni_de=False)
U62 = uctan_uca(f"{NTUM} eksen (A-G dahil), secim de bolmeli", yeni_de=True)
U40H = uctan_uca("40 eksen + HB yonu, secim de bolmeli", yeni_de=False, hb=True)
U62H = uctan_uca(f"{NTUM} eksen + HB yonu, secim de bolmeli", yeni_de=True, hb=True)

print("\nBORU HATTI PLASEBOSU: ayni secim+olcum, artik trafo-permute edilmis")
plas = []
for t, p in enumerate(perm_maske(np.ones(NB, dtype=bool), 2)):
    rp = rb[p]
    ctx = sec_kume(rp, mask=None, yeni_de=False)
    ix = list(range(len(ctx["ad"])))
    d = demet(ix, ctx=ctx)
    v = float(np.mean([kapsam(d["D"], ev, y_blok=rp) for _a, _k, ev in FOLDLAR]))
    plas.append(v)
    print(f"  permutasyon {t + 1}: {len(ctx['ad']):2d} eksen secildi, kapsam R2 = {v:.5f}")
print(
    f"  plasebo ortalamasi {np.mean(plas):.5f}   <-> ayni (sizintili) kurulusun "
    f"gercek degeri {R40['r2']:.5f}"
)
print("  Bu, SECIM SIZINTISININ tabanidir: secim de olcum de ayni bloktan besleniyor.")

with open(os.path.join(BURA, "m155_eksen_gozden_gecirme.json"), "w") as fh:
    json.dump(
        dict(
            taban_mse=TABAN_MSE,
            eksen_sayisi=dict(mevcut=N40, yeni=NTUM - N40),
            ters_isaretli=[ADLAR[i] for i in TERS],
            en_kararsiz5=[ADLAR[i] for i in KARARSIZ5],
            yeni_eksenler=YENI_ADLAR,
            kumeler=[
                dict(
                    baslik=r["baslik"],
                    n=r["n"],
                    K=r["K"],
                    r2=r["r2"],
                    null=r["null"],
                    r2_tum_dik=r["r2_tum"],
                    rho2=r["rho2"],
                    skor=r["skor"],
                )
                for r in HEPSI
            ],
            gereken=GEREKEN,
            sizintisiz=[U40, U62, U40H, U62H],
            boru_hatti_plasebo=float(np.mean(plas)),
        ),
        fh,
        indent=1,
    )
print("\n-> m155_eksen_gozden_gecirme.json    HICBIR GONDERIM YAPILMADI.")
