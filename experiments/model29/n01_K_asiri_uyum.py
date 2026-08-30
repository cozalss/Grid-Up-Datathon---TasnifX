"""BLOK-DISI K EGRISI -- eksen sayisi buyudukce kucultme carpani |c| bozuluyor mu?

SORU. m148 su an 50 eksen kullaniyor (m121'den 40 + m144'ten 10). Havuzu
m144_yeni_aileler.json'daki 329 "kapidan_gecen" eksenin tamamiyla ~379'a
genisletmek ongorulen ||BETA||'yi 0.2774 -> 0.497 cikariyor. Ama ongoru ile
GERCEK dis-ornek korelasyon arasindaki oran |c| K buyudukce dusebilir.

KURULUS.
  * Ongoru  P_K = 1.95 * sqrt(sum_{k<=K} rho_s_k^2)   -- rho_s LB span'inden
    gelir, HICBIR bloktan fit edilmez. Bloga bagli TEK sey ISARET'tir.
  * Gerceklesen  kor_B(K) = <rr_B, duz_B(K)>_w / (||duz_B||_w ||rr_B||_w)
    burada duz_B, B blogunda ayni sirayla dikleştirilmiş birim yonlerin
    A blogundan gelen isaretlerle ve LB genliğiyle bileşimidir.
  * ORAN(A,B,K) = 0.798 * kor_B(K) / P_K        (0.798 = CARPAN, m112)

A = secim/isaret blogu, B = olcum blogu. A != B ise bu TAM blok-disi bir
olcumdur: agirlik LB'den, isaret A'dan, olcum B'de.

SINIRLILIK (raporda belirtilecek): havuzun kendisi (329 eksen) yaz25'te
tarandi. Havuz uyeligi tum K'lar icin ORTAK oldugundan K-egiminin yonunu
bozmaz, ama A=yaz25 satirlarinin SEVIYESI sisiktir.

CIKTI: n01_K_asiri_uyum.json + ekrana tablo.
"""

import itertools
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
QD_ALT, NB_ALT = 0.25, 0.02  # blok kapisi gevsek: K araligini 145e kadar acmak icin
BLOKLAR = ["yaz25", "guz25", "kis26"]
K_LISTE = [5, 10, 20, 30, 40, 50, 65, 80, 100, 120, 145]
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0, buzmeli_r_hat  # noqa: E402

# --------------------------------------------------------------- LB span
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
r_hat, gercek, kL = buzmeli_r_hat(V, L, G, N)
TABAN_MSE = float(M0 - 2 * kL + float((r_hat * r_hat).mean()))
print(f"span: V {V.shape}, saf span skoru {np.sqrt(TABAN_MSE):.6f}")

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


def dizi(df, c):
    return df[c].to_numpy(dtype=np.float64, na_value=np.nan)


sgT = tp.soguk_mu.values.astype(np.float64)
svT = st(a0)
ufTh = tp.ufuk_gun.to_numpy().astype(np.float64)
ufT = st(ufTh)
ayT = st(pd.to_datetime(tp.tarih).dt.month.to_numpy().astype(np.float64))
ESIK = {"ust10": (0.9, True), "ust25": (0.75, True), "alt10": (0.1, False)}


# --------------------------------------------------------------- bloklar
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
    return ff, pp, rr, sg, w2 / w2.mean()


BL = {}
for b in BLOKLAR:
    ff, pp, rr, sg, w2 = blok_kur(b)
    BL[b] = dict(ff=ff, pp=pp, rr=rr, sg=sg, w=w2, m0=float((w2 * rr * rr).mean()))
    print(f"blok {b}: {len(rr):,} satir, ||rr||_w = {np.sqrt(BL[b]['m0']):.5f}")

HAVA_SUT = [
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


def hava_paneli(bf):
    """m144'un panel uretecinin AYNISI (gecikme/kumulatif/anomali)."""
    sut = [c for c in HAVA_SUT if c in tp.columns and c in bf.columns]
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
    return pan


class Kurucu:
    """Eksen adindan (test_vektoru, blok_vektoru) uretir.

    m122 (kur), m144 D/E/F/G/H aileleri ve m148 mnt75 kipini kapsar.
    te_* (B ailesi hedef kodlamasi) DISARIDA -- blok secimine bagli
    oldugu icin havuzu bloklar arasi karsilastirilamaz kilardi (3 eksen).
    """

    def __init__(self, bf, pp):
        self.bf = bf
        self.svB = st(pp)
        self.sgB = bf.soguk_mu.values.astype(np.float64)
        self.ufBh = bf.ufuk_gun.to_numpy().astype(np.float64)
        self.ufB = st(self.ufBh)
        self.ayB = st(pd.to_datetime(bf.tarih).dt.month.to_numpy().astype(np.float64))
        self.CARP = {
            "x_sv": (svT, self.svB),
            "x_soguk": (sgT, self.sgB),
            "x_ufuk": (ufT, self.ufB),
            "x_ay": (ayT, self.ayB),
        }
        self._hav = None
        self._hb = {}

    # ---- hava paneli (tembel)
    def hv(self, c):
        if self._hav is None:
            pan = hava_paneli(self.bf)
            at = tp[["lokasyon", "tarih"]].merge(pan, on=["lokasyon", "tarih"], how="left")
            ab = self.bf[["lokasyon", "tarih"]].merge(pan, on=["lokasyon", "tarih"], how="left")
            self._hav = (at, ab)
        at, ab = self._hav
        if c not in at.columns:
            return None, None
        return dizi(at, c), dizi(ab, c)

    def ham(self, kol):
        """Ham kolon (once tp/bf, sonra hava paneli hv_ onekiyle)."""
        if kol.startswith("hv_"):
            return self.hv(kol[3:])
        if kol in tp.columns and kol in self.bf.columns:
            return dizi(tp, kol), dizi(self.bf, kol)
        return None, None

    def kur(self, ad):
        # H ailesi: M[a]x[b]
        if ad.startswith("M[") and "]x[" in ad:
            i = ad.index("]x[")
            a1, b1 = self.kur(ad[2:i])
            a2, b2 = self.kur(ad[i + 3 : -1])
            if a1 is None or a2 is None or b1 is None or b2 is None:
                return None, None
            # m144 H ailesi carpimi FLOAT32 onbellekte yapti; rho_s'in birebir
            # tutmasi icin ayni duyarlikta carpiyoruz (dogrulama kapisi icin).
            return (
                st(a1.astype(np.float32) * a2.astype(np.float32)),
                st(b1.astype(np.float32) * b2.astype(np.float32)),
            )
        # F ailesi / m122 carpimlari: X*Y  (kolon adlarinda * yok)
        if "*" in ad:
            k1, k2 = ad.split("*", 1)
            a1, b1 = self.ham(k1)
            a2, b2 = self.ham(k2)
            if a1 is None or a2 is None:
                return None, None
            a1, b1, a2, b2 = st(a1), st(b1), st(a2), st(b2)
            if any(v is None for v in (a1, b1, a2, b2)):
                return None, None
            return st(a1 * a2), st(b1 * b2)
        # E ailesi: uf_* tek basina
        if ad.startswith("uf_"):
            return self._ufuk(ad)
        # C ailesi: yil_sin/cos harmonikleri
        kol, kip = ad.split(":", 1) if ":" in ad else (ad, "")
        if kol.startswith("yil_"):
            xt, xb = self._yil(kol)
        else:
            xt, xb = self.ham(kol)
        if xt is None:
            return None, None
        if kip == "":
            return st(xt), st(xb)
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
        if kip.startswith("mnt") or kip.startswith("eks"):
            q = int(kip[3:]) / 100.0
            fv = xt[np.isfinite(xt)]
            if fv.size == 0:
                return None, None
            v_ = float(np.quantile(fv, q))
            f = np.maximum if kip.startswith("mnt") else np.minimum
            return st(f(xt - v_, 0.0)), st(f(xb - v_, 0.0))
        if kip == "kare":
            a_, b_ = st(xt), st(xb)
            if a_ is None or b_ is None:
                return None, None
            return st(a_**2), st(b_**2)
        if kip == "uf_log":
            a_, b_ = st(xt), st(xb)
            if a_ is None or b_ is None:
                return None, None
            return st(a_ * st(np.log1p(ufTh))), st(b_ * st(np.log1p(self.ufBh)))
        if kip.startswith("uf_mentese"):
            d = int(kip[len("uf_mentese") :])
            a_, b_ = st(xt), st(xb)
            if a_ is None or b_ is None:
                return None, None
            return (
                st(a_ * np.maximum(ufTh - d, 0)),
                st(b_ * np.maximum(self.ufBh - d, 0)),
            )
        return None, None

    def _ufuk(self, ad):
        if ad.startswith("uf_mentese"):
            d = int(ad[len("uf_mentese") :])
            return st(np.maximum(ufTh - d, 0)), st(np.maximum(self.ufBh - d, 0))
        if ad.startswith("uf_ters"):
            d = int(ad[len("uf_ters") :])
            return st(np.minimum(ufTh - d, 0)), st(np.minimum(self.ufBh - d, 0))
        if ad.startswith("uf_kova"):
            i = int(ad[len("uf_kova") :].split(":")[0])
            kenar = [0, 24, 48, 73, 98, 10**6]
            at = ((ufTh > kenar[i]) & (ufTh <= kenar[i + 1])).astype(np.float64)
            ab = ((self.ufBh > kenar[i]) & (self.ufBh <= kenar[i + 1])).astype(np.float64)
            if ad.endswith(":x_sv"):
                return st(st(at) * svT), st(st(ab) * self.svB)
            return st(at), st(ab)
        if ad == "uf_log":
            return st(np.log1p(ufTh)), st(np.log1p(self.ufBh))
        if ad == "uf_kare":
            return st(ufTh**2), st(self.ufBh**2)
        if ad == "uf_kok":
            return st(np.sqrt(np.maximum(ufTh, 0))), st(np.sqrt(np.maximum(self.ufBh, 0)))
        return None, None

    def _yil(self, kol):
        # yil_sin1 / yil_cos3 ...
        adf, h = kol[4:-1], int(kol[-1])
        f_ = np.sin if adf == "sin" else np.cos
        yg_t = pd.to_datetime(tp.tarih).dt.dayofyear.to_numpy().astype(np.float64)
        yg_b = pd.to_datetime(self.bf.tarih).dt.dayofyear.to_numpy().astype(np.float64)
        return f_(2 * np.pi * h * yg_t / 365.25), f_(2 * np.pi * h * yg_b / 365.25)


# --------------------------------------------------------------- havuz
with open(os.path.join(M29, "m122_nihai.json")) as fh:
    MEVCUT40 = json.load(fh)["eksenler"]
with open(os.path.join(M29, "m144_yeni_aileler.json")) as fh:
    M144 = json.load(fh)
KAPI = [x for x in M144["kapidan_gecen"] if not x["eksen"].startswith("te_")]
ATLANAN_TE = len(M144["kapidan_gecen"]) - len(KAPI)
HAVUZ = list(MEVCUT40) + [x["eksen"] for x in KAPI if x["eksen"] not in set(MEVCUT40)]
print(f"\nhavuz: {len(MEVCUT40)} mevcut + {len(HAVUZ) - len(MEVCUT40)} yeni = {len(HAVUZ)} eksen")
print(f"  (te_* ailesi disarida: {ATLANAN_TE} eksen -- blok secimine bagli hedef kodlamasi)")

# --------------------------------------------------------- TEST UZAYI GS
# rho_s ve dik birim yonler; Q_dik kapisi burada uygulanir (bloktan bagimsiz).
print("\nTEST UZAYI: span cikarma + sirali dikleştirme (Q_dik >= 0.25)")
K0 = Kurucu(BL["yaz25"]["ff"], BL["yaz25"]["pp"])  # sadece xt icin; xt blok-bagimsiz
UT = []
SECILI, RHO_S, RED = [], [], {}
for ad in HAVUZ:
    xt, _ = K0.kur(ad)
    if xt is None:
        RED[ad] = "kurulamadi"
        continue
    cc = Gi @ ((V.T @ xt) / N)
    xp0 = xt - V @ cc
    Qs = 1.0 - float((xp0 * xp0).mean())
    if Qs < 0.02:
        RED[ad] = "Qs"
        continue
    rho_s = float((r_hat * xt).mean()) / np.sqrt(Qs)
    xp = xp0
    for u in UT:
        xp = xp - float((xp * u).mean()) * u
    Qd = float((xp * xp).mean())
    if Qd < QD_ALT:
        RED[ad] = "Qdik"
        continue
    UT.append(xp / np.sqrt(Qd))
    SECILI.append(ad)
    RHO_S.append(rho_s)
RHO_S = np.array(RHO_S)
# KURUCU DOGRULAMASI: yeniden urettigimiz eksenler m144'un kaydettigi
# rho_s ile ayni mi? Ad ayristirmasinda sessiz bir hata olursa burada patlar.
BEK = {x["eksen"]: x["rho_s"] for x in KAPI}
sap = [(a, r, BEK[a]) for a, r in zip(SECILI, RHO_S) if a in BEK and abs(r - BEK[a]) > 1e-6]
print(
    f"  kurucu dogrulamasi: {sum(1 for a in SECILI if a in BEK)} eksen m144 ile "
    f"karsilastirildi, {len(sap)} uyusmazlik"
)
for a, r, b_ in sap[:8]:
    print(f"    ! {a}: bizim {r:+.6f} vs m144 {b_:+.6f}")
assert len(sap) == 0, "eksen kurucusu m144 ile uyusmuyor -- ad ayristirmasi bozuk"
print(f"  {len(SECILI)}/{len(HAVUZ)} eksen dik kaldi (Q_dik>={QD_ALT}); red: ", end="")
print({k: sum(1 for v in RED.values() if v == k) for k in set(RED.values())})
print(f"  1.95*sqrt(sum rho_s^2) tum secili = {TAVAN * np.sqrt(float((RHO_S**2).sum())):.4f}")
del UT  # test uzayindaki yonlere artik ihtiyac yok (ongoru sadece rho_s'ten)


# ------------------------------------------- BLOK HAM VEKTORLERI (onbellek)
# Eksen kurulumu pahalidir; her blok icin BIR kez kurulup float32 diske
# yazilir. Sonrasi saf dogrusal cebir, dolayisiyla cift bazli tekrar ucuz.
ONB = os.environ.get("N01_ONBELLEK") or os.path.join(
    r"C:/Users/Cem/AppData/Local/Temp/claude",
    "c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX",
    "e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad",
)
os.makedirs(ONB, exist_ok=True)
IMZA = str(len(SECILI))
XB, GECERLI = {}, {}
print("\nBLOK HAM VEKTORLERI")
for b in BLOKLAR:
    fx = os.path.join(ONB, f"n01_xb_{b}_{IMZA}.npy")
    fg = os.path.join(ONB, f"n01_gec_{b}_{IMZA}.npy")
    if os.path.exists(fx) and os.path.exists(fg):
        XB[b], GECERLI[b] = np.load(fx), np.load(fg)
        print(f"  {b}: onbellekten ({int(GECERLI[b].sum())}/{len(SECILI)} gecerli)")
        continue
    d = BL[b]
    kr = Kurucu(d["ff"], d["pp"])
    nb_ = len(d["rr"])
    M = np.zeros((len(SECILI), nb_), dtype=np.float32)
    g = np.zeros(len(SECILI), dtype=bool)
    for i, ad in enumerate(SECILI):
        _, xb = kr.kur(ad)
        if xb is None or not np.isfinite(xb).all():
            continue
        M[i] = xb.astype(np.float32)
        g[i] = True
    XB[b], GECERLI[b] = M, g
    np.save(fx, M)
    np.save(fg, g)
    print(
        f"  {b}: {int(g.sum())}/{len(SECILI)} eksen blokta TANIMLI "
        f"(kalani blokta sabit -- or. yaz esikleri kis26'da hic tetiklenmiyor)"
    )

# ham (dikleştirilmemiş) korelasyonlar -- ISARET buradan gelir (m148 ile ayni)
KOR_HAM = {}
for b in BLOKLAR:
    d = BL[b]
    wr = (d["w"] * d["rr"]).astype(np.float64)
    KOR_HAM[b] = (XB[b].astype(np.float64) @ wr) / (len(wr) * np.sqrt(d["m0"]))
    KOR_HAM[b][~GECERLI[b]] = 0.0

KATS_TUM = TAVAN * np.abs(RHO_S)  # LB genligi (isaretsiz), tum SECILI icin


def gs(b, idx):
    """idx sirasiyla agirlikli Gram-Schmidt. (U, kalan_idx, rho) doner."""
    d = BL[b]
    w, rr = d["w"], d["rr"]
    U, kal, rho = [], [], []
    for i in idx:
        u = XB[b][i].astype(np.float64)
        for v in U:
            u -= float((w * u * v).mean()) * v
        n1 = np.sqrt(float((w * u * u).mean()))
        if n1 < NB_ALT:
            continue
        u /= n1
        U.append(u)
        kal.append(i)
        rho.append(float((w * rr * u).mean()) / np.sqrt(d["m0"]))
    return np.array(U), kal, np.array(rho)


def cift_kur(A, B):
    """A ve B'de birlikte ayakta kalan eksen kumesi + iki blokta dik tabanlar."""
    idx = [i for i in range(len(SECILI)) if GECERLI[A][i] and GECERLI[B][i]]
    for _ in range(4):
        _, kA, _ = gs(A, idx)
        _, kB, _ = gs(B, idx)
        ort = set(kA) & set(kB)
        yeni = [i for i in idx if i in ort]
        if yeni == idx:
            break
        idx = yeni
    UA, _, rhoA = gs(A, idx)
    UBx, _, rhoB = gs(B, idx)
    return idx, UA, rhoA, UBx, rhoB


# --------------------------------------------- ISARET TASIMA TANISI
print(f"\nISARET TASIMA TANISI (eksen basina HAM korelasyon, {len(SECILI)} eksen havuzu)")
print(
    f"{'cift':>16s} {'n':>5s} {'isaret uyumu':>13s} {'kor(kor_A,kor_B)':>18s} {'agirlikli uyum':>15s}"
)
TANI = []
for A, B in itertools.combinations(BLOKLAR, 2):
    m = GECERLI[A] & GECERLI[B]
    ka, kb = KOR_HAM[A][m], KOR_HAM[B][m]
    uy = float((np.sign(ka) == np.sign(kb)).mean())
    r = float(np.corrcoef(ka, kb)[0, 1])
    wgt = KATS_TUM[m] / KATS_TUM[m].sum()
    ruy = float((wgt * (np.sign(ka) == np.sign(kb))).sum())
    TANI.append(dict(A=A, B=B, n=int(m.sum()), isaret_uyumu=uy, kor=r, agirlikli_uyum=ruy))
    print(f"{A + '/' + B:>16s} {int(m.sum()):5d} {uy:12.1%} {r:18.3f} {ruy:15.1%}")
print("  %50 = tesaduf. %50'nin ALTI = TERS DONME (sistematik anti-tasima).")

# --------------------------------------------------------- ORAN + BOOTSTRAP
GRUP = {}
for b in BLOKLAR:
    tn = BL[b]["ff"].tanim.values
    uq, gi_ = np.unique(tn, return_inverse=True)
    GRUP[b] = (len(uq), gi_)

rng = np.random.default_rng(20260830)
NBOOT = 400
CIFT = {}
print("\nCIFT BAZLI ORTAK EKSEN SAYILARI")
for A, B in itertools.permutations(BLOKLAR, 2):
    if (B, A) in CIFT:
        idx, UB2, rhoB, UA, rhoA = CIFT[(B, A)]
        CIFT[(A, B)] = (idx, UA, rhoA, UB2, rhoB)
        continue
    CIFT[(A, B)] = cift_kur(A, B)
    print(f"  {A} -> {B}: {len(CIFT[(A, B)][0])} eksen")
for A in BLOKLAR:
    CIFT[(A, A)] = cift_kur(A, A)


def oran_ve_ga(A, B, K, kip="ham"):
    """ORAN(A,B,K) + trafo-kumeli bootstrap %90 GA. K, cift kumesindeki
    ilk K ekseni (|rho_s| sirasi) kullanir."""
    idx, _UA, rhoA, UBx, rhoB = CIFT[(A, B)]
    ii = np.array(idx[:K])
    if kip == "kahin":
        isr = np.sign(rhoB[:K])
    elif kip == "dik":
        isr = np.sign(rhoA[:K])
    else:
        isr = np.sign(KOR_HAM[A][ii])
    isr = np.where(isr == 0, 1.0, isr)
    kat = isr * KATS_TUM[ii]
    P = float(np.sqrt((kat**2).sum()))
    d = BL[B]
    duz = kat @ UBx[:K]
    w, rr = d["w"], d["rr"]
    n_g, gi_ = GRUP[B]
    s1 = np.bincount(gi_, weights=w * rr * duz, minlength=n_g)
    s2 = np.bincount(gi_, weights=w * duz * duz, minlength=n_g)
    s3 = np.bincount(gi_, weights=w * rr * rr, minlength=n_g)

    def _kor(sw):
        t1, t2, t3 = float(sw @ s1), float(sw @ s2), float(sw @ s3)
        return 0.0 if t2 <= 0 or t3 <= 0 else t1 / np.sqrt(t2 * t3)

    kor = _kor(np.ones(n_g))
    pw = np.full(n_g, 1.0 / n_g)
    bs = np.array([_kor(rng.multinomial(n_g, pw).astype(np.float64)) for _ in range(NBOOT)])
    lo, hi = np.quantile(bs, [0.05, 0.95])
    return CARPAN * kor / P, CARPAN * lo / P, CARPAN * hi / P, P, CARPAN * kor


SONUC = []
KIPLER = ["ham", "dik", "kahin"]
print(
    f"\n{'kip':>6s} {'A(secim)':>9s} {'B(olcum)':>9s} {'K':>5s} {'ongoru P_K':>11s} "
    f"{'gercek rho':>11s} {'ORAN':>7s} {'%90 GA':>17s}"
)
for kip in KIPLER:
    for A, B in itertools.product(BLOKLAR, BLOKLAR):
        if kip == "kahin" and A != B:
            continue
        nmax = len(CIFT[(A, B)][0])
        for K in sorted({min(k, nmax) for k in K_LISTE}):
            o, lo, hi, P, gr = oran_ve_ga(A, B, K, kip=kip)
            SONUC.append(
                dict(kip=kip, A=A, B=B, K=K, P=P, gercek=gr, oran=o, lo=lo, hi=hi, ayni=A == B)
            )
            if kip == "ham":
                et = "  (AYNI BLOK)" if A == B else ""
                print(
                    f"{kip:>6s} {A:>9s} {B:>9s} {K:5d} {P:11.4f} {gr:11.4f} {o:7.3f} "
                    f"[{lo:+.3f},{hi:+.3f}]{et}"
                )

K_GRID = sorted({s["K"] for s in SONUC})


def _sec(kip, ayni, K):
    return np.array(
        [s["oran"] for s in SONUC if s["kip"] == kip and s["ayni"] == ayni and s["K"] == K]
    )


def _P(kip, ayni, K):
    return np.array(
        [s["P"] for s in SONUC if s["kip"] == kip and s["ayni"] == ayni and s["K"] == K]
    )


print("\n\nBLOK-DISI OZET (A != B) -- |ORAN|'in K ile gidisi")
print(
    f"{'K':>5s} {'cift':>5s} {'ort P_K':>8s} {'ort ORAN':>9s} {'ort |ORAN|':>11s} "
    f"{'sd':>7s} {'|ORAN| / ilk':>13s}   cift cift"
)
OZET, ilk = {}, None
for K in K_GRID:
    v = _sec("ham", False, K)
    if len(v) < 2:
        continue
    p = float(_P("ham", False, K).mean())
    OZET[K] = dict(
        n=len(v),
        P=p,
        ort=float(v.mean()),
        mut=float(np.abs(v).mean()),
        sd=float(v.std(ddof=1)),
        tek=[float(x) for x in v],
    )
    if ilk is None:
        ilk = OZET[K]["mut"]
    print(
        f"{K:5d} {len(v):5d} {p:8.4f} {v.mean():9.3f} {np.abs(v).mean():11.3f} "
        f"{v.std(ddof=1):7.3f} {np.abs(v).mean() / ilk:13.3f}   "
        + " ".join(f"{x:+6.3f}" for x in v)
    )

print("\nAYNI BLOK (A=B, sisik ust sinir)  |  ISARET KAHINI (uygulanamaz ust sinir)")
for K in K_GRID:
    v, k2 = _sec("ham", True, K), _sec("kahin", True, K)
    if len(v) < 2:
        continue
    print(
        f"{K:5d}  ayni {v.mean():6.3f} ["
        + " ".join(f"{x:+6.3f}" for x in v)
        + f"]   kahin {k2.mean():6.3f} ["
        + " ".join(f"{x:+6.3f}" for x in k2)
        + "]"
    )

print("\nDUYARLILIK (kip=dik: isaret A'nin dikleştirilmiş yonunden), blok-disi")
for K in K_GRID:
    v = _sec("dik", False, K)
    if len(v) < 2:
        continue
    print(
        f"{K:5d} ort {v.mean():7.3f}  ort|.| {np.abs(v).mean():6.3f}   "
        + " ".join(f"{x:+6.3f}" for x in v)
    )

print("\n\nBEKLENEN GERCEKLESEN |rho| = ort|ORAN|(K) * ort P_K   (blok-disi)")
print(f"{'K':>5s} {'P_K':>9s} {'ort|ORAN|':>10s} {'|rho|_bek':>10s} {'skor(nihai)':>12s}")
EN, ENK = -1e9, None
for K, o in OZET.items():
    rho_bek = o["mut"] * o["P"]
    skor = np.sqrt(max(TABAN_MSE - rho_bek**2, 1e-12))
    print(f"{K:5d} {o['P']:9.4f} {o['mut']:10.3f} {rho_bek:10.4f} {skor:12.5f}")
    if rho_bek > EN:
        EN, ENK = rho_bek, K
print(
    f"\nOPTIMAL K = {ENK}  (beklenen |rho| {EN:.4f}, skor {np.sqrt(max(TABAN_MSE - EN**2, 0)):.5f})"
)
print("UYARI: ort ORAN NEGATIF ise isaret bloklar arasi TASINMIYOR demektir;")
print("  |ORAN| o zaman bir KAZANC degil, yalnizca sinyal BUYUKLUGUDUR.")

with open(os.path.join(M29, "n01_K_asiri_uyum.json"), "w") as fh:
    json.dump(
        dict(
            havuz=len(HAVUZ),
            secili=len(SECILI),
            adlar=SECILI,
            rho_s=[float(x) for x in RHO_S],
            gecerli={b: [bool(x) for x in GECERLI[b]] for b in BLOKLAR},
            cift_n={f"{A}->{B}": len(CIFT[(A, B)][0]) for A, B in CIFT},
            sonuc=SONUC,
            ozet={str(k): v for k, v in OZET.items()},
            tani=TANI,
            kor_ham={b: [float(x) for x in KOR_HAM[b]] for b in BLOKLAR},
            taban_mse=TABAN_MSE,
            optimal_K=ENK,
        ),
        fh,
        indent=1,
    )
print("\nyazildi: n01_K_asiri_uyum.json")
