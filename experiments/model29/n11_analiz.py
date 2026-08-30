"""n11 FAZ 3 -- eksen ALT KUMESI ve AGIRLIKLANDIRMA yontemlerinin
BLOK-DISI karsilastirmasi.

Girdi: n11_eksen_secimi.py'nin faz1/faz2 ciktilari (scratchpad).
Cikti: experiments/model29/n11_eksen_secimi.json

DUZEN. Test ufku 2026-04-01..07-31; yaz25 blogu 2025-04-01..07-31, yani
MEVSIMSEL OLARAK ESLESEN tek blok. guz25 = 08..11, kis26 = 12..03.
kis26'da havuzun yalnizca 97/423 ekseni kurulabiliyor (yaz kolonlari kisin
sabit), bu yuzden kis26 ANA KATLAMADA KULLANILMAZ.

  ANA KATLAMA : uydur = guz25, olc = yaz25   (olcum blogu test'e en yakin)
  SIMETRIK    : uydur = yaz25, olc = guz25   (m148'in GERCEK kurulusu,
                yani rho_cv'yi yaz25'ten alan hali, blok DISINDA olculur)

Her iki katlamada da rho_s liderlik tablosundan gelir (bloklardan bagimsiz).
Ic ice ayar (buzme katsayisi, ridge lambda) UYDURMA blogunun trafo
gruplarinin IKI YARISI arasinda yapilir -- olcum bloguna hic dokunulmaz.
"""

import json
import os

import numpy as np

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")
ARA = (
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad"
)
CARPAN, TAVAN, QD_ALT = 0.798, 1.95, 0.25
AZAMI_EKSEN = 40
KLER = (10, 20, 25, 40, 60, 100, 136)
BOOT = 3000
ILERI_AZAMI = 40  # acgozlu arama maliyeti; K > 40'ta yol on eki kullanilir

F1 = np.load(os.path.join(ARA, "n11_faz1.npz"))
Gt = F1["Gt"]
RHO_S = F1["rho_s"]
with open(os.path.join(ARA, "n11_havuz.json"), encoding="utf-8") as fh:
    HAVUZ = json.load(fh)
P = len(HAVUZ)
YENI = np.array([h["yeni"] for h in HAVUZ])
ADLAR = [h["eksen"] for h in HAVUZ]


class Blok:
    def __init__(self, ad):
        d = np.load(os.path.join(ARA, f"n11_blok_{ad}.npz"))
        self.ad = ad
        self.Gk = d["Gk"].astype(np.float64)
        self.gk = d["gk"]
        self.mk = d["mk"]
        self.G = self.Gk.sum(axis=0)
        self.g = self.gk.sum(axis=0)
        self.m0 = float(self.mk.sum())
        self.kor = d["kor"]
        self.gur = d["gur"]
        self.gecerli = d["gecerli"]
        self.rho_cv = CARPAN * self.kor
        # m148'in blok kapilari
        self.plasebo = np.abs(self.kor) >= 3 * self.gur
        self.tavan = np.abs(self.rho_cv) >= TAVAN * np.abs(RHO_S)
        self.kapi = self.gecerli & self.plasebo & self.tavan

    def alt(self, kumeler):
        """Kume alt kumesinden (ya da onyukleme sayacindan) G, g, m0."""
        w = np.asarray(kumeler, dtype=np.float64)
        return (
            np.tensordot(w, self.Gk, axes=(0, 0)),
            w @ self.gk,
            float(w @ self.mk),
        )


BL = {a: Blok(a) for a in ("yaz25", "guz25", "kis26")}


# --------------------------------------------------------- Gram-Schmidt
def gs_ekle(T, i):
    """xp_i'yi mevcut ortonormal tabana (T satirlari) ekler.
    Doner: (Qd, yeni T satiri) ya da Qd < esik ise (Qd, None)."""
    gi = Gt[:, i]
    if len(T) == 0:
        qd = Gt[i, i]
        if qd < QD_ALT:
            return qd, None
        t = np.zeros(P)
        t[i] = 1.0 / np.sqrt(qd)
        return qd, t
    Tm = np.array(T)
    pj = Tm @ gi
    qd = Gt[i, i] - float(pj @ pj)
    if qd < QD_ALT:
        return qd, None
    t = -pj @ Tm
    t[i] += 1.0
    return qd, t / np.sqrt(qd)


def m148_sirasi(bfit, azami):
    """m148'in KENDI dongusu: m121 taramasi (en fazla 40 KABUL), sonra
    m144'un yenileri; her adayda blok kapilari + Q_dik >= 0.25."""
    T, sec = [], []
    n121 = 0
    for i in range(P):
        if len(sec) >= azami:
            break
        if not YENI[i] and n121 >= AZAMI_EKSEN:
            continue
        if not bfit.kapi[i]:
            continue
        qd, t = gs_ekle(T, i)
        if t is None:
            continue
        T.append(t)
        sec.append(i)
        if not YENI[i]:
            n121 += 1
    return sec, np.array(T)


def rho(c, G, g, m0):
    v = float(c @ G @ c)
    if v <= 1e-14 or m0 <= 0:
        return 0.0
    return CARPAN * float(c @ g) / np.sqrt(m0 * v)


def c_den_k(T, k):
    return T.T @ np.asarray(k, dtype=np.float64)


# ----------------------------------------------------------- agirliklar
def agirlik(kip, sec, bfit, lam=None):
    s = np.array(sec)
    rs, rcv = RHO_S[s], bfit.rho_cv[s]
    se = CARPAN * bfit.gur[s]
    isr = np.sign(rcv)
    isr[isr == 0] = 1.0
    if kip == "a_m148":
        return isr * TAVAN * np.abs(rs)
    if kip == "b_rho_cv":
        return rcv
    if kip == "c_buzme":
        return isr * (lam * np.abs(rcv) + (1 - lam) * TAVAN * np.abs(rs))
    if kip == "e_guven":
        f = np.maximum(0.0, 1.0 - (se**2) / np.maximum(rcv**2, 1e-12))
        return rcv * f
    if kip == "esit":
        return isr * np.ones(len(s))
    raise ValueError(kip)


def ridge_c(sec, G, g, lam):
    s = np.array(sec)
    Gs = G[np.ix_(s, s)]
    d = np.trace(Gs) / len(s)
    cs = np.linalg.solve(Gs + lam * d * np.eye(len(s)), g[s])
    c = np.zeros(P)
    c[s] = cs
    return c


# ------------------------------------------------- ic ice ayar (uydurma)
def ic_bolme(b):
    nk = len(b.mk)
    yA = np.zeros(nk)
    yA[: nk // 2] = 1.0
    yB = 1.0 - yA
    return b.alt(yA), b.alt(yB)


def lam_sec_buzme(sec, bfit, T):
    (GA, gA, mA), (GB, gB, mB) = ic_bolme(bfit)
    s = np.array(sec)
    rs = RHO_S[s]
    en, enl = -1.0, 0.0
    for lam in np.linspace(0, 1, 21):
        tot = 0.0
        for (Gf, gf, mf), (Ge, ge, me) in (
            ((GA, gA, mA), (GB, gB, mB)),
            ((GB, gB, mB), (GA, gA, mA)),
        ):
            rcv = CARPAN * (gf[s] / np.sqrt(mf * np.maximum(np.diag(Gf)[s], 1e-12)))
            isr = np.sign(rcv)
            isr[isr == 0] = 1.0
            k = isr * (lam * np.abs(rcv) + (1 - lam) * TAVAN * np.abs(rs))
            tot += abs(rho(c_den_k(T, k), Ge, ge, me))
        if tot > en:
            en, enl = tot, lam
    return enl


def lam_sec_ridge(sec, bfit):
    (GA, gA, mA), (GB, gB, mB) = ic_bolme(bfit)
    en, enl = -1.0, 1.0
    for lam in (1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        tot = 0.0
        for (Gf, gf, mf), (Ge, ge, me) in (
            ((GA, gA, mA), (GB, gB, mB)),
            ((GB, gB, mB), (GA, gA, mA)),
        ):
            c = ridge_c(sec, Gf, gf, lam)
            tot += abs(rho(c, Ge, ge, me))
        if tot > en:
            en, enl = tot, lam
    return enl


# ------------------------------------------------------- ileri (acgozlu)
def ileri_secim(bfit, azami, lam):
    """Her adimda IC AYRIMDA blok-disi gerceklesen |rho|'yu en cok artiran
    ekseni ekler (A'da uydur -> B'de olc ve tersi; ortalama)."""
    (GA, gA, mA), (GB, gB, mB) = ic_bolme(bfit)
    ciftler = (((GA, gA, mA), (GB, gB, mB)), ((GB, gB, mB), (GA, gA, mA)))
    aday = [i for i in range(P) if bfit.kapi[i]]
    T, sec = [], []
    while len(sec) < azami:
        en, eni, ent = -1.0, None, None
        for i in aday:
            if i in sec:
                continue
            qd, t = gs_ekle(T, i)
            if t is None:
                continue
            s2 = [*sec, i]
            tot = 0.0
            for (Gf, gf, mf), (Ge, ge, me) in ciftler:
                c = ridge_c(s2, Gf, gf, lam)
                tot += abs(rho(c, Ge, ge, me))
            if tot > en:
                en, eni, ent = tot, i, t
        if eni is None:
            break
        sec.append(eni)
        T.append(ent)
    return sec, np.array(T)


# -------------------------------------------------------------- onyukleme
def boot_istatistik(c, bev):
    a = np.einsum("i,kij,j->k", c, bev.Gk, c)
    b = bev.gk @ c
    return a, b, bev.mk


def boot_rho(a, b, m, W):
    v = W @ a
    num = W @ b
    den = np.sqrt(np.maximum((W @ m) * v, 1e-30))
    return CARPAN * num / den


def calis(fit_ad, ev_ad, etiket):
    bfit, bev = BL[fit_ad], BL[ev_ad]
    ortak = bfit.gecerli & bev.gecerli
    sonuc = {}
    rng = np.random.default_rng(11)
    nk = len(bev.mk)
    W = rng.multinomial(nk, np.ones(nk) / nk, size=BOOT).astype(np.float64)

    print(f"\n===== {etiket}: uydur={fit_ad}  olc={ev_ad} =====")
    print(f"kapidan gecen aday: {int((bfit.kapi & ortak).sum())} / {P}")
    # ACGOZLU YOL bir kez kurulur (ic ice); K'lar bu yolun on eklerdir.
    sec40, T40 = m148_sirasi(bfit, ILERI_AZAMI)
    lam_g = lam_sec_ridge(sec40, bfit) if len(sec40) >= 2 else 1.0
    yol, Tyol = ileri_secim(bfit, ILERI_AZAMI, lam_g)
    print(f"acgozlu yol uzunlugu {len(yol)} (lam={lam_g:g})")
    for K in KLER:
        sec, T = m148_sirasi(bfit, K)
        if len(sec) < 2:
            continue
        satir = {}
        kayit = {}
        lamb = lam_sec_buzme(sec, bfit, T)
        lamr = lam_sec_ridge(sec, bfit)
        yontemler = {
            "a_m148": c_den_k(T, agirlik("a_m148", sec, bfit)),
            "b_rho_cv": c_den_k(T, agirlik("b_rho_cv", sec, bfit)),
            "c_buzme": c_den_k(T, agirlik("c_buzme", sec, bfit, lamb)),
            "e_guven": c_den_k(T, agirlik("e_guven", sec, bfit)),
            "esit": c_den_k(T, agirlik("esit", sec, bfit)),
            "f_ridge": ridge_c(sec, bfit.G, bfit.g, lamr),
        }
        seci = yol[: min(K, len(yol))]
        if len(seci) >= 2:
            yontemler["d_ileri"] = ridge_c(seci, bfit.G, bfit.g, lam_g)
        # ust sinir: OLCUM blogunda dogrudan en kucuk kareler (SIZINTILI,
        # yalnizca tavan gostergesi)
        yontemler["_tavan_sizintili"] = ridge_c(sec, bev.G, bev.g, 1e-3)

        ist = {ad: boot_istatistik(c, bev) for ad, c in yontemler.items()}
        r0 = {ad: rho(c, bev.G, bev.g, bev.m0) for ad, c in yontemler.items()}
        a0, b0, m0 = ist["a_m148"]
        ba = boot_rho(a0, b0, m0, W)
        for ad in yontemler:
            aa, bb, mm = ist[ad]
            bx = boot_rho(aa, bb, mm, W)
            d = np.abs(bx) - np.abs(ba)
            orn = np.abs(bx)
            satir[ad] = dict(
                rho=float(r0[ad]),
                rho_ao=[float(np.quantile(orn, 0.025)), float(np.quantile(orn, 0.975))],
                fark=float(abs(r0[ad]) - abs(r0["a_m148"])),
                fark_ao=[float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
                p_iyi=float((d > 0).mean()),
            )
        kayit["sec"] = [ADLAR[i] for i in sec]
        kayit["lam_buzme"] = float(lamb)
        kayit["lam_ridge"] = float(lamr)
        kayit["lam_ileri"] = float(lam_g)
        if len(seci) >= 2:
            kayit["ileri_sec"] = [ADLAR[i] for i in seci]
        sonuc[str(K)] = dict(yontem=satir, meta=kayit, n_eksen=len(sec))

        print(f"\n  K={K} (secilen {len(sec)})  lam_buzme={lamb:.2f} lam_ridge={lamr:g}")
        for ad in (
            "a_m148",
            "b_rho_cv",
            "c_buzme",
            "e_guven",
            "esit",
            "f_ridge",
            "d_ileri",
            "_tavan_sizintili",
        ):
            if ad not in satir:
                continue
            v = satir[ad]
            print(
                f"    {ad:>17s} rho={v['rho']:+.4f} "
                f"[{v['rho_ao'][0]:.4f},{v['rho_ao'][1]:.4f}] "
                f"fark={v['fark']:+.4f} [{v['fark_ao'][0]:+.4f},{v['fark_ao'][1]:+.4f}] "
                f"P(iyi)={v['p_iyi']:.2f}"
            )
    return sonuc


if __name__ == "__main__":
    out = {
        "aciklama": (
            "Eksen alt kumesi ve agirliklandirma yontemlerinin blok-disi "
            "karsilastirmasi. rho = CARPAN * agirlikli korelasyon; yalnizca "
            "BETA'nin YONU'ne bagli. Guven araliklari trafo kumesi onyuklemesi "
            f"(B={BOOT}, {len(BL['yaz25'].mk)} kume)."
        ),
        "havuz": P,
        "ana": calis("guz25", "yaz25", "ANA"),
        "simetrik": calis("yaz25", "guz25", "SIMETRIK"),
    }
    with open(os.path.join(M29, "n11_eksen_secimi.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\nyazildi: experiments/model29/n11_eksen_secimi.json")
