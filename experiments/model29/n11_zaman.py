"""n11 -- GECERLI olcum duzeni: yaz25 ICINDE ZAMAN bolmesi.

NEDEN BLOK-DISI DEGIL. Gorev "yaz25'te sec, guz25+kis26'da olc" dedi.
OLCTUM: bu duzen GECERSIZ. Kanit (n11_isaret.py):
    rho_s (28 liderlik olcumunden kurulan r_hat ile TEST uzayinda
    hesaplanan korelasyon) ile blok korelasyonunun isaret uyumu
        yaz25 : 0.969   kor = +0.834
        guz25 : 0.488   kor = +0.184     (yazi-tura)
        kis26 : 0.302   kor = -0.426     (TERS)
Yani yalnizca yaz25 test ile ayni yonu gosteriyor. Test ufku
2026-04-01..07-31, yaz25 ise 2025-04-01..07-31: ayni mevsim, bir yil
oncesi. guz25/kis26'da secim yapmak bizi TERS yone goturur.

GECERLI VEKIL. yaz25 icinde ZAMAN bolmesi:
    yari 0 = Nisan-Mayis (131,310 satir, m0 = 0.519)
    yari 1 = Haziran-Temmuz (143,619 satir, m0 = 1.363)
Bir yarida SEC + AGIRLIKLANDIR, DIGERINDE OLC; iki yon de calisir ve
ortalanir. Iki yari rejim olarak cok farkli (m0 2.6 kat), yani bu SERT
bir genelleme sinavi. Test'in kaymasi (bir yil ileri, ayni mevsim)
elimizdeki en yakin benzeridir.

Onyukleme: olcum yarisinin trafo (tanim) kumeleri yeniden orneklenir.
"""

import json
import os

import n11_analiz as A
import numpy as np
import pandas as pd
from n11_eksen_secimi import AO, ARA, DN, HEDEF_SOGUK, SIC_AILE, kolonlar, st, yap_kur

KUME = 80
BOOT = 4000
KLER = (5, 8, 10, 13, 17, 20, 25, 30, 40, 50, 60, 80, 100, 136)
YOL = os.path.join(ARA, "n11_yari_gram.npz")
RHO_S = A.RHO_S
P = A.P
ADLAR = A.ADLAR


def kur_gramlar():
    import pyarrow.parquet as pq

    ih = set()
    for a in ADLAR:
        kolonlar(a, ih)
    tumk = set(pq.read_schema(os.path.join(DN, "test.parquet")).names)
    ekstra = ["soguk_mu", "ufuk_gun", "tarih", "tanim", "tuketim", "_blok"]
    e = pd.read_parquet(
        os.path.join(DN, "egitim.parquet"), columns=sorted((ih & tumk) | set(ekstra))
    )
    tp_ref = pd.read_parquet(os.path.join(DN, "test.parquet"), columns=sorted(ih & tumk))
    blk = e[e._blok == "yaz25"]
    sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
    Pr = [
        np.load(os.path.join(AO, f"yaz25_{t}_{aa}_uretim.npy")).astype(np.float64)
        for t, aa in SIC_AILE
        if os.path.exists(os.path.join(AO, f"yaz25_{t}_{aa}_uretim.npy"))
    ]
    z = np.load(os.path.join(DN, "soguk_tahmin_yaz25.npz"))
    idx = np.concatenate([sic.index.values, sog.index.values])
    pb = np.concatenate([np.mean(Pr, axis=0), np.mean([z[q] for q in z.files], axis=0)])
    bf = e.loc[idx].copy()
    del e
    rb = np.log1p(bf.tuketim.values.astype(np.float64)) - pb
    sgm = bf.soguk_mu.values.astype(np.float64)
    ww = np.where(sgm == 1, HEDEF_SOGUK / sgm.mean(), (1 - HEDEF_SOGUK) / (1 - sgm.mean()))
    ww = ww / ww.mean()
    nb = len(rb)
    carpB = {
        "x_sv": st(pb),
        "x_soguk": sgm,
        "x_ufuk": st(bf.ufuk_gun.to_numpy()),
        "x_ay": st(pd.to_datetime(bf.tarih).dt.month.to_numpy().astype(np.float64)),
    }
    kur = yap_kur(bf, carpB, tp_ref)
    X = np.zeros((P, nb), dtype=np.float32)
    for i, a in enumerate(ADLAR):
        v = kur(a)
        if v is not None:
            X[i] = v
    del tp_ref
    ay = pd.to_datetime(bf.tarih).dt.month.to_numpy()
    yari = [ay <= 5, ay >= 6]
    tn = bf.tanim.values
    uqn = pd.unique(tn)
    gi = pd.Series(np.arange(len(uqn)), index=uqn)[tn].to_numpy()
    kg = np.random.default_rng(7).integers(0, KUME, size=len(uqn))[gi]

    Gk = np.zeros((2, KUME, P, P), dtype=np.float32)
    gk = np.zeros((2, KUME, P))
    mk = np.zeros((2, KUME))
    kor = np.zeros((2, P))
    gur = np.zeros((2, P))
    for j, msk in enumerate(yari):
        nj = int(msk.sum())
        Xj = X[:, msk]
        wj = ww[msk]
        rj = rb[msk]
        m0j = float((wj * rj * rj).mean())
        kor[j] = (Xj @ (wj * rj).astype(np.float32)) / nj / np.sqrt(m0j)
        # plasebo: YARI ICINDE trafo gruplarini permute et (m148'in kapisi)
        rngp = np.random.default_rng(5)
        gj = gi[msk]
        uj = np.unique(gj)
        gjl = pd.Series(np.arange(len(uj)), index=uj)[gj].to_numpy()
        pz = []
        for _ in range(20):
            sp = np.argsort(
                np.argsort(rngp.permutation(len(uj))[gjl], kind="stable"), kind="stable"
            )
            pz.append((Xj @ (wj * rj[sp]).astype(np.float32)) / nj / np.sqrt(m0j))
        gur[j] = np.array(pz).std(axis=0)
        kgj = kg[msk]
        for q in range(KUME):
            m = kgj == q
            Xs = Xj[:, m]
            Gk[j, q] = (Xs * wj[m].astype(np.float32)) @ Xs.T / nj
            gk[j, q] = Xs.astype(np.float64) @ (wj[m] * rj[m]) / nj
            mk[j, q] = float((wj[m] * rj[m] * rj[m]).sum()) / nj
        print(f"  yari {j}: {nj:,} satir m0={m0j:.4f}", flush=True)
        del Xj, Xs
    np.savez(YOL, Gk=Gk, gk=gk, mk=mk, kor=kor, gur=gur)


if not os.path.exists(YOL):
    print("yari gramlari kuruluyor...", flush=True)
    kur_gramlar()
D = np.load(YOL)


class Yari:
    def __init__(self, j):
        self.Gk = D["Gk"][j].astype(np.float64)
        self.gk = D["gk"][j]
        self.mk = D["mk"][j]
        self.G = self.Gk.sum(0)
        self.g = self.gk.sum(0)
        self.m0 = float(self.mk.sum())
        self.kor = D["kor"][j]
        self.gur = D["gur"][j]
        self.gecerli = np.ones(P, dtype=bool)
        self.rho_cv = A.CARPAN * self.kor
        self.plasebo = np.abs(self.kor) >= 3 * np.maximum(self.gur, 1e-12)
        self.tavan = np.abs(self.rho_cv) >= A.TAVAN * np.abs(RHO_S)
        self.kapi = self.plasebo & self.tavan

    def alt(self, w):
        w = np.asarray(w, dtype=np.float64)
        return np.tensordot(w, self.Gk, axes=(0, 0)), w @ self.gk, float(w @ self.mk)


Y = [Yari(0), Yari(1)]
print(f"kapidan gecen: yari0={int(Y[0].kapi.sum())} yari1={int(Y[1].kapi.sum())} / {P}")


def kats(kip, sec, bf, lam=None):
    s = np.array(sec)
    rs, rcv = RHO_S[s], bf.rho_cv[s]
    se = A.CARPAN * bf.gur[s]
    isr = np.sign(rcv)
    isr[isr == 0] = 1.0
    isr_s = np.sign(rs)
    isr_s[isr_s == 0] = 1.0
    if kip == "a_m148":
        return isr * A.TAVAN * np.abs(rs)
    if kip == "b_rho_cv":
        return rcv
    if kip == "c_buzme":
        return isr * (lam * np.abs(rcv) + (1 - lam) * A.TAVAN * np.abs(rs))
    if kip == "e_guven":
        return rcv * np.maximum(0.0, 1.0 - se**2 / np.maximum(rcv**2, 1e-12))
    if kip == "esit":
        return isr * np.ones(len(s))
    if kip == "h_isaret_rho_s":
        return isr_s * A.TAVAN * np.abs(rs)
    if kip == "i_ters_var":
        # ters-varyans: guvenilir eksene daha cok agirlik
        return rcv / np.maximum(se**2 + np.median(se) ** 2, 1e-12) * np.median(se) ** 2
    raise ValueError(kip)


def ic_yari(bf):
    """UYDURMA yarisinin kendi kumelerini ikiye bolerek ic ayar."""
    nk = len(bf.mk)
    a = np.zeros(nk)
    a[: nk // 2] = 1.0
    return bf.alt(a), bf.alt(1 - a)


def lam_ridge_sec(sec, bf):
    (GA, gA, mA), (GB, gB, mB) = ic_yari(bf)
    en, enl = -1.0, 1.0
    for lam in (1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        t = 0.0
        for (Gf, gf, mf), (Ge, ge, me) in (
            ((GA, gA, mA), (GB, gB, mB)),
            ((GB, gB, mB), (GA, gA, mA)),
        ):
            t += abs(A.rho(A.ridge_c(sec, Gf, gf, lam), Ge, ge, me))
        if t > en:
            en, enl = t, lam
    return enl


def lam_buzme_sec(sec, bf, T):
    (GA, gA, mA), (GB, gB, mB) = ic_yari(bf)
    s = np.array(sec)
    en, enl = -1.0, 0.0
    for lam in np.linspace(0, 1, 21):
        t = 0.0
        for (Gf, gf, mf), (Ge, ge, me) in (
            ((GA, gA, mA), (GB, gB, mB)),
            ((GB, gB, mB), (GA, gA, mA)),
        ):
            rcv = A.CARPAN * gf[s] / np.sqrt(mf * np.maximum(np.diag(Gf)[s], 1e-12))
            isr = np.sign(rcv)
            isr[isr == 0] = 1.0
            k = isr * (lam * np.abs(rcv) + (1 - lam) * A.TAVAN * np.abs(RHO_S[s]))
            t += abs(A.rho(A.c_den_k(T, k), Ge, ge, me))
        if t > en:
            en, enl = t, lam
    return enl


def ileri(bf, azami, lam):
    (GA, gA, mA), (GB, gB, mB) = ic_yari(bf)
    ciftler = (((GA, gA, mA), (GB, gB, mB)), ((GB, gB, mB), (GA, gA, mA)))
    aday = [i for i in range(P) if bf.kapi[i]]
    T, sec = [], []
    while len(sec) < azami:
        en, eni, ent = -1.0, None, None
        for i in aday:
            if i in sec:
                continue
            qd, t = A.gs_ekle(T, i)
            if t is None:
                continue
            s2 = [*sec, i]
            tot = 0.0
            for (Gf, gf, mf), (Ge, ge, me) in ciftler:
                tot += abs(A.rho(A.ridge_c(s2, Gf, gf, lam), Ge, ge, me))
            if tot > en:
                en, eni, ent = tot, i, t
        if eni is None:
            break
        sec.append(eni)
        T.append(ent)
    return sec, np.array(T)


YONTEMLER = (
    "a_m148",
    "b_rho_cv",
    "c_buzme",
    "e_guven",
    "i_ters_var",
    "h_isaret_rho_s",
    "esit",
    "f_ridge",
    "d_ileri",
)
ILERI_AZAMI = 40

rngb = np.random.default_rng(23)
W = [rngb.multinomial(KUME, np.ones(KUME) / KUME, size=BOOT).astype(np.float64) for _ in range(2)]


def boot(c, y, j):
    a = np.einsum("i,kij,j->k", c, y.Gk, c)
    b = y.gk @ c
    v = W[j] @ a
    return A.CARPAN * (W[j] @ b) / np.sqrt(np.maximum((W[j] @ y.mk) * v, 1e-30))


yol_ileri = {}
for f in (0, 1):
    s40, _ = A.m148_sirasi(Y[f], ILERI_AZAMI)
    lg = lam_ridge_sec(s40, Y[f]) if len(s40) >= 2 else 1.0
    yol_ileri[f] = (ileri(Y[f], ILERI_AZAMI, lg)[0], lg)
    print(f"acgozlu yol yari{f}: {len(yol_ileri[f][0])} eksen, lam={lg:g}", flush=True)

SON = {}
print(f"\n{'K':>4s} " + " ".join(f"{y:>15s}" for y in YONTEMLER))
for K in KLER:
    r = {y: [] for y in YONTEMLER}
    bt = {y: [] for y in YONTEMLER}
    nsec = []
    for f in (0, 1):
        ev = 1 - f
        bf, be = Y[f], Y[ev]
        sec, T = A.m148_sirasi(bf, K)
        if len(sec) < 2:
            continue
        nsec.append(len(sec))
        lamb = lam_buzme_sec(sec, bf, T)
        lamr = lam_ridge_sec(sec, bf)
        cs = {}
        for y in YONTEMLER:
            if y == "f_ridge":
                cs[y] = A.ridge_c(sec, bf.G, bf.g, lamr)
            elif y == "d_ileri":
                yol, lg = yol_ileri[f]
                s2 = yol[: min(K, len(yol))]
                cs[y] = A.ridge_c(s2, bf.G, bf.g, lg) if len(s2) >= 2 else None
            elif y == "c_buzme":
                cs[y] = A.c_den_k(T, kats(y, sec, bf, lamb))
            else:
                cs[y] = A.c_den_k(T, kats(y, sec, bf))
        for y, c in cs.items():
            if c is None:
                continue
            r[y].append(A.rho(c, be.G, be.g, be.m0))
            bt[y].append(np.abs(boot(c, be, ev)))
    if not nsec:
        continue
    kayit = {}
    ba = np.concatenate(bt["a_m148"]) if len(bt["a_m148"]) == 2 else bt["a_m148"][0]
    ort_a = float(np.mean([abs(v) for v in r["a_m148"]]))
    for y in YONTEMLER:
        if not r[y]:
            continue
        ort = float(np.mean([abs(v) for v in r[y]]))
        bx = np.mean(bt[y], axis=0) if len(bt[y]) == 2 else bt[y][0]
        bA = np.mean(bt["a_m148"], axis=0) if len(bt["a_m148"]) == 2 else bt["a_m148"][0]
        d = bx - bA
        kayit[y] = dict(
            rho=ort,
            rho_ao=[float(np.quantile(bx, 0.025)), float(np.quantile(bx, 0.975))],
            fark=ort - ort_a,
            fark_ao=[float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            p_iyi=float((d > 0).mean()),
            yuzde=100.0 * (ort - ort_a) / max(ort_a, 1e-9),
        )
    SON[str(K)] = dict(n_eksen=nsec, yontem=kayit)
    print(
        f"{K:>4d} "
        + " ".join(f"{kayit[y]['rho']:15.4f}" if y in kayit else " " * 15 for y in YONTEMLER)
    )

with open(os.path.join(ARA, "n11_zaman_sonuc.json"), "w", encoding="utf-8") as fh:
    json.dump(SON, fh, ensure_ascii=False, indent=1)
print("\nkaydedildi", os.path.join(ARA, "n11_zaman_sonuc.json"))
