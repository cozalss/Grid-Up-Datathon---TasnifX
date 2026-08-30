"""d_pca GERCEK MI, SANS MI?  -- AY KATLARI + KURAL VARYANTLARI

n15/n15b: 6 adaydan yalniz d_pca iki yonde de a_fit'i gecti
(ORT rho^2 +0.0322, aile duzeltmeli p=0.0027). AMA:
  - kazanc neredeyse tamamen TEK yonden geliyor (+0.0007 / +0.0670)
  - iki yonun urettigi PCA bolmeleri AYNI DEGIL (kural KARARSIZ)

BU BETIK iki bagimsiz saglama yapar:
 1) AY KATLARI: yaz25 dort aya bolunur (Nis,May,Haz,Tem). Her ay sirayla
    OLCUM kati olur, diger UC ay FIT'tir. Dort ek olcum yonu.
 2) KURAL VARYANTLARI: d_pca'nin kucuk degisiklikleri. Kural gercekten
    bilgi tasiyorsa varyantlar da kazanmalidir; yalniz bir varyant
    kazaniyorsa o SANSTIR.
      d_pca      : |R| ozvektorleri, argmax|yuk|      (n15'teki)
      d_pca_sgn  : R (isaretli) ozvektorleri
      d_pca_kat  : |KATS| ile agirliklandirilmis R
      d_pca_5    : ilk 5 ozvektor, 4'e kirpma yerine 5->4 birlestirme
 NOT: PCA kurali r_gercek'e HIC BAKMAZ (yalniz eksenlerin kendi korelasyon
 yapisina bakar), bu yuzden sonuca uydurma riski YOKTUR; tek risk benim
 6 aday denemis olmamdir -- o da aile duzeltmesiyle ele alindi.
"""

import json
import os

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
NRAST = 4000
BLOK_ALT = 0.06
DEMET_HEDEF = 4

with open(os.path.join(SCR, "n09_secim.json"), encoding="utf-8") as fh:
    SEC = json.load(fh)
KUL = SEC["kul"]
RHOS = np.array(SEC["rho_s"])
KATS_TAM = np.array(SEC["KATS"])

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
            k1, k2 = ad[2:-1].split("]x[", 1)
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


ff, pb, rr_t, _ = blok_kur("yaz25")
kurY = Kur(ff, pb)
ay = pd.to_datetime(ff.tarih).dt.month.to_numpy()
XY = np.zeros((K, len(rr_t)), dtype=np.float32)
XOK = np.zeros(K, dtype=bool)
for i, ad in enumerate(KUL[:K]):
    _, xb = kurY(ad)
    if xb is None or not np.isfinite(xb).all():
        continue
    XY[i] = xb.astype(np.float32)
    XOK[i] = True
print(f"{int(XOK.sum())}/{K} eksen kuruldu")
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
SG = ff.soguk_mu.values.astype(np.float64)


def _birlestir_4(HAM, kk):
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
    return [(k, HAM[k]) for k in sorted(AG, key=lambda k: -AG[k])]


def _et(cur, n):
    et = np.full(n, -1, dtype=int)
    for q, (_nm, m) in enumerate(cur):
        et[m] = q
    assert (et >= 0).all()
    return et


def deger(et, kk, ri):
    s = 0.0
    for q in np.unique(et):
        m = et == q
        c = kk[m]
        nn = float((c * c).sum())
        if nn > 0:
            s += float((c @ ri[m]) ** 2) / nn
    return s


def pca_et(Rm, kk, npc=4):
    ev, EV = np.linalg.eigh(Rm)
    V = EV[:, np.argsort(-ev)[:npc]]
    return np.abs(V).argmax(axis=1)


def hazirla(m):
    sg = SG[m]
    w = np.where(sg == 1, HEDEF_SOGUK / sg.mean(), (1 - HEDEF_SOGUK) / (1 - sg.mean()))
    w = w / w.mean()
    r = rr_t[m]
    return w, r, float((w * r * r).mean())


def gs(Xm, w):
    n = Xm.shape[1]
    U = np.zeros((K, n))
    canli = np.zeros(K, dtype=bool)
    for i in range(K):
        if not XOK[i]:
            continue
        u = Xm[i] - Xm[i].mean()
        s = np.sqrt(float((w * u * u).mean()))
        if s < 1e-12:
            continue
        u = u / s
        for j in range(i):
            if canli[j]:
                u -= float((w * u * U[j]).mean()) * U[j]
        n1 = np.sqrt(float((w * u * u).mean()))
        if n1 < NB_ALT:
            continue
        U[i] = u / n1
        canli[i] = True
    return U, canli


AYLAR = {"Nis": 4, "May": 5, "Haz": 6, "Tem": 7}
KATLAR = [
    (f"3ay->{nm}", np.isin(ay, [v for k, v in AYLAR.items() if k != nm]), ay == AYLAR[nm])
    for nm in AYLAR
]
KATLAR += [
    ("NisMay->HazTem", np.isin(ay, [4, 5]), np.isin(ay, [6, 7])),
    ("HazTem->NisMay", np.isin(ay, [6, 7]), np.isin(ay, [4, 5])),
]

ADLAR = ["a_fit", "b_rhos", "d_pca", "d_pca_sgn", "d_pca_kat", "d_pca_5"]
TUM = {}
HAM_KAT = {}
for nm, mf, mo in KATLAR:
    wf, rf, m0f = hazirla(mf)
    wo, ro, m0o = hazirla(mo)
    nf, no = int(mf.sum()), int(mo.sum())
    Xf = XY[:, mf].astype(np.float64)
    korf = (Xf @ (wf * rf)) / (nf * np.sqrt(m0f))
    ISR = np.sign(korf)
    ISR[ISR == 0] = 1.0
    KATS_F = ISR * TAVAN * np.abs(RHOS[:K])
    RHOCV_F = CARPAN * korf
    Xc = Xf - (wf * Xf).mean(axis=1, keepdims=True)
    sd = np.sqrt((wf * Xc * Xc).mean(axis=1))
    sd[sd < 1e-12] = 1.0
    Xn = Xc / sd[:, None]
    R = (Xn * wf) @ Xn.T / nf
    del Xf, Xc, Xn

    Uo, canli = gs(XY[:, mo].astype(np.float64), wo)
    idx = np.flatnonzero(canli)
    nk = len(idx)
    Ui = Uo[idx]
    ri = (Ui @ (wo * ro)) / (no * np.sqrt(m0o))
    tavan2 = float((ri**2).sum())
    kk = KATS_F[idx]
    hv = HV[idx]
    R_o = R[np.ix_(idx, idx)]
    del Uo, Ui

    C = {}
    oran = np.abs(RHOCV_F[idx]) / np.maximum(np.abs(kk), 1e-12)
    yuk = np.zeros(nk, dtype=bool)
    for msk in (hv, ~hv):
        if msk.sum() >= 2:
            yuk |= msk & (oran > float(np.median(oran[msk])))
    C["a_fit"] = _et(
        _birlestir_4(
            {"h/y": hv & yuk, "h/d": hv & ~yuk, "y/y": (~hv) & yuk, "y/d": (~hv) & ~yuk}, kk
        ),
        nk,
    )
    rs = np.abs(RHOS[:K][idx])
    yk2 = np.zeros(nk, dtype=bool)
    for msk in (hv, ~hv):
        if msk.sum() >= 2:
            yk2 |= msk & (rs > float(np.median(rs[msk])))
    C["b_rhos"] = _et(
        _birlestir_4(
            {"h/y": hv & yk2, "h/d": hv & ~yk2, "y/y": (~hv) & yk2, "y/d": (~hv) & ~yk2}, kk
        ),
        nk,
    )
    for adv, Rm, npc in (
        ("d_pca", np.abs(R_o), 4),
        ("d_pca_sgn", R_o, 4),
        ("d_pca_kat", np.abs(R_o) * np.outer(np.abs(kk), np.abs(kk)), 4),
        ("d_pca_5", np.abs(R_o), 5),
    ):
        et0 = pca_et(Rm, kk, npc)
        C[adv] = _et(
            _birlestir_4({f"p{v}": np.equal(et0, v) for v in sorted(set(et0.tolist()))}, kk), nk
        )

    D = {a: deger(C[a], kk, ri) for a in ADLAR}
    rng = np.random.default_rng(5)
    RA = []
    for _ in range(NRAST):
        while True:
            t = rng.integers(0, 4, size=nk)
            if len(np.unique(t)) == 4:
                break
        RA.append(deger(t, kk, ri))
    RA = np.array(RA)
    TUM[nm] = dict(
        n=no,
        nk=nk,
        tavan2=tavan2,
        deger=D,
        rast=dict(
            ort=float(RA.mean()),
            q50=float(np.median(RA)),
            q90=float(np.quantile(RA, 0.9)),
            q99=float(np.quantile(RA, 0.99)),
        ),
        yuzdelik={a: float((D[a] > RA).mean()) for a in ADLAR},
        pca_et=C["d_pca"].tolist(),
        idx=idx.tolist(),
    )
    HAM_KAT[nm] = dict(kk=kk, ri=ri, idx=idx, C={a: C[a] for a in ADLAR})
    print(
        f"\n--- {nm}: olcum {no:,} satir, {nk} eksen, TAVAN rho={np.sqrt(tavan2):.4f}, "
        f"rastgele ort rho^2={RA.mean():.6f}"
    )
    for a in ADLAR:
        print(
            f"    {a:>10s} rho^2={D[a]:.6f} rho={np.sqrt(D[a]):.4f} "
            f"fark={D[a] - D['a_fit']:+.6f}  rastgele yuzdeligi %{100 * TUM[nm]['yuzdelik'][a]:.1f}"
        )

print("\n\n############ OZET: her katta a_fit'e gore fark (rho^2) ############")
print(f"{'kat':>16s} " + " ".join(f"{a:>11s}" for a in ADLAR if a != "a_fit"))
for nm in TUM:
    print(
        f"{nm:>16s} "
        + " ".join(
            f"{TUM[nm]['deger'][a] - TUM[nm]['deger']['a_fit']:+11.6f}"
            for a in ADLAR
            if a != "a_fit"
        )
    )
print(
    f"{'POZITIF KAT':>16s} "
    + " ".join(
        f"{sum(1 for nm in TUM if TUM[nm]['deger'][a] > TUM[nm]['deger']['a_fit']):>7d}/{len(TUM):<3d}"
        for a in ADLAR
        if a != "a_fit"
    )
)
# yalniz KESISMEYEN dort ay kati (bagimsiz olcum kumeleri) ile agirlikli ortalama
BAG = [nm for nm in TUM if nm.startswith("3ay->")]
TN = sum(TUM[nm]["n"] for nm in BAG)
print(f"\n4 AY KATI (kesismeyen olcum kumeleri, toplam {TN:,} satir) agirlikli ortalama:")
for a in ADLAR:
    v = sum(TUM[nm]["n"] * TUM[nm]["deger"][a] for nm in BAG) / TN
    v0 = sum(TUM[nm]["n"] * TUM[nm]["deger"]["a_fit"] for nm in BAG) / TN
    yz = sum(TUM[nm]["n"] * TUM[nm]["yuzdelik"][a] for nm in BAG) / TN
    print(
        f"  {a:>10s} rho^2={v:.6f} rho={np.sqrt(v):.4f} fark={v - v0:+.6f} "
        f"ort.rastgele-yuzdelik %{100 * yz:.1f}"
    )

# ---- HAVUZLANMIS SANS TESTI: TEK bir sabit bolme, DORT ayrik ay katinda
# olculur. Aday kurallarinin havuzlanmis kazanci, rastgele bir SABIT
# bolmenin havuzlanmis kazancindan ayirt edilebiliyor mu?
print("\n\n############ HAVUZLANMIS SANS TESTI (4 ayrik ay kati) ############")
rngH = np.random.default_rng(909)
NH = 20000
PH = np.zeros(NH)
for t in range(NH):
    while True:
        et_tam = rngH.integers(0, 4, size=K)
        if len(np.unique(et_tam)) == 4:
            break
    PH[t] = (
        sum(
            TUM[nm]["n"] * deger(et_tam[HAM_KAT[nm]["idx"]], HAM_KAT[nm]["kk"], HAM_KAT[nm]["ri"])
            for nm in BAG
        )
        / TN
    )
v0 = sum(TUM[nm]["n"] * TUM[nm]["deger"]["a_fit"] for nm in BAG) / TN
print(
    f"  rastgele SABIT bolme havuzlanmis rho^2: ort {PH.mean():.6f} "
    f"medyan {np.median(PH):.6f} %90 {np.quantile(PH, 0.9):.6f} "
    f"%99 {np.quantile(PH, 0.99):.6f} maks {PH.max():.6f}"
)
print(f"  a_fit = {v0:.6f}  -> rastgelenin %{100 * (v0 > PH).mean():.1f} yuzdeligi")
NAD = len(ADLAR) - 1
HAVUZ = {}
for a in ADLAR:
    v = sum(TUM[nm]["n"] * TUM[nm]["deger"][a] for nm in BAG) / TN
    p = float((v <= PH).mean())
    HAVUZ[a] = dict(rho2=v, p_ham=p, p_aile=1.0 - (1.0 - p) ** NAD, yuzdelik=float((v > PH).mean()))
    print(
        f"  {a:>10s} rho^2={v:.6f} fark={v - v0:+.6f} p_ham={p:.4f} "
        f"p_aile({NAD})={HAVUZ[a]['p_aile']:.4f}"
    )
TUM["_havuz"] = HAVUZ
TUM["_havuz_rast"] = dict(
    ort=float(PH.mean()),
    q50=float(np.median(PH)),
    q90=float(np.quantile(PH, 0.9)),
    q99=float(np.quantile(PH, 0.99)),
    maks=float(PH.max()),
    n=NH,
)

with open(os.path.join(M29, "n15c_ay_katlari.json"), "w", encoding="utf-8") as fh:
    json.dump(TUM, fh, indent=1, ensure_ascii=False)
print("\nyazildi: n15c_ay_katlari.json")
