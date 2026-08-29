"""s2 -- 9 hakli 3 gunluk gonderim plani icin karar analizi. YEREL, gonderim YOK.

Cebir (m6 tabani, log1p uzayi, d_j = log1p(aday)-log1p(m6)):
    dMSE(c) = -2 c'L + c'G c ,  L_j = -<e,d_j>/N ,  G_ij = <d_i,d_j>/N
    tek yon optimumu: c*=L/Q, kazanc = L^2/Q = r^2 ,  r_j = L_j/sqrt(Q_j)
    ortak optimum : c*=G^-1 L, kazanc = L'G^-1 L = r' C^-1 r   (ORTUSME ICERIDE)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
BURA = Path(__file__).resolve().parent
g = json.loads((BURA / "s1_gram.json").read_text(encoding="utf-8"))
ADLAR = g["adlar"]
IX = {a: i for i, a in enumerate(ADLAR)}
C = np.array(g["C"])
Q = np.array([g["Q"][a] for a in ADLAR])
SQ = np.sqrt(Q)
KURT = np.array([g["kurtoz"][a] for a in ADLAR])

M0 = 1.005688  # m6 MSE
HEDEF = 1.000820  # 2. siranin MSE'si
GEREK = M0 - HEDEF  # 0.004868
R_G7 = 0.002728 / np.sqrt(0.002494)  # olculmus g7 kalitesi = 0.054626
SIG_L = 9.5e-5  # skor gurultusu, L biriminde
KOS_M4 = {
    "g7": 0.0,
    "y46": -0.115,
    "y45": 0.005,
    "z1": 0.186,
    "z2": 0.161,
    "z3": 0.564,
    "r1": 0.0,
    "r3": 0.0,
    "y40": 0.218,
    "y42": 0.504,
}
ADAYLAR = [a for a in ADLAR if a != "g7"]

# ---------------------------------------------------------------- ONSEL
# Olculmus kaliteler: v101 0.1243, m4 0.0641, p51 0.0493, g7 0.0546.
# Taban iyilestikce azalan verim -> yeni yonlerin medyani olculmuslerin
# ALTINDA. Log-normal: medyan 0.045 * dikligi * kurtoz cezasi.
#   diklik  = sqrt(1-kos(m4)^2)   (m4 ekseni zaten m6 icinde -> o bilesen tuketilmis)
#   kurtoz  = (6/k)^0.25          (sivri yon = birkac satira dayali, kirilgan)
#   sigma_j = 0.55*(k/6)^0.15     (sivri yon = daha belirsiz)
MED = {}
SIG = {}
for a in ADAYLAR:
    k = KURT[IX[a]]
    MED[a] = 0.045 * np.sqrt(1 - KOS_M4[a] ** 2) * (6.0 / k) ** 0.25
    SIG[a] = 0.55 * (k / 6.0) ** 0.15
# Fizibilite tavani: ulasilabilir toplam hizalanma normu R (azalan verim).
R_TAVAN_MED, R_TAVAN_SIG, R_ALT = 0.105, 0.30, 0.056  # R >= r_g7 (g7 olculdu)
P_ARTI = 0.85  # bir adayin GERCEKTEN iyilestirme yonu olma olasiligi

RNG = np.random.default_rng(20260829)
NMC = 20000


def cek(n, p_arti=P_ARTI, tavan_med=R_TAVAN_MED):
    R = np.empty((n, len(ADLAR)))
    R[:, IX["g7"]] = R_G7
    for a in ADAYLAR:
        m = MED[a] * np.exp(SIG[a] * RNG.standard_normal(n))
        isa = np.where(RNG.random(n) < p_arti, 1.0, -1.0)
        R[:, IX[a]] = m * isa
    Rcap = np.maximum(R_ALT, tavan_med * np.exp(R_TAVAN_SIG * RNG.standard_normal(n)))
    return R, Rcap**2


RMAT, TAVAN = cek(NMC)
LMAT = RMAT * SQ
GURULTU = SIG_L * RNG.standard_normal((NMC, len(ADLAR)))


def kazanc(idx, cmat, L):
    ix = np.array(idx)
    Gs = C[np.ix_(ix, ix)] * np.outer(SQ[ix], SQ[ix])
    k = 2 * (cmat * L[:, ix]).sum(1) - np.einsum("ni,ij,nj->n", cmat, Gs, cmat)
    return np.minimum(k, TAVAN)  # fizibilite: kazanc <= ||u||^2


def coz(idx, Lhat, ridge=1e-9):
    ix = np.array(idx)
    Gs = C[np.ix_(ix, ix)] * np.outer(SQ[ix], SQ[ix])
    return np.linalg.solve(Gs + ridge * np.eye(len(ix)), Lhat[:, ix].T).T


L_ONSEL = (
    np.array(
        [R_G7 if a == "g7" else (2 * P_ARTI - 1) * MED[a] * np.exp(SIG[a] ** 2 / 2) for a in ADLAR]
    )
    * SQ
)


def prob_gonderim(j):
    """SONDA: m6 + g7opt + c0*d_j. c0 onselden. Hem skor hem olcum."""
    idx = [IX["g7"], IX[j]]
    c0 = coz(idx, L_ONSEL[None, :].repeat(1, 0))[0]
    return idx, np.repeat(c0[None, :], NMC, 0)


def birlesim(olculenler):
    idx = [IX["g7"]] + [IX[a] for a in olculenler]
    Lhat = LMAT + GURULTU
    Lhat[:, IX["g7"]] = LMAT[:, IX["g7"]]  # g7 zaten bilinen
    return idx, coz(idx, Lhat)


def strateji(gun_planlari):
    """gun_planlari: [[eylem,...],...]; eylem = ('sonda',ad) | ('birlesim',)"""
    olculen = []
    en_iyi = np.zeros(NMC)
    gunluk = []
    for gun in gun_planlari:
        yeni = []
        for e in gun:
            if e[0] == "sonda":
                idx, cm = prob_gonderim(e[1])
                yeni.append(e[1])
            else:
                idx, cm = birlesim(olculen)
            en_iyi = np.maximum(en_iyi, kazanc(idx, cm, LMAT))
        olculen += yeni
        gunluk.append((float((en_iyi >= GEREK).mean()), float(np.median(en_iyi))))
    return gunluk


# ------------------------------------------------------- 1. GUN IKILI TABLOSU
print(f"GEREKEN dMSE = {GEREK:.6f} | g7 tek basina = {R_G7**2:.6f} | kalan = {GEREK - R_G7**2:.6f}")
print("\nONSEL (log-normal, medyan/sigma) ve tek-yon beklenen kazanc r^2:")
for a in ADAYLAR:
    print(
        f"  {a:4s} medyan r={MED[a]:.4f} sigma={SIG[a]:.3f}  E[r^2]={MED[a] ** 2 * np.exp(2 * SIG[a] ** 2):.5f}"
        f"  P(r>0.07)={1 - 0.5 * (1 + math.erf((np.log(0.07 / MED[a])) / (SIG[a] * np.sqrt(2)))):.3f}"
    )

sonuc = {}
for i, A in enumerate(ADAYLAR):
    for B in ADAYLAR[i + 1 :]:
        gp = [[("sonda", A), ("sonda", B), ("birlesim",)]]
        sonuc[(A, B)] = strateji(gp)[0]
sirali = sorted(sonuc.items(), key=lambda kv: -kv[1][0])
print("\n=== 1. GUN IKILI TABLOSU (HAK1=sonda A, HAK2=sonda B, HAK3=ortak optimum) ===")
print(f"{'ikili':<12}{'P(2.sira)':>10}{'medyan dMSE':>14}{'medyan LB':>12}")
for (A, B), (p, med) in sirali:
    print(f"{A + '+' + B:<12}{p:>10.3f}{-med:>14.6f}{np.sqrt(M0 - med):>12.6f}")

json.dump(
    {
        "gerek": GEREK,
        "g7": R_G7**2,
        "ikili": {f"{A}+{B}": {"P2": p, "medyan_dMSE": -m} for (A, B), (p, m) in sirali},
    },
    open(BURA / "s2_ikili.json", "w"),
    indent=1,
)

# ------------------------------------------------------------ OLCUM GURULTUSU
print("\n=== OLCUM GURULTUSU (sigma_L = 9.5e-5) ===")
print(
    f"{'yon':<5}{'Q':>9}{'sqrtQ':>8}{'sigma_r':>9}{'sigma_r/r_med':>14}{'plug-in kayip':>15}{'kayip/gerek':>12}"
)
for a in ADLAR:
    q = Q[IX[a]]
    sr = SIG_L / np.sqrt(q)
    rm = R_G7 if a == "g7" else MED[a]
    kayip = SIG_L**2 / q  # tek yon icin E[eps'G^-1 eps] = sigma^2/Q
    print(
        f"{a:<5}{q:>9.5f}{np.sqrt(q):>8.4f}{sr:>9.5f}{sr / rm:>13.1%}{kayip:>15.3e}{kayip / GEREK:>11.2%}"
    )
ix3 = [IX["g7"], IX["r1"], IX["r3"]]
Gs = C[np.ix_(ix3, ix3)] * np.outer(SQ[ix3], SQ[ix3])
print(
    f"  g7+r1+r3 birlesik plug-in kayip = {SIG_L**2 * np.trace(np.linalg.inv(Gs)):.3e}"
    f"  ({SIG_L**2 * np.trace(np.linalg.inv(Gs)) / GEREK:.2%} of gerek)"
)
ix3b = [IX["g7"], IX["y46"], IX["z2"]]
Gb = C[np.ix_(ix3b, ix3b)] * np.outer(SQ[ix3b], SQ[ix3b])
print(
    f"  g7+y46+z2 birlesik plug-in kayip = {SIG_L**2 * np.trace(np.linalg.inv(Gb)):.3e}"
    f"  ({SIG_L**2 * np.trace(np.linalg.inv(Gb)) / GEREK:.2%} of gerek)"
)

# --------------------------------------------------------------- 3 GUNLUK SIRA
SIRA = ["y40", "z2", "y46", "z1", "r3", "y45", "r1", "y42", "z3"]
S = {
    "A_2olcum+1optimum/gun": [
        [("sonda", "y40"), ("sonda", "z2"), ("birlesim",)],
        [("sonda", "y46"), ("sonda", "z1"), ("birlesim",)],
        [("sonda", "r3"), ("sonda", "y45"), ("birlesim",)],
    ],
    "B_3olcum_ilkgun": [
        [("sonda", "y40"), ("sonda", "z2"), ("sonda", "y46")],
        [("birlesim",), ("sonda", "z1"), ("sonda", "r3")],
        [("birlesim",), ("sonda", "y45"), ("sonda", "r1")],
    ],
    "C_1olcum+2optimum": [
        [("sonda", "y40"), ("sonda", "z2"), ("birlesim",)],
        [("sonda", "y46"), ("birlesim",), ("birlesim",)],
        [("sonda", "z1"), ("sonda", "r3"), ("birlesim",)],
    ],
    "D_hepsi_olcum_sonda": [
        [("sonda", "y40"), ("sonda", "z2"), ("sonda", "y46")],
        [("sonda", "z1"), ("sonda", "r3"), ("sonda", "y45")],
        [("birlesim",), ("birlesim",), ("birlesim",)],
    ],
}
print("\n=== UC GUNLUK STRATEJI (kumulatif P(2.sira) / medyan dMSE) ===")
strat = {}
for ad, gp in S.items():
    g_ = strateji(gp)
    strat[ad] = g_
    print(
        f"{ad:<26}"
        + "  ".join(f"gun{i + 1}: P={p:.3f} med={-m:.5f}" for i, (p, m) in enumerate(g_))
    )

# ------------------------------------------------------------------ DUYARLILIK
print("\n=== DUYARLILIK (ilk gun, en iyi 3 ikili) ===")
TEST = [("z2", "y40"), ("y46", "y40"), ("y46", "z2"), ("z1", "z2"), ("r1", "r3")]
print(f"{'senaryo':<34}" + "".join(f"{A + '+' + B:>12}" for A, B in TEST))
for ad, r0, pa, tv in [
    ("temel (r0=.045,p+=.85,tavan.105)", 0.045, 0.85, 0.105),
    ("kotumser r0=.030", 0.030, 0.85, 0.105),
    ("isaret belirsiz p+=0.70", 0.045, 0.70, 0.105),
    ("isaret kesin p+=1.00", 0.045, 1.00, 0.105),
    ("dar tavan 0.090", 0.045, 0.85, 0.090),
    ("genis tavan 0.140", 0.045, 0.85, 0.140),
    ("kotumser+isaret belirsiz", 0.030, 0.70, 0.105),
]:
    for a in ADAYLAR:
        k = KURT[IX[a]]
        MED[a] = r0 * np.sqrt(1 - KOS_M4[a] ** 2) * (6.0 / k) ** 0.25
    RMAT, TAVAN = cek(NMC, p_arti=pa, tavan_med=tv)
    LMAT = RMAT * SQ
    L_ONSEL = (
        np.array(
            [R_G7 if a == "g7" else (2 * pa - 1) * MED[a] * np.exp(SIG[a] ** 2 / 2) for a in ADLAR]
        )
        * SQ
    )
    globals().update(RMAT=RMAT, TAVAN=TAVAN, LMAT=LMAT, L_ONSEL=L_ONSEL)
    row = [strateji([[("sonda", A), ("sonda", B), ("birlesim",)]])[0][0] for A, B in TEST]
    print(f"{ad:<34}" + "".join(f"{v:>12.3f}" for v in row))

json.dump(
    {
        "gerek": GEREK,
        "g7_katki": R_G7**2,
        "kalan": GEREK - R_G7**2,
        "en_iyi_ikili": [f"{A}+{B}" for (A, B), _ in sirali[:5]],
        "ikili_P2": {f"{A}+{B}": p for (A, B), (p, m) in sirali},
        "ikili_medyan_dMSE": {f"{A}+{B}": -m for (A, B), (p, m) in sirali},
        "strateji": {
            k: [{"gun": i + 1, "P2": p, "medyan_dMSE": -m} for i, (p, m) in enumerate(v)]
            for k, v in strat.items()
        },
        "onsel": {
            "tip": "log-normal",
            "r0": 0.045,
            "medyan": MED,
            "sigma": SIG,
            "P_isaret_arti": P_ARTI,
            "R_tavan_medyan": R_TAVAN_MED,
        },
        "gurultu": {
            "sigma_L": SIG_L,
            "plugin_kayip": {a: SIG_L**2 / float(Q[IX[a]]) for a in ADLAR},
        },
        "kosul_sayisi_tam10": g["kosul_tam"],
        "kosinus_matrisi": {a: {b: round(float(C[IX[a], IX[b]]), 4) for b in ADLAR} for a in ADLAR},
        "Q": g["Q"],
        "kurtoz": g["kurtoz"],
    },
    open(BURA / "s1_karar.json", "w", encoding="utf-8"),
    indent=1,
    ensure_ascii=False,
)
print("\n-> s1_karar.json yazildi")
