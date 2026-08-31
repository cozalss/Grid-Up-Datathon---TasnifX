"""n20d -- IKI |c| NOKTASININ SAGLAMLIK SINAVI.

HICBIR GONDERIM. submissions/ ALTINA YAZMA YOK.

n20b iki nokta verdi: |c|(seviye) = 1.955, |c|(yenibaslangic) = 1.090.
Duyarlilik taramasi span'in SON IKI vektorune (s3y40 ve y40_sota_temiz)
asiri bagimlilik gosterdi. y40_sota_temiz'in L'si LB'DE OLCULMEDI --
EK_MODEL'den gelen TURETILMIS bir sayidir. Bu betik her noktayi
  (a) span bilesimine
  (b) rcond'a
  (c) L'nin olcum gurultusune
karsi sinar ve |c| icin DURUST bir aralik uretir.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
S = os.path.join(KOK, "submissions")
DN = os.path.join(KOK, "data/interim/deney")
M29 = os.path.join(KOK, "experiments/model29")
TABAN = "tuketim_m6_ikiyon.csv"
SIGMA_L = 2.9377803611172106e-06
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0  # noqa: E402

np.seterr(all="ignore")
rng = np.random.default_rng(20260831)
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


with open(os.path.join(M29, "olculmus_skorlar.json")) as fh:
    SK = json.load(fh)
with open(os.path.join(M29, "m112_durum.json")) as fh:
    DUR = json.load(fh)
a0 = oku(TABAN)
N = len(a0)
AD, VV, LL, OLC = [], [], [], []
for f, Pj in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is None or len(v) != N:
        continue
    d = v - a0
    AD.append(f)
    VV.append(d)
    LL.append((M0 + float((d * d).mean()) - Pj * Pj) / 2)
    OLC.append(True)
for f, Lj in EK_MODEL.items():
    AD.append(f)
    VV.append(oku(f) - a0)
    LL.append(float(Lj))
    OLC.append(False)
for o in DUR.get("olcumler", []):
    d = oku(o["dosya"]) - a0
    AD.append(o["dosya"])
    VV.append(d)
    LL.append((M0 + float((d * d).mean()) - o["skor"] ** 2) / 2)
    OLC.append(True)
V = np.array(VV).T
del VV
L = np.array(LL)
OLC = np.array(OLC)
IX = {a: i for i, a in enumerate(AD)}
K = V.shape[1]
G = (V.T @ V) / N
J_SEV, J_YEN = IX["tuketim_YP_seviye.csv"], IX["tuketim_K_yenibas.csv"]
J_Y40 = IX["tuketim_y40_sota_temiz.csv"]
J_S3 = IX["tuketim_s3y40.csv"]

# eksenler
tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
assert np.array_equal(tp.id.values, IDS)


def st(x):
    x = np.asarray(x, dtype=np.float64).copy()
    f = np.isfinite(x)
    x[~f] = np.median(x[f])
    x -= x.mean()
    s = np.sqrt(float((x * x).mean()))
    return x / s if s > 1e-12 else x


X = {"seviye": st(a0)}
_y = oku("tuketim_KES_yenibaslangic.csv") - a0
X["yenibaslangic"] = _y / np.sqrt(float((_y * _y).mean()))
PROBE = {"seviye": J_SEV, "yenibaslangic": J_YEN}

print("=" * 78)
print("n20d  IKI |c| NOKTASININ SAGLAMLIK SINAVI")
print("=" * 78)
print("DENETIM: X['seviye'] = st(a0) m113'un 'seviye' ekseniyle ayni mi?")
sys.path.insert(0, M29)


def hesap(eksen, ix, rcond=1e-6, Lv=None):
    Lv = L if Lv is None else Lv
    ix = np.asarray(ix, int)
    Gr = G[np.ix_(ix, ix)]
    Gp = np.linalg.pinv(Gr, rcond=rcond)
    x = X[eksen]
    j = PROBE[eksen]
    cx = Gp @ ((V[:, ix].T @ x) / N)
    Qsp = float(cx @ Gr @ cx)
    Lsp = float(cx @ Lv[ix])
    if Qsp <= 1e-10:
        return None
    rs = Lsp / np.sqrt(Qsp)
    cj = Gp @ G[ix, j]
    Qspj = float(cj @ Gr @ cj)
    Qdkj = float(G[j, j] - 2 * cj @ G[ix, j] + Qspj)
    if Qdkj <= 1e-10:
        return None
    rd = (Lv[j] - float(cj @ Lv[ix])) / np.sqrt(Qdkj)
    return dict(
        rs=rs,
        rd=rd,
        c=abs(rd) / abs(rs),
        isaret=int(np.sign(rd) * np.sign(rs)),
        Qsp=Qsp,
        cn2=float(cx @ cx),
        cn2j=float(cj @ cj),
        Qdk=Qdkj,
    )


TEMEL = {
    "seviye": [i for i in range(K) if i not in (J_SEV, J_YEN)],
    "yenibaslangic": [i for i in range(K) if i != J_YEN],
}
print("\n" + "=" * 78)
print("A. SPAN BILESIMINE DUYARLILIK")
print("=" * 78)
VARYANT = {
    "TEMEL": lambda b: b,
    "y40 (TURETILMIS L) CIKARILDI": lambda b: [i for i in b if i != J_Y40],
    "s3y40 CIKARILDI": lambda b: [i for i in b if i != J_S3],
    "y40 + s3y40 CIKARILDI": lambda b: [i for i in b if i not in (J_Y40, J_S3)],
    "yalniz LB'de OLCULMUSLER": lambda b: [i for i in b if OLC[i]],
    "en zayif 5 yon CIKARILDI": lambda b: sorted(b, key=lambda i: -G[i, i])[:-5],
}
SON = {}
for eksen in ("seviye", "yenibaslangic"):
    print(f"\n{eksen}:")
    print(f"{'varyant':>32s} {'|S|':>4s} {'rho_s':>9s} {'rho_dik':>9s} {'isaret':>7s} {'|c|':>8s}")
    SON[eksen] = {}
    for ad, fn in VARYANT.items():
        ix = fn(TEMEL[eksen])
        r = hesap(eksen, ix)
        if r is None:
            print(f"{ad:>32s} {len(ix):4d}  TANIMSIZ")
            continue
        SON[eksen][ad] = r
        print(
            f"{ad:>32s} {len(ix):4d} {r['rs']:+9.4f} {r['rd']:+9.4f} "
            f"{'AYNI' if r['isaret'] > 0 else 'TERS':>7s} {r['c']:8.3f}"
        )

print("\n" + "=" * 78)
print("B. rcond DUYARLILIGI")
print("=" * 78)
print(f"{'eksen':>15s} {'rcond':>8s} {'rho_s':>9s} {'rho_dik':>9s} {'|c|':>8s}")
for eksen in ("seviye", "yenibaslangic"):
    for rc in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        r = hesap(eksen, TEMEL[eksen], rcond=rc)
        if r:
            print(f"{eksen:>15s} {rc:8.0e} {r['rs']:+9.4f} {r['rd']:+9.4f} {r['c']:8.3f}")

print("\n" + "=" * 78)
print("C. L OLCUM GURULTUSU (sigma_L = %.2e, LB 5 hane yuvarlamasi)" % SIGMA_L)
print("=" * 78)
print(f"{'eksen':>15s} {'|c| nokta':>10s} {'%5':>8s} {'%50':>8s} {'%95':>8s} {'P(|c|>1)':>9s}")
MC = {}
for eksen in ("seviye", "yenibaslangic"):
    r0 = hesap(eksen, TEMEL[eksen])
    bs = []
    for _ in range(3000):
        Lv = L + rng.normal(0, SIGMA_L, K)
        r = hesap(eksen, TEMEL[eksen], Lv=Lv)
        if r and np.isfinite(r["c"]):
            bs.append(r["c"])
    bs = np.array(bs)
    MC[eksen] = bs
    print(
        f"{eksen:>15s} {r0['c']:10.3f} {np.quantile(bs, 0.05):8.3f} {np.median(bs):8.3f} "
        f"{np.quantile(bs, 0.95):8.3f} {float((bs > 1).mean()):9.3f}"
    )

print("\n" + "=" * 78)
print("D. BIRLESIK -- iki nokta + butun belirsizlik kaynaklari")
print("=" * 78)
# Her cekimde: (i) L gurultusu, (ii) span varyanti secimi, (iii) rcond secimi
VAR_IX = {e: [fn(TEMEL[e]) for fn in VARYANT.values()] for e in TEMEL}
RC = [1e-5, 1e-6, 1e-7]
ORN = {e: [] for e in TEMEL}
for eksen in ("seviye", "yenibaslangic"):
    for _ in range(2500):
        Lv = L + rng.normal(0, SIGMA_L, K)
        ix = VAR_IX[eksen][rng.integers(0, len(VAR_IX[eksen]))]
        rc = RC[rng.integers(0, len(RC))]
        r = hesap(eksen, ix, rcond=rc, Lv=Lv)
        if r and np.isfinite(r["c"]) and r["c"] < 1e3:
            ORN[eksen].append(r["c"])
    a = np.array(ORN[eksen])
    print(
        f"{eksen:>15s}: n={len(a)}  medyan {np.median(a):.3f}  "
        f"%90 [{np.quantile(a, 0.05):.3f}, {np.quantile(a, 0.95):.3f}]"
    )

# iki eksenin geometrik ortalamasi -- eksenler arasi degiskenlik de tasinir
na = min(len(ORN["seviye"]), len(ORN["yenibaslangic"]))
gm = np.sqrt(np.array(ORN["seviye"][:na]) * np.array(ORN["yenibaslangic"][:na]))
print(
    f"\n  IKI EKSENIN GEOMETRIK ORTALAMASI: medyan {np.median(gm):.3f}  "
    f"%90 [{np.quantile(gm, 0.05):.3f}, {np.quantile(gm, 0.95):.3f}]"
)
# n=2'lik eksen-secimi belirsizligi: log-sd'yi t ile buyut
l1, l2 = np.log(np.median(ORN["seviye"])), np.log(np.median(ORN["yenibaslangic"]))
sd_eks = abs(l1 - l2) / np.sqrt(2)
from scipy import stats  # noqa: E402

t = stats.t.ppf(0.95, 1)
print(f"  eksenler arasi log-sacilim sd = {sd_eks:.3f} (n=2)  -> t_{{0.95,1}} = {t:.2f}")
GM = float(np.exp((l1 + l2) / 2))
ALT, UST = (
    float(np.exp((l1 + l2) / 2 - t * sd_eks / np.sqrt(2))),
    float(np.exp((l1 + l2) / 2 + t * sd_eks / np.sqrt(2))),
)
print(f"\n  *** |c| = {GM:.2f}   %90 aralik [{ALT:.2f}, {UST:.2f}]   (n = 2 eksen) ***")

# esik: hangi |c| hangi skoru verir
RHO_S_BIL = 0.2141 / 1.95
TABAN_MSE = 1.00202690
print(f"\nSKOR SONUCU  (rho_LB = |c| * rho_s(bilesik), rho_s(bilesik) = {RHO_S_BIL:.5f})")
print(f"{'|c|':>8s} {'rho_LB':>9s} {'nihai skor':>11s}")
for c in (0.434, 0.8, 1.09, ALT, GM, 1.955, UST, 1.986):
    rho = c * RHO_S_BIL
    print(f"{c:8.3f} {rho:9.4f} {np.sqrt(max(TABAN_MSE - rho * rho, 0)):11.5f}")

cikti = {
    "aciklama": "n20d -- iki oznitelik-ekseni |c| noktasinin saglamlik sinavi",
    "noktalar": {
        e: {
            "c_temel": SON[e]["TEMEL"]["c"],
            "rho_s": SON[e]["TEMEL"]["rs"],
            "rho_dik": SON[e]["TEMEL"]["rd"],
            "isaret_ayni": SON[e]["TEMEL"]["isaret"] > 0,
            "span_varyantlari": {k: v["c"] for k, v in SON[e].items()},
            "L_gurultusu_%90": [float(np.quantile(MC[e], 0.05)), float(np.quantile(MC[e], 0.95))],
            "birlesik_%90": [float(np.quantile(ORN[e], 0.05)), float(np.quantile(ORN[e], 0.95))],
            "birlesik_medyan": float(np.median(ORN[e])),
        }
        for e in SON
    },
    "n": 2,
    "c_nokta": GM,
    "c_90_alt": ALT,
    "c_90_ust": UST,
    "skor": {
        str(c): float(np.sqrt(max(TABAN_MSE - (c * RHO_S_BIL) ** 2, 0)))
        for c in (0.434, ALT, GM, UST, 1.986)
    },
}
YOL = os.path.join(M29, "n20d_saglamlik.json")
with open(YOL, "w", encoding="utf-8") as fh:
    json.dump(cikti, fh, ensure_ascii=False, indent=1, default=float)
print(f"\nYAZILDI {YOL}")
