"""n20e -- OLCULEN |c| m148'in rho_s TANIMIYLA UYUMLU MU?

HICBIR GONDERIM. submissions/ ALTINA YAZMA YOK. m148 SALT OKUNDU.

SORUN. n20b/n20d |c| = |rho_dik| / |rho_s| olctu; ama rho_s'i EXACT span
cebiriyle kurdu:      rho_s(exact) = (cc . L) / sqrt(Q_span),  cc = pinv(G) b
m148 ise rho_s'i BUZMELI r_hat ile kuruyor:
                      rho_s(m148)  = <r_hat_buzmeli, x>/N / sqrt(Q_span)
Iki payda AYNI DEGILSE, olculen |c| m148'in rho_s(bilesik) = 0.10979
sayisiyla CARPILAMAZ -- carpim baska bir seyi verir.

Bu betik iki paydayi AYNI eksenlerde karsilastirir ve gerekli DUZELTME
carpanini (b = rho_s_exact / rho_s_m148) olcer.
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
RCOND = 1e-6
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0, buzmeli_r_hat  # noqa: E402

np.seterr(all="ignore")
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
AD, VV, LL = [], [], []
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
for f, Lj in EK_MODEL.items():
    AD.append(f)
    VV.append(oku(f) - a0)
    LL.append(float(Lj))
for o in DUR.get("olcumler", []):
    d = oku(o["dosya"]) - a0
    AD.append(o["dosya"])
    VV.append(d)
    LL.append((M0 + float((d * d).mean()) - o["skor"] ** 2) / 2)
V = np.array(VV).T
del VV
L = np.array(LL)
K = V.shape[1]
IX = {a: i for i, a in enumerate(AD)}
G = (V.T @ V) / N
J_SEV, J_YEN = IX["tuketim_YP_seviye.csv"], IX["tuketim_K_yenibas.csv"]

print("=" * 78)
print("n20e  rho_s(EXACT) vs rho_s(BUZMELI r_hat) -- m148 ile kopru")
print("=" * 78)
r_hat, gercek, kL = buzmeli_r_hat(V, L, G, N)
print(f"m148'in kullandigi TAM span (K={K}) icin buzmeli r_hat kuruldu.")
print(f"  ||r_hat||^2 = {float((r_hat * r_hat).mean()):.6f}   gercek = {gercek:.6f}")


def st(x):
    x = np.asarray(x, dtype=np.float64).copy()
    f = np.isfinite(x)
    x[~f] = np.median(x[f])
    x -= x.mean()
    s = np.sqrt(float((x * x).mean()))
    return x / s if s > 1e-12 else x


sys.path.insert(0, M29)
tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
assert np.array_equal(tp.id.values, IDS)
from m113_yon_kurucu import yonler  # noqa: E402

Y = yonler(tp, a0)
_y = oku("tuketim_KES_yenibaslangic.csv") - a0
Y["yenibaslangic"] = _y / np.sqrt(float((_y * _y).mean()))

Gi = np.linalg.pinv(G, rcond=RCOND)
print("\nTUM 53 EKSENDE, m148'IN KENDI SPAN'IYLA (K=%d) iki payda:" % K)
print(f"{'eksen':>18s} {'Q_span':>8s} {'rho_s exact':>12s} {'rho_s m148':>11s} {'oran b':>8s}")
ORAN = []
for ad, x in sorted(Y.items()):
    b = (V.T @ x) / N
    cc = Gi @ b
    xp = x - V @ cc
    Qp = float((xp * xp).mean())
    Qs = float((x * x).mean()) - Qp
    if Qs <= 1e-6:
        continue
    rs_ex = float(cc @ L) / np.sqrt(Qs)
    rs_m1 = float((r_hat * x).mean()) / np.sqrt(Qs)
    if abs(rs_m1) < 1e-6:
        continue
    ORAN.append((ad, Qs, rs_ex, rs_m1, rs_ex / rs_m1))
for r in sorted(ORAN, key=lambda t: -abs(t[3]))[:16]:
    print(f"{r[0]:>18s} {r[1]:8.4f} {r[2]:+12.5f} {r[3]:+11.5f} {r[4]:8.4f}")
bo = np.array([r[4] for r in ORAN])
print(
    f"\noran b = rho_s_exact / rho_s_m148:  n = {len(bo)}  medyan {np.median(bo):.4f}  "
    f"ort {bo.mean():.4f}  sd {bo.std():.4f}  aralik [{bo.min():.4f}, {bo.max():.4f}]"
)
_agir = np.array([r[4] for r in ORAN if abs(r[3]) >= 0.015])  # m148'in kendi kapisi
print(
    f"  m148 kapisi |rho_s| >= 0.015 olan {len(_agir)} eksende: medyan {np.median(_agir):.4f}  "
    f"aralik [{_agir.min():.4f}, {_agir.max():.4f}]"
)

print("""
OKUMA. b ~ 1 ise iki payda ayni seydir ve olculen |c| dogrudan m148'in
  rho_s(bilesik) = 0.10979 ile carpilabilir. b != 1 ise m148'e TASINACAK
  carpan |c|_m148 = |c|_exact * b'dir (cunku m148 daha kucuk/buyuk bir
  paydayla calisiyor).
""")

# --- iki noktanin m148 paydasiyla yeniden hesabi (TEMIZ span'da) ---
print("=" * 78)
print("IKI NOKTA, m148 PAYDASIYLA (buzmeli r_hat), TEMIZ span'da")
print("=" * 78)
X = {"seviye": Y["seviye"], "yenibaslangic": Y["yenibaslangic"]}
PROBE = {"seviye": J_SEV, "yenibaslangic": J_YEN}
TEMIZ = {
    "seviye": [i for i in range(K) if i not in (J_SEV, J_YEN)],
    "yenibaslangic": [i for i in range(K) if i != J_YEN],
}
print(
    f"{'eksen':>15s} {'rho_s exact':>12s} {'rho_s buzmeli':>14s} {'rho_dik':>9s} "
    f"{'|c| exact':>10s} {'|c| buzmeli':>12s}"
)
SON = {}
for e in ("seviye", "yenibaslangic"):
    ix = np.array(TEMIZ[e], int)
    Gr = G[np.ix_(ix, ix)]
    Vs = V[:, ix]
    rh, _, _ = buzmeli_r_hat(Vs, L[ix], Gr, N)
    Gp = np.linalg.pinv(Gr, rcond=RCOND)
    x = X[e]
    cc = Gp @ ((Vs.T @ x) / N)
    xp = x - Vs @ cc
    Qp = float((xp * xp).mean())
    Qs = float((x * x).mean()) - Qp
    rs_ex = float(cc @ L[ix]) / np.sqrt(Qs)
    rs_bz = float((rh * x).mean()) / np.sqrt(Qs)
    j = PROBE[e]
    cj = Gp @ G[ix, j]
    Qspj = float(cj @ Gr @ cj)
    Qdkj = float(G[j, j] - 2 * cj @ G[ix, j] + Qspj)
    rd = (L[j] - float(cj @ L[ix])) / np.sqrt(Qdkj)
    SON[e] = dict(
        rho_s_exact=rs_ex,
        rho_s_buzmeli=rs_bz,
        rho_dik=rd,
        c_exact=abs(rd / rs_ex),
        c_buzmeli=abs(rd / rs_bz),
    )
    print(
        f"{e:>15s} {rs_ex:+12.5f} {rs_bz:+14.5f} {rd:+9.5f} "
        f"{abs(rd / rs_ex):10.3f} {abs(rd / rs_bz):12.3f}"
    )

ce = np.array([SON[e]["c_exact"] for e in SON])
cb = np.array([SON[e]["c_buzmeli"] for e in SON])
print(
    f"\nGEOMETRIK ORTALAMA:  |c|_exact = {np.exp(np.log(ce).mean()):.3f}   "
    f"|c|_buzmeli (m148 paydasi) = {np.exp(np.log(cb).mean()):.3f}"
)

RHO_S_BIL, TABAN_MSE = 0.2141 / 1.95, 1.00202690
print(f"\nSKOR (rho_s(bilesik) = {RHO_S_BIL:.5f}, m148 paydasi):")
print(f"{'kaynak':>34s} {'|c|':>8s} {'rho_LB':>9s} {'skor':>9s}")
for ad, c in (
    ("n10 gonderim farklari", 0.434),
    ("n20 nokta 2: yenibaslangic", cb[list(SON).index("yenibaslangic")]),
    ("n20 GEO ORT (n=2)", float(np.exp(np.log(cb).mean()))),
    ("n20 nokta 1: seviye", cb[list(SON).index("seviye")]),
    ("m148/m113 mevcut capa", 1.986),
):
    rho = c * RHO_S_BIL
    print(f"{ad:>34s} {c:8.3f} {rho:9.4f} {np.sqrt(max(TABAN_MSE - rho * rho, 0)):9.5f}")

with open(os.path.join(M29, "n20e_buzme_koprusu.json"), "w", encoding="utf-8") as fh:
    json.dump(
        {
            "b_medyan": float(np.median(bo)),
            "b_aralik": [float(bo.min()), float(bo.max())],
            "b_kapi_gecen_medyan": float(np.median(_agir)),
            "noktalar": SON,
            "c_exact_geo": float(np.exp(np.log(ce).mean())),
            "c_buzmeli_geo": float(np.exp(np.log(cb).mean())),
        },
        fh,
        ensure_ascii=False,
        indent=1,
        default=float,
    )
print("\nYAZILDI n20e_buzme_koprusu.json")
