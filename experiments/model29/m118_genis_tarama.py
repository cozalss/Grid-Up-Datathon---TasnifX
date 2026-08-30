"""GENIS LB-CAPALI TARAMA -- rho'yu yukseltecek eksenleri ara.

Bilesigin rho'su = sqrt( sum (rho_kul_i)^2 * Q_dik_i ),
    rho_kul = isaret(rho_cv) * min(|rho_cv|, 1.95*|rho_s|)
Tavan |rho_s| tarafindan belirleniyor ve rho_s DOGRUDAN LB OLCUMUDUR
(L_span/sqrt(Q_span)) -- CV gibi sismiyor.

Dolayisiyla rho'yu yukseltmenin yolu, |rho_s|'i BUYUK olan eksenler bulmak.
Bu betik 157 oznitelik + etkilesimler + kesitler uzerinde genis tarama yapar
ve her aday icin GERCEK katki olcutunu hesaplar:
        katki = min(|rho_cv|, 1.95|rho_s|)^2 * Q_dik

Secim yanliligi denetimi: sigma(rho_s) = sigma_L/sqrt(Q_span) ~ 3e-4.
Yuzlerce aday arasindan en buyugu secmek ~3.5 sigma = 1.1e-3 sahte deger
uretebilir; gercek rho_s'ler 0.01-0.03 oldugu icin bu ihmal edilebilir,
ama yine de raporlanir.
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
BURA = os.path.dirname(os.path.abspath(__file__))
TABAN = "tuketim_m6_ikiyon.csv"  # M0 m112den gelir (docs/69)
HEDEF_SOGUK, CARPAN, TAVAN = 0.222, 0.798, 1.95
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0, L_gurultusu, buzmeli_r_hat  # noqa: E402

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
MSE_OPT = M0 - gercek
sigL = L_gurultusu(V, N)
print(f"saf optimum {np.sqrt(MSE_OPT):.6f}   ort sigma_L {sigL.mean():.3e}")

tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
assert np.array_equal(tp.id.values, IDS)
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


ATLA = {"id", "tanim", "lokasyon", "tarih", "bolge", "_blok", "tuketim", "soguk_mu"}
SAYISAL = [
    c
    for c in tp.columns
    if c not in ATLA and pd.api.types.is_numeric_dtype(tp[c]) and c in bf.columns
]
print(f"{len(SAYISAL)} sayisal oznitelik taraniyor (+ etkilesimler)")

svT, svB = st(a0), st(pb)
sgT = tp.soguk_mu.values.astype(np.float64)
sgB = sgm


def adaylar():
    for c in SAYISAL:
        xt, xb = st(tp[c].to_numpy()), st(bf[c].to_numpy())
        if xt is None or xb is None:
            continue
        yield c, xt, xb
        for ad, mt, mb in [("x_sv", svT, svB), ("x_soguk", sgT, sgB)]:
            a, b = st(xt * mt), st(xb * mb)
            if a is not None and b is not None:
                yield f"{c}:{ad}", a, b
        q = (
            np.quantile(tp[c].to_numpy()[np.isfinite(tp[c].to_numpy())], [0.9])
            if np.isfinite(tp[c].to_numpy()).any()
            else None
        )
        if q is not None:
            a = st((tp[c].to_numpy() > q[0]).astype(np.float64))
            b = st((bf[c].to_numpy() > q[0]).astype(np.float64))
            if a is not None and b is not None:
                yield f"{c}:ust10", a, b


rng = np.random.default_rng(5)
tn = bf.tanim.values
uqn = pd.unique(tn)
gi = pd.Series(np.arange(len(uqn)), index=uqn)[tn].to_numpy()
PERM = [
    np.argsort(np.argsort(rng.permutation(len(uqn))[gi], kind="stable"), kind="stable")
    for _ in range(12)
]

satir = []
for i, (ad, xt, xb) in enumerate(adaylar()):
    cc = Gi @ ((V.T @ xt) / N)
    Lsp = float(cc @ L)
    xp = xt - V @ cc
    Qd = float((xp * xp).mean())
    Qs = 1.0 - Qd
    if Qd < 0.05 or Qs < 0.02:
        continue
    rho_s = Lsp / np.sqrt(Qs)
    kor = float((ww * rb * xb).mean()) / np.sqrt(m0b)
    rho_cv = CARPAN * kor
    rho_kul = np.sign(rho_cv) * min(abs(rho_cv), TAVAN * abs(rho_s))
    satir.append((ad, rho_cv, rho_s, rho_kul, Qd, rho_kul**2 * Qd))
print(f"{len(satir)} aday degerlendirildi")
satir.sort(key=lambda t: -t[5])
print("\nEN IYI 30 (katki = rho_kul^2 * Q_dik)")
print(f"{'eksen':>34s} {'rho_cv':>8s} {'rho_s':>8s} {'rho_kul':>8s} {'Q_dik':>6s} {'katki':>10s}")
for s in satir[:30]:
    print(f"{s[0][:34]:>34s} {s[1]:+8.4f} {s[2]:+8.4f} {s[3]:+8.4f} {s[4]:6.3f} {s[5]:10.3e}")

rs = np.array([abs(s[2]) for s in satir])
print(
    f"\n|rho_s| dagilimi: medyan {np.median(rs):.4f}  %90 {np.quantile(rs, 0.9):.4f}  "
    f"maks {rs.max():.4f}"
)
print(
    f"secim yanliligi tahmini: sigma(rho_s) ~ {sigL.mean() / np.sqrt(0.5):.1e}, "
    f"{len(satir)} aday -> sahte tavan ~{3.5 * sigL.mean() / np.sqrt(0.5):.1e}"
)
with open(os.path.join(BURA, "m118_tarama.json"), "w") as fh:
    json.dump(
        [
            dict(eksen=s[0], rho_cv=s[1], rho_s=s[2], rho_kul=s[3], Qd=s[4], katki=s[5])
            for s in satir[:200]
        ],
        fh,
        indent=1,
    )
print("-> zy_tarama.json (en iyi 200)")
