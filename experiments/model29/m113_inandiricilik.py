"""INANDIRICILIK SUZGECI -- CV tahminini LB'nin KENDI olcumuyle sinama.

Sorun: CV blok artigi, HAM blok modelinin mevsim rampasi yanliligiyla dolu.
a0 bu yanliliga sahip DEGIL (2026 daha serin bulgusu). Bu yuzden mevsimsel
eksenlerde CV korelasyonu sisik.

SUZGEC: her yon x icin LB, x'in span parcasindaki korelasyonu OLCMUSTUR:
    rho_s = L_span / sqrt(Q_span)
CV ise tum x icin rho_cv = carpan * kor ongoruyor. Ikisi ayni buyukluk
mertebesinde degilse, CV o eksende GUVENILMEZDIR.

    inandiricilik = rho_s / rho_cv

  seviye  : rho_s=+0.0156  rho_cv=-0.0304  -> mertebe ayni (isaret farki
            Kural 56 geregi beklenir), CV LB'yi 0.80 oraninda dogru bildi.
  ay      : rho_s=+0.0115  rho_cv=+0.1585  -> 14 kat sapma, GUVENILMEZ.

Karar kurali: |rho_cv| > 4*|rho_s| ise eksen "mevsim-kirli" sayilir ve
tahmini rho_s mertebesine indirilir.
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
EK_MODEL = {}  # bosaltildi (docs/69): s3y40 kendi skoruyla Gram'da
HEDEF_SOGUK, CARPAN = 0.222, 0.798
sys.path.insert(0, M29)
from m112_kalibre import M0  # noqa: E402

sys.path.insert(0, BURA)
from m113_yon_kurucu import yonler  # noqa: E402

MEVSIMSEL = {
    "ay",
    "cdd22",
    "cdd24",
    "cdd22_ort7",
    "ufuk_gun",
    "ufuk_kare",
    "log_ufuk",
    "sicaklik_ort",
    "sicaklik_max",
    "nem_ort",
    "vpd",
    "et0",
    "gunes_radyasyon",
    "gun_uzunlugu",
    "ulusal_gunluk",
    "toprak_nem",
    "yagis",
    "soguk_x_ufuk",
    "soguk_x_cdd",
    "seviye_x_ufuk",
    "seviye_x_cdd",
}

e = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
blk = e[e._blok == "yaz25"]
sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
P = [
    np.load(os.path.join(AO, f"yaz25_{t}_{a}_uretim.npy")).astype(np.float64)
    for t in (1000, 1001, 1002)
    for a in ("cat", "xgb", "lgbm")
    if os.path.exists(os.path.join(AO, f"yaz25_{t}_{a}_uretim.npy"))
]
z = np.load(os.path.join(DN, "soguk_tahmin_yaz25.npz"))
idx = np.concatenate([sic.index.values, sog.index.values])
p = np.concatenate([np.mean(P, axis=0), np.mean([z[k] for k in z.files], axis=0)])
df = e.loc[idx].copy()
r = np.log1p(df.tuketim.values.astype(np.float64)) - p
sg = df.soguk_mu.values.astype(np.float64)
w = np.where(sg == 1, HEDEF_SOGUK / sg.mean(), (1 - HEDEF_SOGUK) / (1 - sg.mean()))
w = w / w.mean()
Yb = yonler(df, p)
m0b = float((w * r * r).mean())

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
nrm = float(((V @ (Gi @ L)) ** 2).mean())
tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
Yt = yonler(tp, a0)

rng = np.random.default_rng(5)
tn = df.tanim.values
uq = pd.unique(tn)
gi = pd.Series(np.arange(len(uq)), index=uq)[tn].to_numpy()
PERM = [
    np.argsort(np.argsort(rng.permutation(len(uq))[gi], kind="stable"), kind="stable")
    for _ in range(30)
]

print(
    f"{'eksen':>18s} {'kor_cv':>8s} {'z':>7s} {'Q_dik':>7s} {'rho_s(LB)':>10s} "
    f"{'rho_cv':>8s} {'oran':>7s} {'HUKUM':>12s} {'rho_kul':>8s} {'kazanc':>10s}"
)
rows = []
for ad in Yb:
    if ad not in Yt or ad in ("seviye",):
        continue
    xb = np.asarray(Yb[ad], np.float64)
    xt = np.asarray(Yt[ad], np.float64)
    c = Gi @ ((V.T @ xt) / N)
    Lsp = float(c @ L)
    xtp = xt - V @ c
    Qd = float((xtp * xtp).mean())
    Qs = 1.0 - Qd
    if Qd < 0.02 or Qs < 0.02:
        continue
    kor = float((w * r * xb).mean()) / np.sqrt(m0b)
    gur = np.std([float((w * r[s] * xb).mean()) / np.sqrt(m0b) for s in PERM])
    zsk = kor / (gur + 1e-12)
    rho_s = Lsp / np.sqrt(Qs)
    rho_cv = CARPAN * kor
    oran = abs(rho_cv) / (abs(rho_s) + 1e-9)
    kirli = (ad in MEVSIMSEL) or (oran > 4.0)
    rho_kul = np.sign(rho_cv) * min(abs(rho_cv), 4.0 * abs(rho_s)) if kirli else rho_cv
    if abs(zsk) < 3.0:
        rho_kul = 0.0
    hukum = "MEVSIM-KIRLI" if kirli else ("GURULTU" if abs(zsk) < 3 else "TEMIZ")
    kaz = (rho_kul**2) * Qd
    rows.append((ad, kor, zsk, Qd, rho_s, rho_cv, oran, hukum, rho_kul, kaz))
rows.sort(key=lambda t: -t[9])
for s in rows:
    print(
        f"{s[0]:>18s} {s[1]:+8.4f} {s[2]:+7.1f} {s[3]:7.4f} {s[4]:+10.4f} "
        f"{s[5]:+8.4f} {s[6]:7.1f} {s[7]:>12s} {s[8]:+8.4f} {s[9]:10.3e}"
    )

temiz = [s for s in rows if s[7] == "TEMIZ"]
print(f"\nTEMIZ eksen sayisi: {len(temiz)}")
print(f"saf optimum: {np.sqrt(M0 - nrm):.6f}")
for ad, sec in [("TEMIZ ilk 5", temiz[:5]), ("TUM temiz", temiz)]:
    t = sum(s[9] for s in sec)
    print(f"  {ad:12s}: toplam kazanc {t:.6f} -> {np.sqrt(max(M0 - nrm - t, 1e-9)):.6f}")
print("  2. sira icin 0.002256 gerek | 1. sira icin 0.020778")
print("\n6 HAK ICIN SIRALI PLAN (yalniz TEMIZ):")
for i, s in enumerate(temiz[:6], 1):
    print(
        f"  {i}. {s[0]:>18s}  rho_tah={s[8]:+.4f}  Q_dik={s[3]:.3f}  "
        f"kazanc={s[9]:.3e}  z={s[2]:+.1f}"
    )
with open(os.path.join(BURA, "zl_plan.json"), "w") as fh:
    json.dump(
        [
            dict(
                eksen=s[0],
                kor=s[1],
                z=s[2],
                Qd=s[3],
                rho_s=s[4],
                rho_cv=s[5],
                oran=s[6],
                hukum=s[7],
                rho=s[8],
                kazanc=s[9],
            )
            for s in rows
        ],
        fh,
        indent=1,
    )
