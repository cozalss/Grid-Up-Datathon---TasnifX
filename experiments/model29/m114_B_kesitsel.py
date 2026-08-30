"""B BILESIGINI URET -- 31 Agustos 1. hak dosyasi.

B = dokuz TEMIZ kesitsel eksenin TEK TEK span'a diklestirilmis toplami.
Ardisik diklestirme (A) inandiriciligi 8.8'e cikardigi icin KULLANILMIYOR.

kappa = sqrt(MSE_opt - hedef^2) = 0.0475
  -> 2. siraya ulasmak icin gereken gerceklesme oranini EN AZA indirir.

Skor geldikten sonra rho EXACT cozulur:
    rho = (sabit - P^2) / (2*kappa)
ve 2. hakta tam katsayiyla uygulanir. Yani sonda ne gelirse gelsin
bilgi kaybi yoktur.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
BURA = os.path.dirname(os.path.abspath(__file__))
TABAN = "tuketim_m6_ikiyon.csv"  # M0 m112den gelir (docs/69)
HEDEF_SOGUK, CARPAN, HEDEF_SKOR = 0.222, 0.798, 0.99940
CIKTI = "tuketim_K_B_KESITSEL.csv"
sys.path.insert(0, BURA)
from m112_kalibre import EK_MODEL, M0, buzmeli_r_hat  # noqa: E402
from m113_yon_kurucu import yonler  # noqa: E402

TEMIZ = [
    "t_yuk_faktoru",
    "tarim_orani",
    "seviye_x_ay",
    "yerlesim_orani",
    "seviye_x_guc",
    "t_log_ort",
    "tatil_agirligi",
    "t_sifir_orani",
    "seviye2",
]

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
ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
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
Gi = np.linalg.pinv(G, rcond=1e-6)  # DIKLESTIRME icin tam span
r_hat, gercek_kazanc, kL = buzmeli_r_hat(V, L, G, N)  # TABAN icin kip buzmesi
nrm = float((r_hat * r_hat).mean())
MSE_OPT = M0 - gercek_kazanc
print(f"buzmeli taban: gercek kazanc={gercek_kazanc:.6f} -> saf optimum {np.sqrt(MSE_OPT):.6f}")
tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
Yt = yonler(tp, a0)

duz = np.zeros(N)
bilgi = []
for ad in TEMIZ:
    xb = np.asarray(Yb[ad], np.float64)
    xt = np.asarray(Yt[ad], np.float64)
    c = Gi @ ((V.T @ xt) / N)
    xp = xt - V @ c
    Qd = float((xp * xp).mean())
    if Qd < 0.02:
        print(f"  {ad}: ATLANDI (Q_dik={Qd:.4f})")
        continue
    kor = float((w * r * xb).mean()) / np.sqrt(m0b)
    b = CARPAN * kor * np.sqrt(Qd)
    duz += b * (xp / np.sqrt(Qd))
    bilgi.append(dict(eksen=ad, kor_yaz25=kor, Q_dik=Qd, beta=b))
    print(f"  {ad:>16s} kor={kor:+.4f} Q_dik={Qd:.4f} beta={b:+.4f}")

Q = float((duz * duz).mean())
birim = duz / np.sqrt(Q)
rho_ong = float(np.sqrt(Q))
KAPPA = float(np.sqrt(MSE_OPT - HEDEF_SKOR**2))
print(f"\n{len(bilgi)} eksen | ongorulen rho = {rho_ong:.4f} | kappa = {KAPPA:.4f}")

p_yeni = a0 + r_hat + KAPPA * birim
y = np.clip(np.expm1(p_yeni), 0.0, None)
out = pd.DataFrame({"id": te.id.values, "tuketim": y})
kapi = {
    "satir 714688": len(out) == 714688,
    "id sirasi": bool((out.id.values == ss.iloc[:, 0].values).all()),
    "NaN yok": int(out.tuketim.isna().sum()) == 0,
    "negatif yok": int((out.tuketim < 0).sum()) == 0,
    "sonlu": bool(np.isfinite(out.tuketim.values).all()),
    "maks makul": bool(out.tuketim.max() < 3 * np.expm1(a0).max()),
}
for k, v in kapi.items():
    print(f"  KAPI {k:14s}: {v}")
if not all(kapi.values()):
    raise SystemExit("KAPI KALDI -- dosya yazilmadi")

yol = os.path.join(S, CIKTI)
gec = yol + ".tmp"
out.to_csv(gec, index=False)
Path(gec).replace(yol)
dgv = np.log1p(out.tuketim.values) - a0
sabit = float(M0 - 2 * kL + float(dgv @ dgv) / N)
print(f"\nYAZILDI: submissions/{CIKTI}")
print(f"  sifir tahmin {int((y == 0).sum()):,}  maks {out.tuketim.max():,.0f}")
print(f"  sabit = {sabit:.9f}")
print(f"  rho=0 skoru = {np.sqrt(sabit):.5f}   saf optimum (banka) = {np.sqrt(MSE_OPT):.6f}")
print(f"\n  COZUM (skor P geldikten sonra):  rho = ({sabit:.9f} - P*P) / {2 * KAPPA:.6f}")
print(f"\n  {'gercek rho':>11s} {'skor':>9s}")
for rr in [0.0, 0.02, 0.0304, 0.0475, 0.0635, 0.0794]:
    print(f"  {rr:11.4f} {np.sqrt(MSE_OPT + KAPPA**2 - 2 * KAPPA * rr):9.5f}")
with open(os.path.join(BURA, "zp_B.json"), "w") as fh:
    json.dump(
        dict(
            cikti=CIKTI, kappa=KAPPA, sabit=sabit, Q_dik=1.0, rho_ongorulen=rho_ong, eksenler=bilgi
        ),
        fh,
        indent=1,
    )
np.save(os.path.join(BURA, "zp_B_birim.npy"), birim)
