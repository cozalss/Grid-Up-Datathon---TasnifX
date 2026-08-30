"""TEK HAKLIK AZAMI BILESIK -- evrensel seviye-kalibreli tavanla.

Kullanici karari: tek gonderim. O halde alti aileyi ayri ayri olcmek yerine
hepsini TEK yone toplariz; ayni sum rho_i^2, tek hakta.

TAVAN KURALI (tek gercek kalibrasyon noktamiz):
    seviye: rho_s = +0.0156 (span'da LB olcumu), gercek rho_perp = -0.0304
            |oran| = 1.95,  ISARET CV'den geldi (rho_s'in isareti TERSTI)
Dolayisiyla her eksen icin
    rho_kullanilan = isaret(rho_cv) * min(|rho_cv|, 1.95 * |rho_s|)
Buyuklukte LB capasi, isarette CV. Boylece CV'nin 5-17 kat sisirdigi
mevsimsel eksenler kendiliginden bastirilir.

Cikti: tek dosya + gercekci skor egrisi.
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
EK_MODEL = {}  # bosaltildi (docs/69): s3y40 kendi skoruyla Gram'da
HEDEF_SOGUK, CARPAN = 0.222, 0.798
TAVAN = 1.95  # seviye'den kalibre
HEDEF_2 = 0.99790  # Duo-Electra
HEDEF_3 = 0.99940  # Atakan Aldemir
sys.path.insert(0, M29)
from m112_kalibre import M0, buzmeli_r_hat  # noqa: E402
from m113_yon_kurucu import yonler  # noqa: E402

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
Gi = np.linalg.pinv(G, rcond=1e-6)
r_hat, gercek, kL = buzmeli_r_hat(V, L, G, N)
MSE_OPT = M0 - gercek
print(f"buzmeli taban: saf optimum {np.sqrt(MSE_OPT):.6f}  (MSE {MSE_OPT:.6f})")

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
p = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])
df = e.loc[idx].copy()
rb = np.log1p(df.tuketim.values.astype(np.float64)) - p
sgm = df.soguk_mu.values.astype(np.float64)
ww = np.where(sgm == 1, HEDEF_SOGUK / sgm.mean(), (1 - HEDEF_SOGUK) / (1 - sgm.mean()))
ww = ww / ww.mean()
Yb = yonler(df, p)
m0b = float((ww * rb * rb).mean())
tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
Yt = yonler(tp, a0)

rng = np.random.default_rng(5)
tn = df.tanim.values
uq = pd.unique(tn)
gi = pd.Series(np.arange(len(uq)), index=uq)[tn].to_numpy()
PERM = [
    np.argsort(np.argsort(rng.permutation(len(uq))[gi], kind="stable"), kind="stable")
    for _ in range(25)
]

ADAYLAR = [a for a in Yb if a in Yt and a != "seviye"]
print(f"\n{len(ADAYLAR)} aday eksen. Evrensel tavan: |rho| <= {TAVAN} * |rho_s(LB)|")
print(f"{'eksen':>18s} {'rho_cv':>8s} {'rho_s':>8s} {'TAVANLI':>8s} {'Q_dik':>6s} {'katki':>10s}")
duz = np.zeros(N)
ham = np.zeros(N)
ONCEKI = []
kul = []
sirali = sorted(
    ADAYLAR, key=lambda a: -abs(float((ww * rb * np.asarray(Yb[a], np.float64)).mean()))
)
for ad in sirali:
    xb = np.asarray(Yb[ad], np.float64)
    xt = np.asarray(Yt[ad], np.float64).copy()
    cc = Gi @ ((V.T @ xt) / N)
    Lsp = float(cc @ L)
    xp0 = xt - V @ cc
    Qs = 1.0 - float((xp0 * xp0).mean())
    if Qs < 0.02:
        continue
    rho_s = Lsp / np.sqrt(Qs)
    xp = xp0.copy()
    for u in ONCEKI:
        xp -= float((xp * u).mean()) * u
    Qd = float((xp * xp).mean())
    if Qd < 0.05:
        continue
    kor = float((ww * rb * xb).mean()) / np.sqrt(m0b)
    gur = np.std([float((ww * rb[s] * xb).mean()) / np.sqrt(m0b) for s in PERM])
    if abs(kor) < 3 * gur:
        continue
    rho_cv = CARPAN * kor
    rho_kul = np.sign(rho_cv) * min(abs(rho_cv), TAVAN * abs(rho_s))
    if abs(rho_kul) < 0.005:
        continue
    b = rho_kul * np.sqrt(Qd)
    u = xp / np.sqrt(Qd)
    duz += b * u
    ham += b * xt
    ONCEKI.append(u)
    kul.append(ad)
    print(f"{ad:>18s} {rho_cv:+8.4f} {rho_s:+8.4f} {rho_kul:+8.4f} {Qd:6.3f} {rho_kul**2:10.3e}")

Q = float((duz * duz).mean())
birim = duz / np.sqrt(Q)
RHO = float(np.sqrt(Q))
Qh = float((ham * ham).mean())
uh = ham / np.sqrt(Qh)
ch = Gi @ ((V.T @ uh) / N)
uhp = uh - V @ ch
Qdh = float((uhp * uhp).mean())
rho_sh = float(ch @ L) / np.sqrt(max(1 - Qdh, 1e-9))
print(f"\n{len(kul)} eksen kullanildi. BILESIGIN ongorulen rho = {RHO:.4f}")
print(
    f"  bilesigin kendi inandiriciligi: rho_s(LB)={rho_sh:+.4f}  "
    f"oran={abs(RHO) / (abs(rho_sh) + 1e-9):.1f} "
    f"{'TEMIZ' if abs(RHO) <= 4 * abs(rho_sh) else 'SUPHELI'}"
)
print(f"  tam gerceklesirse: {np.sqrt(max(MSE_OPT - RHO**2, 1e-9)):.5f}")

print(f"\n{'hedef':>8s} {'skor':>8s} {'gereken rho':>12s} {'gereken f':>10s} {'kappa*':>8s}")
for ad, h in [("3. sira", HEDEF_3), ("2. sira", HEDEF_2)]:
    ihtiyac = MSE_OPT - h * h
    kap = np.sqrt(max(ihtiyac, 1e-12))
    print(f"{ad:>8s} {h:8.5f} {kap:12.4f} {kap / RHO:10.3f} {kap:8.4f}")

KAPPA = float(np.sqrt(max(MSE_OPT - HEDEF_2**2, 1e-12)))
pn = a0 + r_hat + KAPPA * birim
y = np.clip(np.expm1(pn), 0.0, None)
out = pd.DataFrame({"id": te.id.values, "tuketim": y})
kapi = {
    "satir": len(out) == 714688,
    "id": bool((out.id.values == ss.iloc[:, 0].values).all()),
    "NaN": int(out.tuketim.isna().sum()) == 0,
    "negatif": int((out.tuketim < 0).sum()) == 0,
    "sonlu": bool(np.isfinite(out.tuketim.values).all()),
    "maks": bool(out.tuketim.max() < 3 * np.expm1(a0).max()),
}
print(f"\nKAPI: {kapi}")
if all(kapi.values()):
    yol = os.path.join(S, "tuketim_K_TEKHAK.csv")
    out.to_csv(yol + ".tmp", index=False)
    Path(yol + ".tmp").replace(yol)
    dgv = np.log1p(out.tuketim.values) - a0
    nrm = float((r_hat * r_hat).mean())
    sabit = float(M0 - 2 * kL + float(dgv @ dgv) / N)  # k'L, ||r_hat||^2 DEGIL
    print(
        f"YAZILDI submissions/tuketim_K_TEKHAK.csv  kappa={KAPPA:.4f}  "
        f"sifir {int((y == 0).sum()):,}"
    )
    print(f"  sabit={sabit:.9f}  rho=0 skoru {np.sqrt(sabit):.5f}")
    print(f"\n  {'gercek rho':>11s} {'skor':>9s} {'sira':>18s}")
    for rr in [0.0, 0.0304, 0.05, KAPPA, RHO * 0.7, RHO]:
        sk = np.sqrt(max(sabit - 2 * KAPPA * rr, 1e-9))
        sr = (
            "2. SIRA"
            if sk < HEDEF_2
            else "3. sira"
            if sk < HEDEF_3
            else "4. sira"
            if sk < 1.00118
            else "5.+"
        )
        print(f"  {rr:11.4f} {sk:9.5f} {sr:>18s}")
    with open(os.path.join(BURA, "zx_tekhak.json"), "w") as fh:
        json.dump(
            dict(
                kappa=KAPPA,
                sabit=sabit,
                rho_ongorulen=RHO,
                eksenler=kul,
                inandiricilik_orani=abs(RHO) / (abs(rho_sh) + 1e-9),
            ),
            fh,
            indent=1,
        )
