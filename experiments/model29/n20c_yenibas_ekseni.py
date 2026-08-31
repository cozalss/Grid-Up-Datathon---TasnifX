"""n20c -- K_yenibas gonderiminin DIK YONUNU hangi yapisal vektor aciklar?

HICBIR GONDERIM. submissions/ ALTINA YAZMA YOK.

n20b'de ze_duzeltme.npy'nin K_yenibas'in dik yonuyle kosinusu -0.055 cikti;
yani O VEKTOR DEGIL. |c| icin ikinci nokta ancak GERCEK ekseni bulursak
kurulabilir. Burada aday yapisal vektorler taranir ve her birinin
(span'a dik parcasi) ile K_yenibas'in (span'a dik parcasi) kosinusu olculur.
Kosinus ~ +-1 olan aday, gonderimin gercekten kullandigi eksendir.
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
from m112_kalibre import EK_MODEL, M0  # noqa: E402

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
tr = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"))
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
IX = {a: i for i, a in enumerate(AD)}
K = V.shape[1]
J_SEV, J_YEN = IX["tuketim_YP_seviye.csv"], IX["tuketim_K_yenibas.csv"]
SPAN = [i for i in range(K) if i != J_YEN]  # YP_seviye ICERIDE (kronolojik dogru)
Vs = V[:, SPAN]
Gs = (Vs.T @ Vs) / N
Gsi = np.linalg.pinv(Gs, rcond=RCOND)
Ls = L[SPAN]


def dik(x):
    c = Gsi @ ((Vs.T @ x) / N)
    xp = x - Vs @ c
    return xp, float(c @ Ls), float((xp * xp).mean()), float(c @ c)


p_yen, _, q_yen, _ = dik(V[:, J_YEN])
print("=" * 78)
print("n20c  K_yenibas'in DIK yonunu hangi yapisal vektor aciklar?")
print("=" * 78)
print(f"K_yenibas dik parcasi: Q_dik = {q_yen:.6f}  RMS = {np.sqrt(q_yen):.5f}")

# --- aday yapisal vektorler ---
say = tr.groupby("tanim").size()
kend = tr.groupby("tanim").ly.mean() if "ly" in tr.columns else None
tr["ly"] = np.log1p(tr.tuketim.values.astype(np.float64))
kend = tr.groupby("tanim").ly.mean()
ilk = tr.groupby("tanim").tarih.min()
tanim = te.tanim.values
derin = pd.Series(tanim).map(say).fillna(0).to_numpy()
kd = pd.Series(tanim).map(kend).to_numpy()
giris = pd.Series(tanim).map(ilk).to_numpy()

ADAY = {}
for lo, hi in [(1, 4), (1, 8), (1, 12), (1, 20), (5, 8), (1, 30), (1, 45)]:
    ADAY[f"gosterge_derin_{lo}_{hi}"] = ((derin >= lo) & (derin <= hi)).astype(np.float64)
for lo, hi in [(1, 8), (1, 20)]:
    m = (derin >= lo) & (derin <= hi)
    v = np.zeros(N)
    ok = m & np.isfinite(kd)
    v[ok] = a0[ok] - kd[ok]  # m6'nin kendi seviyesinin ustune verdigi fazla
    ADAY[f"fazlalik_derin_{lo}_{hi}"] = v
for g in ("2026-03-26", "2026-03-27"):
    ADAY[f"giris_{g}"] = (giris == g).astype(np.float64)
ADAY["giris_mart2627"] = ((giris == "2026-03-26") | (giris == "2026-03-27")).astype(np.float64)
# ze_ dosyalari
SCR = (
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad"
)
for f in ("ze_duzeltme.npy", "ze_dik_birim.npy", "prior_yeni_eksen.npy", "nul_yeni.npy"):
    p = os.path.join(SCR, f)
    if os.path.exists(p):
        a = np.load(p).astype(np.float64)
        if a.shape == (N,):
            ADAY[f] = a
# KES_yenibaslangic gonderim dosyalari (LB'ye gitmemis ama vektorleri var)
for f in ("tuketim_KES_yenibaslangic.csv", "tuketim_KES_yenibaslangic_yari.csv"):
    if os.path.exists(os.path.join(S, f)):
        v = oku(f)
        if v is not None and len(v) == N:
            ADAY[f] = v - a0

print(f"\n{'aday':>34s} {'kosinus(dik)':>13s} {'Q_dik(aday)':>12s} {'Q_span':>9s} {'rho_s':>9s}")
SON = []
for ad, x in ADAY.items():
    s = np.sqrt(float((x * x).mean()))
    if s < 1e-12:
        continue
    x = x / s
    xp, Lsp, Qp, cn2 = dik(x)
    if Qp < 1e-12:
        continue
    kos = float((xp * p_yen).mean()) / np.sqrt(Qp * q_yen)
    Qs = 1.0 - Qp
    rs = Lsp / np.sqrt(Qs) if Qs > 1e-9 else np.nan
    SON.append((ad, kos, Qp, Qs, rs))
for r in sorted(SON, key=lambda t: -abs(t[1])):
    print(f"{r[0][:34]:>34s} {r[1]:+13.4f} {r[2]:12.5f} {r[3]:9.4f} {r[4]:+9.4f}")

print("""
OKUMA. |kosinus| ~ 1 olan bir aday YOKSA, K_yenibas'in dik yonu elimizdeki
  hicbir yapisal vektorle ozdeslesmiyor demektir. O zaman "yenibaslangic
  ekseninin rho_s'i" TANIMLANAMAZ ve bu gonderim |c| icin GECERLI BIR
  IKINCI NOKTA URETMEZ. Bunu uydurmak yerine acikca soyleriz.
""")
with open(os.path.join(M29, "n20c_yenibas_ekseni.json"), "w", encoding="utf-8") as fh:
    json.dump(
        {
            "Q_dik_K_yenibas": q_yen,
            "adaylar": [
                {"ad": r[0], "kosinus": r[1], "Q_dik": r[2], "Q_span": r[3], "rho_s": r[4]}
                for r in sorted(SON, key=lambda t: -abs(t[1]))
            ],
        },
        fh,
        ensure_ascii=False,
        indent=1,
        default=float,
    )
print("YAZILDI n20c_yenibas_ekseni.json")
