"""KIRICI: Plan B iddialarini bagimsiz turet ve sayisal olarak kir."""

import json
import os

import numpy as np
import pandas as pd
from m30_ozellik import KOK

S = os.path.join(KOK, "submissions")
OUT = {}

skor = {
    k: v for k, v in json.load(open("olculmus_skorlar.json")).items() if k != "gun1_baseline.csv"
}
adlar = sorted(skor)
P = np.array([np.log1p(pd.read_csv(os.path.join(S, a)).tuketim.values) for a in adlar])
N = P.shape[1]
m = np.array([skor[a] ** 2 for a in adlar])
TAB = "tuketim_m6_ikiyon.csv"
ti = adlar.index(TAB)
a0 = P[ti]
m0 = m[ti]
D = P - a0
Q = np.einsum("ij,ij->i", D, D) / N
L = (m0 + Q - m) / 2  # L_j = <r,d_j>/N ,  r = t - a0
print(f"N={N}  n={len(adlar)}  m0={m0:.6f}")

# --- IDDIA 1a: turetimin kendisi -------------------------------------------
# m_j = |g_j - t|^2/N = |d_j - r|^2/N = Q_j - 2<r,d_j>/N + m0
# => <r,d_j>/N = (m0 + Q_j - m_j)/2 = L_j          (m101'deki "-L_j" yorumu YANLIS isaret,
#                                                   ama kullanilan cebir tutarli)
# tek yon icin optimum k = L/Q, kazanc = L^2/Q. DOGRULAMA: L>0 olan yonler kazandirmali.
print(
    "dogrulama: en iyi tek yon kazanci", f"{max(L**2 / Q):.6f} ({adlar[int(np.argmax(L**2 / Q))]})"
)

# --- g7 ---------------------------------------------------------------------
g = np.log1p(pd.read_csv(os.path.join(S, "tuketim_g7_span_tau3.csv")).tuketim.values)
n = len(adlar)
Gd = P @ P.T / N
M = np.block([[Gd, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
rhs = np.concatenate([P @ g / N, [1.0]])
w = np.linalg.lstsq(M, rhs, rcond=None)[0][:n]
e = w @ P - g  # span disi bilesen (isaretli)
artik = float((e**2).mean())
dg = g - a0
Qg = float((dg**2).mean())
Lg = float(w @ L)
print(f"\n[g7] sum(w)={w.sum():.9f}  artik={artik:.3e}  |w|_1={np.abs(w).sum():.3f}")
print(f"     L={Lg:+.6f}  Q={Qg:.6f}  kazanc={Lg**2 / Qg:.6f}  skor={np.sqrt(m0 - Lg**2 / Qg):.5f}")

# --- IDDIA 1b: w TEK DEGIL -> L(g7) degisiyor mu? ---------------------------
rank = np.linalg.matrix_rank(M, tol=1e-9)
u_, s_, vt_ = np.linalg.svd(M)
null = vt_[s_ < 1e-9 * s_[0]]  # M'nin cekirdegi
print(f"\n[cekirdek] M {M.shape} rank={rank}  bos boyut={len(null)}")
sap = []
rng = np.random.default_rng(0)
for t_ in range(200):
    if len(null) == 0:
        break
    c = rng.normal(size=len(null))
    v = (c @ null)[:n]
    for olcek in (1.0, 10.0, 100.0):
        w2 = w + olcek * v / max(np.abs(v).max(), 1e-30)
        r2 = float(((w2 @ P - g) ** 2).mean())
        sap.append((abs(float(w2 @ L) - Lg), abs(w2.sum() - 1.0), r2, np.abs(w2).sum()))
sap = np.array(sap) if sap else np.zeros((1, 4))
print(
    f"  |dL| maks={sap[:, 0].max():.3e}  |sum(w)-1| maks={sap[:, 1].max():.3e}"
    f"  artik maks={sap[:, 2].max():.3e}  |w|_1 maks={sap[:, 3].max():.1f}"
)

# --- IDDIA 1c: span DISI bilesenin L'ye katkisi -----------------------------
# <r, e>/N bilinemez. Sinir: r'nin span-disi bileseniyle Cauchy-Schwarz.
Gp = D @ D.T / N
r_span2 = float(L @ np.linalg.pinv(Gp, rcond=1e-10) @ L)  # |P_span r|^2/N
r_perp2 = max(m0 - r_span2, 0.0)
sinir = float(np.sqrt(r_perp2 * artik))
print(f"\n[span disi] |e|^2/N={artik:.3e} (rms={np.sqrt(artik):.2e} log-birim)")
print(f"  |r_span|^2/N={r_span2:.6f}  |r_perp|^2/N={r_perp2:.6f}")
print(f"  |<r,e>/N| <= {sinir:.3e}   (L={Lg:.6f}'nin %{100 * sinir / abs(Lg):.1f}'i)")
band = []
for d in (-sinir, 0.0, sinir):
    Lx = Lg + d
    band.append(np.sqrt(max(m0 - Lx**2 / Qg, 0)))
print(
    f"  skor bandi: {band[0]:.5f} .. {band[1]:.5f} .. {band[2]:.5f}"
    f"   (yari genislik {abs(band[0] - band[1]):.5f})"
)

# CSV yuvarlama katkisi: dosyalarin kendi hassasiyeti
raw = pd.read_csv(os.path.join(S, "tuketim_g7_span_tau3.csv")).tuketim.values
ondalik = max(
    len(str(x).split(".")[-1])
    for x in pd.read_csv(
        os.path.join(S, "tuketim_g7_span_tau3.csv"), dtype=str, nrows=2000
    ).tuketim.values
    if "." in str(x)
)
print(f"  g7 CSV ondalik basamak (ilk 2000) = {ondalik}")

# --- LOO: makineyi gercek veriyle sina --------------------------------------
loo = []
for i in range(n):
    if i == ti:
        continue
    idx = [j for j in range(n) if j != i]
    Pi = P[idx]
    Mi = np.block([[Pi @ Pi.T / N, np.ones((n - 1, 1))], [np.ones((1, n - 1)), np.zeros((1, 1))]])
    wi = np.linalg.lstsq(Mi, np.concatenate([Pi @ P[i] / N, [1.0]]), rcond=None)[0][: n - 1]
    ri = float(((wi @ Pi - P[i]) ** 2).mean())
    loo.append((adlar[i], float(wi @ L[idx]), float(L[i]), ri))
la = np.array([[x[1], x[2], x[3]] for x in loo])
hata = la[:, 0] - la[:, 1]
print(
    f"\n[LOO] n={len(loo)}  |dL| ort={np.abs(hata).mean():.2e}  medyan={np.median(np.abs(hata)):.2e}"
    f"  maks={np.abs(hata).max():.2e}"
)
iyi = la[:, 2] < 1e-6
print(f"  artik<1e-6 olan {iyi.sum()} dosyada |dL| maks={np.abs(hata[iyi]).max():.2e}")
for ad, ph, gh, ri in sorted(loo, key=lambda x: -abs(x[1] - x[2]))[:4]:
    print(f"    {ad:32s} tahmin {ph:+.5f} gercek {gh:+.5f} artik {ri:.1e}")

OUT["iddia1"] = dict(
    sum_w=float(w.sum()),
    artik=artik,
    L_g7=Lg,
    Q_g7=Qg,
    kazanc=float(Lg**2 / Qg),
    ongoru=float(np.sqrt(m0 - Lg**2 / Qg)),
    w_tekil_degil_bos_boyut=int(len(null)),
    L_sapmasi_cekirdek_yonunde=float(sap[:, 0].max()),
    span_disi_L_siniri=sinir,
    skor_band_yari=float(abs(band[0] - band[1])),
    loo_ortalama_hata=float(np.abs(hata).mean()),
    loo_maks_hata=float(np.abs(hata).max()),
)

# --- IDDIA 2: ucul sistem ---------------------------------------------------
ad3 = [
    "tuketim_g7_span_tau3.csv",
    "tuketim_y46_amnezik_kirpik.csv",
    "tuketim_y45_mevsimsel_kirpik.csv",
]
D3 = np.array([np.log1p(pd.read_csv(os.path.join(S, a)).tuketim.values) - a0 for a in ad3])
G3 = D3 @ D3.T / N
kos = G3 / np.sqrt(np.outer(np.diag(G3), np.diag(G3)))
print(f"\n[3-lu] cond(G)={np.linalg.cond(G3):.1f}  Q={np.diag(G3).round(4)}")
print("  kosinus:\n" + "\n".join("   " + " ".join(f"{v:+6.3f}" for v in s) for s in kos))
kal = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
en_k1 = 0.0
kotu = None
for r1 in (Lg / np.sqrt(Qg),):
    for r2 in kal:
        for r3 in kal:
            Lv = np.array([r1 * np.sqrt(G3[0, 0]), r2 * np.sqrt(G3[1, 1]), r3 * np.sqrt(G3[2, 2])])
            k = np.linalg.solve(G3, Lv)
            if np.abs(k).sum() > en_k1:
                en_k1, kotu = float(np.abs(k).sum()), (r2, r3, k.copy())
# negatif kalite senaryolari da (L isareti ters gelebilir)
en_k1n = 0.0
kotun = None
for r2 in [-x for x in kal] + kal:
    for r3 in [-x for x in kal] + kal:
        Lv = np.array([Lg, r2 * np.sqrt(G3[1, 1]), r3 * np.sqrt(G3[2, 2])])
        k = np.linalg.solve(G3, Lv)
        if np.abs(k).sum() > en_k1n:
            en_k1n, kotun = float(np.abs(k).sum()), (r2, r3, k.copy())
print(
    f"  |k|_1 maks (r>=0 senaryolari) {en_k1:.3f} @ r2={kotu[0]} r3={kotu[1]} k={kotu[2].round(3)}"
)
print(
    f"  |k|_1 maks (isaret serbest)   {en_k1n:.3f} @ r2={kotun[0]} r3={kotun[1]} k={kotun[2].round(3)}"
)
OUT["iddia2"] = dict(
    cond=float(np.linalg.cond(G3)),
    kosinus=kos.round(4).tolist(),
    k1_maks_pozitif=en_k1,
    k1_maks_isaret_serbest=en_k1n,
)

# --- IDDIA 4: kapi + basabas ------------------------------------------------
ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
kapilar = {}
for a in [
    "tuketim_y46_amnezik_kirpik.csv",
    "tuketim_y45_mevsimsel_kirpik.csv",
    "tuketim_g7_span_tau3.csv",
    TAB,
]:
    df = pd.read_csv(os.path.join(S, a))
    v = df.tuketim.values
    lg = np.log1p(v)
    d = lg - a0
    Qa = float((d**2).mean())
    kapilar[a] = dict(
        satir=len(df),
        kolon=list(df.columns),
        id_ss=bool((df.id.values == ss.iloc[:, 0].values).all()),
        id_test=bool((df.id.values == te.id.values).all()),
        nan=int(df.tuketim.isna().sum()),
        neg=int((v < 0).sum()),
        sonsuz=int((~np.isfinite(v)).sum()),
        maks=float(v.max()),
        ort=float(v.mean()),
        medyan=float(np.median(v)),
        sifir=int((v == 0).sum()),
        Q=Qa,
        basabas_skor=float(np.sqrt(m0 + Qa)),  # L=0 gelirse cikacak skor
    )
print("\n[kapi/basabas]")
for a, k in kapilar.items():
    print(
        f"  {a:34s} satir={k['satir']} idss={k['id_ss']} idte={k['id_test']} nan={k['nan']}"
        f" neg={k['neg']} maks={k['maks']:.0f} ort={k['ort']:.2f} sifir={k['sifir']}"
        f" Q={k['Q']:.4f} basabas={k['basabas_skor']:.5f}"
    )
OUT["iddia4"] = kapilar

# gereken kalite: skorun tabani gecmesi icin L > Q/2
for a in ["tuketim_y46_amnezik_kirpik.csv", "tuketim_y45_mevsimsel_kirpik.csv"]:
    Qa = kapilar[a]["Q"]
    print(
        f"  {a}: kendi skoru tabani gecsin diye gereken r=L/sqrt(Q) > {np.sqrt(Qa) / 2:.4f}"
        f"  (olculmus en iyi r=0.124) -> gerceklesmez, skor {kapilar[a]['basabas_skor']:.4f} civari"
    )

json.dump(OUT, open("w1_kirici_ham.json", "w"), indent=1, ensure_ascii=False)
print("\nyazildi w1_kirici_ham.json")
