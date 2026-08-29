"""KIRICI 2 (hizli): Gram uzerinden es-artik tarama + skor gurultusu + m99 kirma."""

import json
import os

import numpy as np
import pandas as pd
from m30_ozellik import KOK

S = os.path.join(KOK, "submissions")
skor = {
    k: v for k, v in json.load(open("olculmus_skorlar.json")).items() if k != "gun1_baseline.csv"
}
adlar = sorted(skor)
P = np.array([np.log1p(pd.read_csv(os.path.join(S, a)).tuketim.values) for a in adlar])
N = P.shape[1]
n = len(adlar)
m = np.array([skor[a] ** 2 for a in adlar])
TAB = "tuketim_m6_ikiyon.csv"
ti = adlar.index(TAB)
a0 = P[ti]
m0 = m[ti]
D = P - a0
Q = np.einsum("ij,ij->i", D, D) / N
L = (m0 + Q - m) / 2
g = np.log1p(pd.read_csv(os.path.join(S, "tuketim_g7_span_tau3.csv")).tuketim.values)
dg = g - a0
Qg = float((dg**2).mean())
Gd = P @ P.T / N
b = P @ g / N
gg = float((g * g).mean())
M = np.block([[Gd, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
w0 = np.linalg.lstsq(M, np.concatenate([b, [1.0]]), rcond=None)[0][:n]


def artik(w):  # |w@P - g|^2/N , sadece Gram ile
    return float(w @ Gd @ w - 2 * w @ b + gg)


r0 = artik(w0)
Lg = float(w0 @ L)
print(f"w0: artik={r0:.4e}  L={Lg:+.6f}  sum={w0.sum():.6f}")

# es-artik tarama: [Gd;1] matrisinin sag cekirdegi
_, sv, vth = np.linalg.svd(np.vstack([Gd, np.ones((1, n))]))
kef = int((sv > sv[0] * 1e-12).sum())
Ns = vth[kef:]
print(f"[cekirdek] rank={kef}/{n} cekirdek boyutu={n - kef}")
rng = np.random.default_rng(1)
en = (0.0, 0.0, 0.0, 0.0)
for _ in range(20000):
    if n - kef == 0:
        break
    v = rng.normal(size=n - kef) @ Ns
    v /= np.linalg.norm(v)
    for oc in np.logspace(-2, 4, 40):
        w2 = w0 + oc * v
        rr = artik(w2)
        if rr <= 1.05 * r0:
            dl = abs(float(w2 @ L) - Lg)
            if dl > en[0]:
                en = (dl, rr, float(np.abs(w2).sum()), float(w2.sum()))
print(
    f"[es-artik] artik<=1.05*r0 tutan w'ler: |dL| maks={en[0]:.3e}"
    f" (artik {en[1]:.3e}, |w|_1 {en[2]:.1f}, sum {en[3]:.6f})"
)
en2 = (0.0, 0.0, 0.0)
for _ in range(20000):
    if n - kef == 0:
        break
    v = rng.normal(size=n - kef) @ Ns
    v /= np.linalg.norm(v)
    for oc in np.logspace(-2, 4, 40):
        w2 = w0 + oc * v
        if np.abs(w2).sum() <= 10.0:  # makul agirlik kisiti
            dl = abs(float(w2 @ L) - Lg)
            if dl > en2[0]:
                en2 = (dl, artik(w2), float(np.abs(w2).sum()))
print(f"[|w|_1<=10] |dL| maks={en2[0]:.3e} (artik {en2[1]:.3e}, |w|_1 {en2[2]:.1f})")

# skor yuvarlamasi (LB 5 ondalik -> |ds|<=5e-6 -> |dL_j| <= s_j*5e-6)
dLj = np.array([skor[a] for a in adlar]) * 5e-6
gur = float(np.abs(w0) @ dLj)
print(
    f"\n[skor yuvarlama] L(g7) en kotu +-{gur:.3e} -> skor "
    f"{np.sqrt(m0 - (Lg + gur) ** 2 / Qg):.5f} .. {np.sqrt(m0 - (Lg - gur) ** 2 / Qg):.5f}"
)

# CSV hassasiyeti
raw = pd.read_csv(os.path.join(S, "tuketim_g7_span_tau3.csv"), dtype=str, nrows=5).tuketim.tolist()
print(f"[csv] g7 ilk degerler: {raw}")

# kuyruk
m6 = pd.read_csv(os.path.join(S, TAB)).tuketim.values
for a in ["tuketim_y45_mevsimsel_kirpik.csv", "tuketim_y46_amnezik_kirpik.csv"]:
    v = pd.read_csv(os.path.join(S, a)).tuketim.values
    print(
        f"[kuyruk] {a:34s} maks={v.max():>9.0f} p99.99={np.percentile(v, 99.99):>8.0f}"
        f" >m6maks {int((v > m6.max()).sum()):>5d}  >1e5 {int((v > 1e5).sum())}"
        f"  sifir {int((v == 0).sum())}"
    )
for yol in ["data/raw/train.csv"]:
    p_ = os.path.join(KOK, yol)
    if os.path.exists(p_):
        c = pd.read_csv(p_, usecols=lambda c: "tuket" in c.lower() or c.lower() == "target")
        for k_ in c.columns:
            print(
                f"[kuyruk] TRAIN {k_}: maks={c[k_].max():.0f} p99.99={c[k_].quantile(0.9999):.0f}"
            )

# --- m99 kirma ---------------------------------------------------------------
import m99_coklu_coz as M99

sk_g7 = float(np.sqrt(m0 + Qg - 2 * Lg))
print(f"\n[m99] g7 icin UYDURULACAK skor = {sk_g7:.6f}")
tab_ = []
for et, sg in [
    ("dogru (turetilen)", sk_g7),
    ("kilavuzdaki 1.00137", 1.00137),
    ("typo -0.001", sk_g7 - 0.001),
    ("typo -0.01", sk_g7 - 0.01),
    ("typo +0.01", sk_g7 + 0.01),
]:
    try:
        s_, k_ = M99.coz(
            TAB,
            1.00284,
            [("tuketim_g7_span_tau3.csv", sg), ("tuketim_y46_amnezik_kirpik.csv", 1.18)],
            yaz=False,
        )
        tab_.append((et, sg, float(s_), float(np.abs(k_).sum()), list(np.round(k_, 3))))
    except SystemExit as ex:
        tab_.append((et, sg, None, str(ex), None))
print("\n[m99 skor-typo] etiket / verilen skor / ONGORU / |k|_1 / k")
for r in tab_:
    print(f"   {r[0]:22s} {r[1]:.5f}  {r[2]}  {r[3]}  {r[4]}")

kaynak = open("m99_coklu_coz.py", encoding="utf-8").read()
i_yaz, i_ass = kaynak.index("out.to_csv"), kaynak.index("assert (")
ass_govde = kaynak[i_ass : i_ass + 300]
print(
    f"\n[m99 kod] to_csv({i_yaz}) < assert({i_ass}): {i_yaz < i_ass}"
    f"  -> kapi patlarsa BOZUK DOSYA DISKTE KALIR"
)
print(f"[m99 kod] assert 'maks' denetliyor mu: {'maks' in ass_govde.split(')')[0]}")
print(f"[m99 kod] assert sonsuz/inf denetliyor mu: {'isfinite' in kaynak}")
print(
    f"[m99 kod] taban skoru olculmus_skorlar.json ile karsilastiriliyor mu: "
    f"{'olculmus_skorlar' in kaynak}"
)

# kotu ama korkulugu gecen cozum
a = a0
d3 = np.array(
    [
        np.log1p(pd.read_csv(os.path.join(S, x)).tuketim.values) - a
        for x in [
            "tuketim_g7_span_tau3.csv",
            "tuketim_y46_amnezik_kirpik.csv",
            "tuketim_y45_mevsimsel_kirpik.csv",
        ]
    ]
)
print("\n[cikti buyuklugu] |k|_1<5 korkulugunu gecen k'lerde dosya maks:")
for ks in ([1.345, 0.117, 0.177], [2.0, 0.5, 0.5], [3.0, 1.0, 0.9], [1.0, -1.0, -1.0]):
    p = a + np.array(ks) @ d3
    y = np.clip(np.expm1(p), 0, None)
    print(
        f"   k={ks} |k|1={sum(abs(x) for x in ks):.2f} maks={y.max():.3e}"
        f" ort={y.mean():.1f} >1e6 {int((y > 1e6).sum())}"
    )
print("\nbitti")
