"""KIRICI 2: w tekil degil mi, skor yuvarlamasi, m99 korkuluk atlatma, y45 kuyrugu."""

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
OUT = {}

# --- A) FARKLI COZUCULER ile w -> L(g7) yayilimi -----------------------------
# Kisit: sum(w)=1, hedef w@P ~ g. Rank eksik (22/25) -> sonsuz cozum.
# Her cozumu ARTIK ile birlikte raporla; artik ayni ise L ayni mi?
Gd = P @ P.T / N
M = np.block([[Gd, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
rhs = np.concatenate([P @ g / N, [1.0]])
sonuc = []
for etiket, wv in [
    ("lstsq rcond=None", np.linalg.lstsq(M, rhs, rcond=None)[0][:n]),
    ("lstsq rcond=1e-12", np.linalg.lstsq(M, rhs, rcond=1e-12)[0][:n]),
    ("lstsq rcond=1e-8", np.linalg.lstsq(M, rhs, rcond=1e-8)[0][:n]),
    ("lstsq rcond=1e-6", np.linalg.lstsq(M, rhs, rcond=1e-6)[0][:n]),
    ("pinv rcond=1e-10", (np.linalg.pinv(M, rcond=1e-10) @ rhs)[:n]),
]:
    sonuc.append(
        (
            etiket,
            float(((wv @ P - g) ** 2).mean()),
            float(wv @ L),
            float(np.abs(wv).sum()),
            float(wv.sum()),
        )
    )
# dogrudan D uzayinda (sum kisiti gereksiz): dg ~ sum u_i d_i
for etiket, rc in [
    ("D-uzayi rcond=None", None),
    ("D-uzayi rcond=1e-8", 1e-8),
    ("D-uzayi rcond=1e-6", 1e-6),
]:
    u = np.linalg.lstsq(D.T, dg, rcond=rc)[0]
    sonuc.append(
        (
            etiket,
            float(((u @ D - dg) ** 2).mean()),
            float(u @ L),
            float(np.abs(u).sum()),
            float(u.sum()),
        )
    )
print("cozucu                 artik        L(g7)      |w|_1    sum(w)")
for e_, r_, l_, a_, s_ in sonuc:
    print(f"  {e_:20s} {r_:.3e}  {l_:+.6f}  {a_:8.2f}  {s_:+.6f}")
Ls = np.array([x[2] for x in sonuc])
print(f"  -> L yayilimi {Ls.min():+.6f} .. {Ls.max():+.6f}  genislik {Ls.max() - Ls.min():.2e}")

# ES-ARTIK perturbasyon: artigi <=1.05x tutan w'ler arasinda L ne kadar kayar?
w0 = np.linalg.lstsq(M, rhs, rcond=None)[0][:n]
r0 = float(((w0 @ P - g) ** 2).mean())
# P(+1 satir) matrisinin sag cekirdegi = w yonunde etkisiz yonler
_, sv, vth = np.linalg.svd(np.vstack([Gd, np.ones((1, n))]), full_matrices=True)
k_eff = int((sv > sv[0] * 1e-10).sum())
Nspace = vth[k_eff:]
print(f"\n[es-artik] P(+1 satir) rank={k_eff}/{n}, cekirdek boyutu={n - k_eff}")
rng = np.random.default_rng(1)
en = (0.0, 0.0, 0.0)
for _ in range(4000):
    if n - k_eff == 0:
        break
    v = rng.normal(size=n - k_eff) @ Nspace
    for oc in np.logspace(-1, 4, 25):
        w2 = w0 + oc * v / np.linalg.norm(v)
        rr = float(((w2 @ P - g) ** 2).mean())
        if rr <= 1.05 * r0:
            dl = abs(float(w2 @ L) - float(w0 @ L))
            if dl > en[0]:
                en = (dl, rr, float(np.abs(w2).sum()))
print(
    f"  artik<=1.05x tutan w'ler arasinda |dL| maks = {en[0]:.3e}"
    f"  (artik {en[1]:.2e}, |w|_1 {en[2]:.1f})"
)

# --- B) SKOR YUVARLAMASI: LB 5 ondalik -> L gurultusu ------------------------
dL_tek = np.abs(np.array([skor[a] for a in adlar])) * 1e-5  # dm=2*s*5e-6 -> dL=s*5e-6*... ust sinir
dL_tek = np.array([skor[a] for a in adlar]) * 5e-6  # |dL_j| <= s_j*5e-6
kotu_gurultu = float(np.abs(w0) @ dL_tek)
print(
    f"\n[skor yuvarlama] her L_j'de +-{dL_tek.mean():.2e} -> L(g7)'de en kotu +-{kotu_gurultu:.2e}"
)
Lg = float(w0 @ L)
for d in (-kotu_gurultu, kotu_gurultu):
    print(f"   skor {np.sqrt(m0 - (Lg + d) ** 2 / Qg):.5f}")
OUT["skor_yuvarlama_dL"] = kotu_gurultu

# --- C) y45 kuyruk denetimi --------------------------------------------------
tr = None
for yol in ["data/raw/train.csv", "data/processed/train.parquet"]:
    p_ = os.path.join(KOK, yol)
    if os.path.exists(p_):
        tr = pd.read_parquet(p_) if yol.endswith("parquet") else pd.read_csv(p_, nrows=3000000)
        break
kol = None
if tr is not None:
    for c in tr.columns:
        if "tuket" in c.lower() or c.lower() in ("target", "y"):
            kol = c
            break
print("\n[y45 kuyruk]")
m6 = pd.read_csv(os.path.join(S, TAB)).tuketim.values
for a in ["tuketim_y45_mevsimsel_kirpik.csv", "tuketim_y46_amnezik_kirpik.csv"]:
    v = pd.read_csv(os.path.join(S, a)).tuketim.values
    print(
        f"  {a:34s} maks={v.max():>10.0f} p99.99={np.percentile(v, 99.99):>9.0f}"
        f" >m6maks: {(v > m6.max()).sum():>5d} adet  >1e5: {(v > 1e5).sum()}"
    )
if kol:
    y = tr[kol].values
    print(f"  TRAIN {kol}: maks={np.nanmax(y):.0f}  p99.99={np.nanpercentile(y, 99.99):.0f}")
    OUT["train_maks"] = float(np.nanmax(y))

# --- D) m99 KORKULUK ATLATMA -------------------------------------------------
import m99_coklu_coz as M99

print("\n[m99 korkuluk atlatma]")
notlar = []
# D1: YANLIS SKOR yazilirsa (Plan B'de g7'nin skoru UYDURULACAK) -> sessiz kabul
Lg_dogru = Lg
sk_g7 = float(np.sqrt(m0 + Qg - 2 * Lg_dogru))
print(f"  g7 icin m99'a yazilmasi gereken UYDURMA skor = {sk_g7:.6f}")
for yanlis, et in [
    (sk_g7, "dogru"),
    (sk_g7 - 0.001, "-0.001 typo"),
    (sk_g7 - 0.01, "-0.01 typo"),
    (1.00137, "kilavuzdaki yuvarlak"),
]:
    try:
        s_, k_ = M99.coz(
            TAB,
            1.00284,
            [("tuketim_g7_span_tau3.csv", yanlis), ("tuketim_y46_amnezik_kirpik.csv", 1.18)],
            yaz=False,
        )
        notlar.append((et, yanlis, float(s_), float(np.abs(k_).sum())))
    except SystemExit as ex:
        notlar.append((et, yanlis, None, str(ex)))
print("  skor-typo duyarliligi (ONGORULEN RMSLE / |k|_1):")
for et, v_, s_, k_ in notlar:
    print(f"    {et:22s} skor={v_:.6f} -> ongoru {s_}  |k|_1 {k_}")

# D2: kapi denetimi maks degeri KONTROL ETMIYOR + assert'ten ONCE dosya yaziliyor
kaynak = open(os.path.join(os.path.dirname(M99.__file__), "m99_coklu_coz.py")).read()
i_yaz = kaynak.index("out.to_csv")
i_ass = kaynak.index("assert (")
print(
    f"\n  m99: out.to_csv konumu {i_yaz} < assert konumu {i_ass} -> "
    f"KAPI PATLASA BILE DOSYA DISKTE KALIR: {i_yaz < i_ass}"
)
print(
    f"  m99 kapi 'maks' degerini yaziyor ama DENETLEMIYOR: "
    f"{'maks' in kaynak.split('assert (')[1].split(')')[0] is False}"
)
# D3: cozum dosyasinin gercek maks degeri kotu senaryoda ne olur?
a = np.log1p(pd.read_csv(os.path.join(S, TAB)).tuketim.values)
for et, ks in [
    ("k=[1.345,0.117,0.177] (tarama maks)", [1.345, 0.117, 0.177]),
    ("k=[2.0,0.5,0.5] (|k|1=3, korkuluk GECER)", [2.0, 0.5, 0.5]),
    ("k=[3.0,1.0,1.0] (|k|1=5, sinirda)", [3.0, 1.0, 1.0]),
]:
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
    p = a + np.array(ks) @ d3
    y = np.clip(np.expm1(p), 0, None)
    print(f"  {et:44s} maks={y.max():.3e}  ort={y.mean():.1f}  >1e6: {(y > 1e6).sum()}")

json.dump(OUT, open("w2_kirici_ham.json", "w"), indent=1, ensure_ascii=False)
print("\nbitti")
