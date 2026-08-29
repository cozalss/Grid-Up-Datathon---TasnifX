"""r3: r_span'in dogrulanmasi + kuresel kayma teshisi + r1_artik.json birlestirme.

DENETIM 1  <r_span, d_j>/N =? -L_j   (kesik SVD nedeniyle tam olmaz;
           artik, atilan ozyonlere dusen paydir -- buyuklugu raporlanir)
DENETIM 2  kuresel kayma ort(r_span) k ile nasil degisiyor?
           (dusuk k'de teshisin isaret cevirmesinin sebebi budur)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

BURA = Path(__file__).resolve().parent
KOK = BURA.parents[1]
SKOR = json.loads((BURA / "olculmus_skorlar.json").read_text(encoding="utf-8"))
TABAN = "tuketim_m6_ikiyon.csv"
meta = json.loads((BURA / "g1_meta.json").read_text(encoding="utf-8"))
dosyalar = meta["dosyalar"]
X = np.load(BURA / "g1_X.npy")
N = X.shape[1]
i0 = dosyalar.index(TABAN)
p0 = X[i0]
M0 = SKOR[TABAN] ** 2
jj = [j for j in range(len(dosyalar)) if j != i0]
adlar = [dosyalar[j].replace("tuketim_", "").replace(".csv", "") for j in jj]
D = X[jj] - p0
n = D.shape[0]
G = (D @ D.T) / N
Qd = np.diag(G).copy()
mj = np.array([SKOR[dosyalar[j]] ** 2 for j in jj])
L = (M0 + Qd - mj) / 2.0
w, V = np.linalg.eigh(G)
o = np.argsort(-w)
w, V = w[o], V[:, o]
Lt = V.T @ L
r = np.load(BURA / "r1_rspan.npy")

print("== DENETIM 1: <r_span, d_j>/N =? -L_j  (k=15) ==")
ic = (D @ r) / N
print("%-24s %11s %11s %11s %9s" % ("yon", "-L_j (olculen)", "<r_s,d>/N", "fark", "fark/|L|"))
den1 = []
for j in np.argsort(-np.abs(L)):
    f = ic[j] - (-L[j])
    print(
        "%-24s %+11.6f %+11.6f %+11.6f %9.3f"
        % (adlar[j][:24], -L[j], ic[j], f, abs(f) / max(abs(L[j]), 1e-12))
    )
    den1.append(dict(yon=adlar[j], eksi_L=float(-L[j]), ic=float(ic[j]), fark=float(f)))
print("ort |fark| = %.3e ; ort |L| = %.3e" % (np.abs(ic + L).mean(), np.abs(L).mean()))

print("\n== DENETIM 2: kuresel kayma ort(r_span) ve enerji dagilimi ==")
kur = {}
for k in [5, 8, 10, 12, 13, 15, 17, 18, 20]:
    a = V[:, :k] @ (Lt[:k] / w[:k])
    rk = -(a @ D)
    kaz = float(a @ L)
    kur[str(k)] = dict(
        ort=float(rk.mean()),
        sd=float(rk.std()),
        kazanc=kaz,
        ort_kare_pay=float(rk.mean() ** 2 / max(kaz, 1e-15)),
    )
    print(
        "k=%2d  ort %+9.6f  sd %8.6f  kazanc %.6f  ort^2/kazanc %.4f"
        % (k, rk.mean(), rk.std(), kaz, rk.mean() ** 2 / max(kaz, 1e-15))
    )

print("\nYORUM: dusuk k'de ort(r_span) buyuk NEGATIF -> cozum enerjisini")
print("kuresel bir seviye kaymasina harciyor; kohort ORTALAMALARI bu")
print("kaymayi tasir. KONTRASTLAR (kohortlar arasi fark) ise kaymadan")
print("bagimsizdir ve cok daha kararlidir.")

# ---- birlestir
art = json.loads((BURA / "r1_artik.json").read_text(encoding="utf-8"))
aday = json.loads((BURA / "r2_aday.json").read_text(encoding="utf-8"))
art["denetim_ic_carpim"] = den1
art["kuresel_kayma"] = kur
art["adaylar"] = aday["adaylar"]
art["kontrast_kararliligi"] = aday["kontrast"]
art["q_hedef_aday"] = aday["q_hedef"]
json.dump(art, open(BURA / "r1_artik.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\nr1_artik.json GUNCELLENDI (denetim + kuresel kayma + adaylar)")
