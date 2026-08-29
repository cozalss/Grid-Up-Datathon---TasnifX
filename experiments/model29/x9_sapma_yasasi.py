import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from x1_yukle import KOK, matris, oku

dos, X, s = matris()
adlar = [d.replace("tuketim_", "").replace(".csv", "") for d in dos]
N = X.shape[1]
m = s**2
D = np.empty((25, 25))
for i in range(25):
    D[i] = ((X - X[i]) ** 2).mean(1)
E = (m[:, None] + m[None, :] - D) / 2
print("LOO: ongoru hatasi |w|_1 ve SAPMA (log-uzayi yer degistirmesi) ile buyuyor mu?")
print(
    f"{'dosya':22s} {'gercek':>8s} {'ongoru':>8s} {'hata':>10s} {'|w|_1':>8s} {'artik':>9s} {'sapma':>7s}"
)
sat = []
for i in range(25):
    oth = [j for j in range(25) if j != i]
    A = X[oth]
    w, *_ = np.linalg.lstsq(A.T, X[i], rcond=None)
    w = w / w.sum()
    r = X[i] - A.T @ w
    art = float((r**2).mean())
    og = float(w @ E[np.ix_(oth, oth)] @ w)
    if art > 1e-6:
        continue
    # sapma: en yakin tek dosyaya gore degil, agirlik merkezine gore yer degistirme
    sapma = float(np.sqrt(((A.T @ w - X[oth][np.argmax(np.abs(w))]) ** 2).mean()))
    hata = np.sqrt(max(og, 0)) - s[i]
    sat.append((adlar[i], s[i], np.sqrt(max(og, 0)), hata, np.abs(w).sum(), art, sapma))
for a, g, o, h, w1, ar, sp in sorted(sat, key=lambda t: -abs(t[3])):
    print(f"{a:22s} {g:8.5f} {o:8.5f} {h:+10.2e} {w1:8.3f} {ar:9.2e} {sp:7.4f}")
h = np.array([t[3] for t in sat])
w1 = np.array([t[4] for t in sat])
sp = np.array([t[6] for t in sat])
print(
    f"\nn={len(sat)}  hata RMS = {np.sqrt((h**2).mean()):.3e}   maks|hata| = {np.abs(h).max():.3e}"
)
print(f"  korelasyon(|hata|, |w|_1) = {np.corrcoef(np.abs(h), w1)[0, 1]:+.3f}")
print(f"  korelasyon(|hata|, sapma) = {np.corrcoef(np.abs(h), sp)[0, 1]:+.3f}")
print(
    f"  docs/56 iddiasi sd=9.28e-5 ; g3_aday.json bandi 7.53e-5 -> olculen {np.sqrt((h**2).mean()):.2e}"
)
# g7 ve y46 icin sapma
g7 = oku(KOK / "submissions/tuketim_g7_span_tau3.csv")
y46 = oku(KOK / "submissions/tuketim_y46_amnezik_kirpik.csv")
m6 = X[adlar.index("m6_ikiyon")]
print("\nSAPMA KARSILASTIRMASI (m6 tabanina, log-uzayi RMS):")
print(f"  g7 (HAK1)                      : {np.sqrt(((g7 - m6) ** 2).mean()):.4f}")
print(f"  LOO'daki en buyuk dogrulanmis  : {sp.max():.4f}")
print(f"  y46 (HAK2, ham dosya)          : {np.sqrt(((y46 - m6) ** 2).mean()):.4f}")
for k in [0.24, 0.37, 0.49]:
    print(
        f"  HAK3 g7+{k:.2f}*(y46-g7)          : {np.sqrt(((g7 + k * (y46 - g7) - m6) ** 2).mean()):.4f}"
    )
