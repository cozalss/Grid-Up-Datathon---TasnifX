import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from x1_yukle import KOK, matris, oku

dos, X, s = matris()
adlar = [d.replace("tuketim_", "").replace(".csv", "") for d in dos]
N = X.shape[1]
g7 = oku(KOK / "submissions/tuketim_g7_span_tau3.csv")
bey = json.loads((Path(__file__).parent / "g3_aday.json").read_text(encoding="utf-8"))["tau=3"]["w"]
wb = np.zeros(25)
for k, v in bey.items():
    wb[adlar.index(k)] = v
p = X.T @ wb
neg = p < 0
print("== KIRPMA YONU ==")
print(
    f"  kirpilan satir sayisi {neg.sum()}, bu satirlarda g7 log1p degeri: min={g7[neg].min():.3e} maks={g7[neg].max():.3e}"
)
print(f"  ham p bu satirlarda: min={p[neg].min():.5f} maks={p[neg].max():.5f}")
print(f"  -> hepsi TAM SIFIR mi? {bool((g7[neg] == 0).all())}")
print("  gercek t>=0 oldugundan kirpma hatayi SADECE AZALTIR.")
# kazanc araligi
d2 = ((g7 - p) ** 2).mean()
m = s**2
D = np.empty((25, 25))
for i in range(25):
    D[i] = ((X - X[i]) ** 2).mean(1)
E = (m[:, None] + m[None, :] - D) / 2
mse_ham = wb @ E @ wb
# ust sinir: 2|p_i| t_i ; t_i <= makul tavan (tum dosyalarin max log1p)
tmax = X[:, neg].max(0)
ust = (p[neg] ** 2 - 2 * p[neg] * tmax).sum() / N
print(f"  kazanc >= {d2:.3e} (MSE), <= {ust:.3e} (t_i<=diger dosyalarin maks tahmini)")
print(
    f"  => gercek MSE in [{mse_ham - ust:.8f}, {mse_ham - d2:.8f}]  RMSLE [{np.sqrt(mse_ham - ust):.7f}, {np.sqrt(mse_ham - d2):.7f}]"
)

print("\n== E TANIMSIZLIGI (skor tutarsizligi) ==")
lam, V = np.linalg.eigh(E)
print("  en kucuk 5 ozdeger:", np.array2string(lam[:5], precision=3))
neg_i = lam < 0
print(f"  negatif ozdeger sayisi = {neg_i.sum()}, toplam negatif enerji = {lam[lam < 0].sum():.3e}")
# afin kisit altinda tanimsizlik somurusu: |w|_1<=3, sum w=1
v = V[:, 0]
if abs(v.sum()) > 1e-12:
    pass
print(f"  en kucuk ozvektorun bilesen toplami = {v.sum():.4f}, |v|_1={np.abs(v).sum():.3f}")
# olcekle: sum w=1 saglayan w = wb + c*u, u toplami 0
u = v - v.sum() / 25
u = u / np.abs(u).sum()  # |u|_1 = 1
for c in [1, 3, 10, 33]:
    print(
        f"    wb + {c}*u : |w|_1={np.abs(wb + c * u).sum():7.2f}  ongoru MSE={(wb + c * u) @ E @ (wb + c * u):.8f}"
    )

print("\n== TUTARLILIK: her skor tek tek dogrulanabilir mi? ==")
# ucgen esitsizligi: |sqrt(m_i)-sqrt(m_j)| <= ||x_i-x_j||/sqrt(N) <= sqrt(m_i)+sqrt(m_j)
kotu = []
for i in range(25):
    for j in range(i + 1, 25):
        dij = np.sqrt(D[i, j])
        a = np.sqrt(m[i])
        b = np.sqrt(m[j])
        if dij < abs(a - b) - 1e-9 or dij > a + b + 1e-9:
            kotu.append((adlar[i], adlar[j], dij, abs(a - b), a + b))
print(f"  ucgen esitsizligi ihlali: {len(kotu)}")
for k in kotu[:10]:
    print("   ", k)
# en gergin ciftler
gerg = []
for i in range(25):
    for j in range(i + 1, 25):
        dij = np.sqrt(D[i, j])
        a = np.sqrt(m[i])
        b = np.sqrt(m[j])
        gerg.append((dij - abs(a - b), adlar[i], adlar[j], dij, abs(a - b)))
gerg.sort()
print("  en gergin 5 cift (dij - |ri-rj|):")
for g in gerg[:5]:
    print(f"    {g[0]:+.5f}  {g[1]} / {g[2]}  d={g[3]:.5f} |dr|={g[4]:.5f}")
