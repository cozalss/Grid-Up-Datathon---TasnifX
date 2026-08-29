import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from x1_yukle import KOK, matris, oku

dos, X, s = matris()
adlar = [d.replace("tuketim_", "").replace(".csv", "") for d in dos]
N = X.shape[1]
m6 = oku(KOK / "submissions/tuketim_m6_ikiyon.csv")
y46 = oku(KOK / "submissions/tuketim_y46_amnezik_kirpik.csv")
g7 = oku(KOK / "submissions/tuketim_g7_span_tau3.csv")
m4 = X[adlar.index("m4_hava_capali")]
v102 = X[adlar.index("v102_kappa_optimum")]
d46 = y46 - m6
print("== y46 YON OLCUMLERI (taban m6) ==")
print(f"  Q = ||d||^2/N = {(d46**2).mean():.6f}   (docs: 0.38844)")
print(
    f"  kurtoz(d) = {stats.kurtosis(d46, fisher=False):.3f} (Fisher-disi)  Fisher={stats.kurtosis(d46):.3f}"
)
sq = d46**2
idx = np.argsort(sq)[::-1]
print(f"  Q'nun en kotu %1 satirdan gelen payi = {sq[idx[: N // 100]].sum() / sq.sum():.4f}")
print(
    f"  kos(d46, m4-m6) = {np.dot(d46, m4 - m6) / np.linalg.norm(d46) / np.linalg.norm(m4 - m6):+.4f}"
)
print(
    f"  kos(d46, m4-v102) = {np.dot(d46, m4 - v102) / np.linalg.norm(d46) / np.linalg.norm(m4 - v102):+.4f}  (docs: -0.115)"
)
print(
    f"  kos(d46, g7-m6) = {np.dot(d46, g7 - m6) / np.linalg.norm(d46) / np.linalg.norm(g7 - m6):+.4f}"
)
# span'a dikligi: d46'yi 25 dosyanin fark uzayina yansit
Dm = X - m6
Gm = Dm @ Dm.T / N
rhs = Dm @ d46 / N
c, *_ = np.linalg.lstsq(Gm, rhs, rcond=None)
proj = Dm.T @ c
print(
    f"  d46'nin OLCULMUS SPAN'a dusen payi = {(proj**2).mean() / (d46**2).mean():.4f}  -> YENI bilgi %{100 * (1 - (proj**2).mean() / (d46**2).mean()):.1f}"
)
print("\n== 'tabana esitlendi' iddiasi ==")
esit = (np.abs(y46 - m6) < 1e-12).sum()
print(f"  y46 == m6 birebir satir = {esit}  (%{100 * esit / N:.3f})")
for ad, x in [
    ("y45", oku(KOK / "submissions/tuketim_y45_mevsimsel_kirpik.csv")),
    ("y40", oku(KOK / "submissions/tuketim_y40_sota_temiz.csv")),
    ("y31_amnezik", oku(KOK / "submissions/tuketim_y31_amnezik.csv")),
    ("y41_amnezik_temiz", oku(KOK / "submissions/tuketim_y41_amnezik_temiz.csv")),
]:
    print(
        f"  {ad:20s} m6 ile birebir satir = {(np.abs(x - m6) < 1e-12).sum():7d}   Q={((x - m6) ** 2).mean():.6f}"
    )

print("\n== LOO FALSIFIKASYON: her dosyanin skoru digerlerinden ongorulebilir mi? ==")
m = s**2
D = np.empty((25, 25))
for i in range(25):
    D[i] = ((X - X[i]) ** 2).mean(1)
E = (m[:, None] + m[None, :] - D) / 2
sonuc = []
for i in range(25):
    oth = [j for j in range(25) if j != i]
    # x_i'yi digerlerinin AFIN kombinasyonu ile en iyi yaklas
    A = X[oth]
    w, *_ = np.linalg.lstsq(A.T, X[i], rcond=None)
    w = w / w.sum()
    r = X[i] - A.T @ w
    art = (r**2).mean()
    Es = E[np.ix_(oth, oth)]
    ong = w @ Es @ w  # artik yok sayilarak ALT sinir
    sonuc.append((adlar[i], m[i], ong, art, np.sqrt(max(ong, 0)), s[i]))
print(f"  {'dosya':22s} {'gercek':>9s} {'ongoru(art=0)':>13s} {'artik':>10s} {'ongoru+artik':>13s}")
for a, mi, og, ar, _, si in sonuc:
    print(
        f"  {a:22s} {si:9.5f} {np.sqrt(max(og, 0)):13.5f} {ar:10.4f} {np.sqrt(max(og + ar, 0)):13.5f}"
    )
