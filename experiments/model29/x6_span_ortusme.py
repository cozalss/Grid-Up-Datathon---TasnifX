import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from x1_yukle import KOK, matris, oku

dos, X, s = matris()
adlar = [d.replace("tuketim_", "").replace(".csv", "") for d in dos]
N = X.shape[1]
m6 = oku(KOK / "submissions/tuketim_m6_ikiyon.csv")
Dm = X - m6  # 25 x N, m6 satiri sifir
U, S, Vt = np.linalg.svd(Dm @ Dm.T / N)  # 25x25
print("fark uzayi tekil degerleri (Gram ozdegeri):")
print(np.array2string(S, precision=4, suppress_small=False))
print(f"sayisal rank (S>1e-10*S0) = {(1e-10 * S[0] < S).sum()}")
adaylar = {
    "y46": "tuketim_y46_amnezik_kirpik.csv",
    "y45": "tuketim_y45_mevsimsel_kirpik.csv",
    "y40": "tuketim_y40_sota_temiz.csv",
    "g7": "tuketim_g7_span_tau3.csv",
    "b2_k15": "tuketim_b2_span_k15.csv",
    "m3": "tuketim_m3_hl1_capali.csv",
}
print(f"\n{'aday':8s} {'Q':>9s} " + " ".join(f"yeni%@r{r:<3d}" for r in [2, 5, 10, 15, 20, 24]))
for ad, f in adaylar.items():
    d = oku(KOK / "submissions" / f) - m6
    Q = (d**2).mean()
    rhs = Dm @ d / N
    sat = []
    for r in [2, 5, 10, 15, 20, 24]:
        c = U[:, :r] @ ((U[:, :r].T @ rhs) / S[:r])
        proj = Dm.T @ c
        sat.append(f"{100 * (1 - (proj**2).mean() / Q):9.1f}")
    print(f"{ad:8s} {Q:9.5f} " + " ".join(sat))
