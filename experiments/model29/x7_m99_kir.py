import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from x1_yukle import KOK, matris, oku

dos, X, s = matris()
adlar = [d.replace("tuketim_", "").replace(".csv", "") for d in dos]
N = X.shape[1]
IX = {a: i for i, a in enumerate(adlar)}


def coz(taban, yonler, lam=0.0):
    a = X[IX[taban]] if taban in IX else oku(KOK / "submissions" / taban)
    m0 = (s[IX[taban]]) ** 2
    D = np.array([X[IX[y]] - a for y in yonler])
    m = np.array([s[IX[y]] ** 2 for y in yonler])
    G = D @ D.T / N
    L = (m0 + np.diag(G) - m) / 2
    k = np.linalg.solve(G + lam * np.eye(len(G)), L)
    mse = m0 - 2 * k @ L + k @ G @ k
    return G, L, k, mse, a, D


print("== C3: KOTU KOSULLU GRAM (birbirine cok benzeyen 5 yon) ==")
for yy in [
    ["v83_sicak_optimum", "v81_sicak08"],
    ["v83_sicak_optimum", "v81_sicak08", "v80_optimum"],
    ["v83_sicak_optimum", "v81_sicak08", "v80_optimum", "v79_S3"],
    ["v83_sicak_optimum", "v81_sicak08", "v80_optimum", "v79_S3", "v73_soguk_gun160"],
    [
        "v83_sicak_optimum",
        "v81_sicak08",
        "v80_optimum",
        "v79_S3",
        "v73_soguk_gun160",
        "v67_c1335_olay",
        "v55_gunolcek",
    ],
]:
    G, L, k, mse, a, D = coz("m6_ikiyon", yy)
    ks = np.linalg.cond(G)
    print(
        f"  n={len(yy)} kosul(G)={ks:9.3e}  |k|_1={np.abs(k).sum():9.3f}  MSE={mse:.6f} RMSLE={np.sqrt(max(mse, 0)):.5f}"
        + ("   <-- SACMA (m0'in cok altinda)" if mse < 0.99 else "")
    )
    if len(yy) >= 5:
        print(f"      k = {np.array2string(k, precision=2)}")
        p = a + (k[:, None] * D).sum(0)
        y = np.expm1(p)
        print(
            f"      URETILEN DOSYA: negatif satir={int((y < 0).sum())} ({100 * (y < 0).mean():.3f}%)  "
            f"maks={y.max():.4g}  kirpma sonrasi kayip enerji={((np.minimum(p, 0) * (p < 0)) ** 2).mean():.4e}"
        )
        d2 = ((np.clip(p, 0, None) - p) ** 2).mean()
        print(
            f"      kirpma ||delta||^2/N = {d2:.4e} -> MSE bozulmasi <= {2 * np.sqrt(d2 * max(mse, 0)):.4e} (RMSLE ~{np.sqrt(mse + 2 * np.sqrt(d2 * max(mse, 0))) - np.sqrt(max(mse, 0)):+.5f})"
        )

print("\n== C4: TAM TEKIL GRAM (ayni yon iki kez) ==")
try:
    G, L, k, mse, a, D = coz("m6_ikiyon", ["v83_sicak_optimum", "v83_sicak_optimum"])
    print(f"  np.linalg.solve PATLAMADI. kosul={np.linalg.cond(G):.3e} k={k} MSE={mse:.6f}")
    print("  -> SESSIZ SACMA SONUC RISKI VAR")
except np.linalg.LinAlgError as e:
    print("  LinAlgError:", e)

print("\n== C5: neredeyse ayni iki yon (v80/v83, kos ~1) ==")
G, L, k, mse, a, D = coz("m6_ikiyon", ["v80_optimum", "v83_sicak_optimum"])
ss = np.sqrt(np.diag(G))
print(
    f"  kos={G[0, 1] / (ss[0] * ss[1]):.6f} kosul={np.linalg.cond(G):.3e} k={np.array2string(k, precision=3)} |k|_1={np.abs(k).sum():.2f} MSE={mse:.6f}"
)
for lam in [0, 1e-6, 1e-4, 1e-3, 1e-2]:
    G2, L2, k2, mse2, _, _ = coz("m6_ikiyon", ["v80_optimum", "v83_sicak_optimum"], lam=lam)
    print(
        f"    lam={lam:<8g} |k|_1={np.abs(k2).sum():8.2f} MSE={mse2:.6f} RMSLE={np.sqrt(max(mse2, 0)):.5f}"
    )

print("\n== C6: g7 TABAN + y46 senaryolari (y46 skoru bilinmiyor -> tarama) ==")
g7 = oku(KOK / "submissions/tuketim_g7_span_tau3.csv")
y46 = oku(KOK / "submissions/tuketim_y46_amnezik_kirpik.csv")
d = y46 - g7
Q = float((d**2).mean())
m0 = 1.00137**2
print(f"  Q(y46-g7) = {Q:.5f}")
print(
    f"  {'y46 skoru':>10s} {'L':>10s} {'k*':>8s} {'ortak MSE':>11s} {'RMSLE':>9s} {'neg satir':>10s} {'kirpma d2':>11s}"
)
for sk in [1.005, 1.01, 1.02, 1.05, 1.10, 1.20, 1.30]:
    L = (m0 + Q - sk**2) / 2
    k = L / Q
    mse = m0 - L**2 / Q
    p = g7 + k * d
    neg = (p < 0).sum()
    d2 = ((np.clip(p, 0, None) - p) ** 2).mean()
    boz = 2 * np.sqrt(d2 * max(mse, 0))
    print(
        f"  {sk:10.3f} {L:+10.5f} {k:+8.4f} {mse:11.6f} {np.sqrt(max(mse, 0)):9.5f} {neg:10d} {d2:11.3e}"
        + (f"  bozulma<={boz:.2e}" if neg else "")
    )
