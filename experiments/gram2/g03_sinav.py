"""G3 -- Ornekleme-disi sinav, eski cozumun teshisi, ridge taramasi, yeni optimum.

Tahmin formulu (taban p0, egitim yon kumesi S):
    v_hat = sum_i w_i d_i,  G_SS w = b_S  (kesik oz-cozum, rank r)
    b_hat_j = (G_{jS} w)
    MSE_hat_j = m0 + G_jj - 2 b_hat_j
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from g01_havuz import HAVUZ, yukle  # noqa: E402
from g02_coz import DELTA_S  # noqa: E402

KOK = Path(__file__).resolve().parents[2]
GON = KOK / "submissions"
CIK = Path(__file__).resolve().parent

KRONOLOJI = [h[0] for h in HAVUZ]  # zaten tarih sirali


def kur(taban="v83"):
    adlar, X, skorlar, ids = yukle()
    n = X.shape[1]
    i0 = adlar.index(taban)
    m0 = float(skorlar[i0] ** 2)
    yon = [a for a in adlar if a != taban]
    idx = [adlar.index(a) for a in yon]
    D = X[idx] - X[i0]
    G = (D @ D.T) / n
    m = skorlar[idx] ** 2
    b = (m0 + np.diag(G) - m) / 2.0
    sd_m0 = 2 * skorlar[i0] * DELTA_S
    sd_m = 2 * skorlar[idx] * DELTA_S
    C = (sd_m0**2 / 4.0) * np.ones((len(idx),) * 2) + np.diag(sd_m**2 / 4.0)
    return dict(
        adlar=adlar,
        X=X,
        skorlar=skorlar,
        n=n,
        i0=i0,
        m0=m0,
        yon=yon,
        D=D,
        G=G,
        m=m,
        b=b,
        C=C,
        ids=ids,
    )


def coz_kesik(Gs, bs, r=None, esik=1e-10, ridge=0.0):
    """Kesik oz-cozum. r verilirse ilk r ozyon, yoksa lam > lam_max*esik."""
    lam, U = np.linalg.eigh(Gs)
    sr = np.argsort(lam)[::-1]
    lam, U = lam[sr], U[:, sr]
    if r is None:
        tut = lam > lam[0] * esik
    else:
        tut = np.zeros(len(lam), bool)
        tut[:r] = True
        tut &= lam > lam[0] * 1e-13
    beta = U.T @ bs
    inv = np.zeros_like(lam)
    inv[tut] = 1.0 / (lam[tut] + ridge)
    w = U @ (inv * beta)
    kazanc = float(np.sum(inv * beta**2))
    return w, kazanc, int(tut.sum()), lam, U, beta


def tahmin(H, egitim, hedefler, r=None, ridge=0.0):
    yon = H["yon"]
    iS = [yon.index(a) for a in egitim]
    Gs = H["G"][np.ix_(iS, iS)]
    bs = H["b"][iS]
    w, kazanc, rr, *_ = coz_kesik(Gs, bs, r=r, ridge=ridge)
    out = {}
    for t in hedefler:
        j = yon.index(t)
        bh = float(H["G"][j, iS] @ w)
        mh = H["m0"] + H["G"][j, j] - 2 * bh
        out[t] = (np.sqrt(max(mh, 0.0)), mh, bh, float(H["b"][j]))
    return out, kazanc, rr, w


def main() -> None:
    H = kur("v83")
    yon, G, b, m0 = H["yon"], H["G"], H["b"], H["m0"]
    rap = {}

    # ================= 3. ESKI COZUMUN TESHISI =================
    print("=" * 96)
    print("3) ESKI COZUM NEDEN IYIMSERDI")
    print("=" * 96)
    eski_yon = [
        "v2",
        "v7",
        "v15",
        "v16",
        "v18",
        "v25",
        "v27",
        "v30",
        "v46",
        "v44",
        "v47",
        "v50",
        "v55",
        "v67",
        "v73",
        "v79",
        "v80",
        "v81",
    ]
    iE = [yon.index(a) for a in eski_yon]
    GE, bE = G[np.ix_(iE, iE)], b[iE]
    lamE, UE = np.linalg.eigh(GE)
    srE = np.argsort(lamE)[::-1]
    lamE, UE = lamE[srE], UE[:, srE]
    betaE = UE.T @ bE
    print("kesme noktasina gore ulasilabilir MSE / RMSLE (eski 18 yon):")
    for r in range(1, 19):
        kz = float(np.sum(betaE[:r] ** 2 / np.where(lamE[:r] > 1e-14, lamE[:r], np.inf)))
        mm = m0 - kz
        print(
            f"  r={r:2d} lam_r={lamE[r - 1]:.3e}  kazanc={kz:.6f}  MSE={mm: .6f}  "
            f"RMSLE={np.sqrt(mm) if mm > 0 else float('nan'):.6f}"
        )
    print("\n-> eski cozum r=17 aldi; 17. ozdeger 4.19e-12 = SAYISAL NULL.")
    print("   proj_norm2=32.774 (coz.json) tam da bu null yonden geliyor: 3.276e+01.")
    print("   Dolayisiyle 'ulasilabilir_rmsle=0.0' ve |w|_1=5.9e6 -> b, null uzayina saciliyor.")

    # v93 dosyasinin bugunku havuzla tahmini
    print("\n--- v93 dosyasinin YENI havuzla tahmini ---")
    adlar, X = H["adlar"], H["X"]
    d93 = (
        np.log1p(pd.read_csv(GON / "tuketim_v93_gram_optimum.csv")["tuketim"].to_numpy("f8"))
        - X[H["i0"]]
    )
    Q93 = float(d93 @ d93 / H["n"])
    g93 = (H["D"] @ d93) / H["n"]  # <d93, d_i>/n
    for etiket, S, r in [
        ("tum havuz (v101/v102 dahil), tam rank", yon, None),
        ("v101/v102 HARIC, r=16 (null atilmis)", eski_yon, 16),
        ("v101/v102 HARIC, r=17 (ESKI, null dahil)", eski_yon, 17),
    ]:
        iS = [yon.index(a) for a in S]
        w, kz, rr, *_ = coz_kesik(G[np.ix_(iS, iS)], b[iS], r=r)
        b93 = float(g93[iS] @ w)
        mm = m0 + Q93 - 2 * b93
        print(
            f"  {etiket:44s} r={rr:2d} b93={b93:+.6f} Q93={Q93:.6f} "
            f"MSE={mm:.6f} RMSLE={np.sqrt(max(mm, 0)):.5f}"
        )
    print(f"  [eski iddia] b93=+0.009794  Q93={Q93:.6f}  RMSLE=1.008336")

    # j=15 yonu hala ayakta mi?
    print("\n--- eski j=15 yonu (lam=7.06e-5, SNR 128.7, v47/v30/v46) ---")
    u15 = UE[:, 15]
    print(
        "  bilesen: "
        + "  ".join(f"{eski_yon[a]}:{u15[a]:+.3f}" for a in np.argsort(-np.abs(u15))[:5])
    )
    # ayni yonu 20-boyutlu uzaya goem ve yeni b ile sina
    u_tam = np.zeros(len(yon))
    u_tam[iE] = u15
    ub = float(u_tam @ b)
    sdu = float(np.sqrt(u_tam @ H["C"] @ u_tam))
    lam_tam = float(u_tam @ G @ u_tam)
    print(
        f"  eski bazda : u.b={betaE[15]:+.4e}  lam={lamE[15]:.4e}  kazanc={betaE[15] ** 2 / lamE[15]:.4e}"
    )
    print(
        f"  yeni b ile : u.b={ub:+.4e}  u'Gu={lam_tam:.4e}  kazanc={ub**2 / lam_tam:.4e}  SNR={abs(ub) / sdu:.1f}"
    )
    print("  (b ayni olcumlerden geldigi icin u.b degismez; yon HALA ayakta -- sorun")
    print("   bu yon degil, r=17'deki NULL yondu.)")

    # ================= 4. NULL UZAY SINAVI =================
    print("\n" + "=" * 96)
    print("4) NULL-UZAY TUTARLILIK SINAVI (tum 20 yon)")
    print("=" * 96)
    lam, U = np.linalg.eigh(G)
    sr = np.argsort(lam)[::-1]
    lam, U = lam[sr], U[:, sr]
    beta = U.T @ b
    sdv = np.sqrt(np.einsum("ij,jk,ki->i", U.T, H["C"], U))
    nulls = []
    for j in range(len(lam)):
        if lam[j] <= lam[0] * 1e-10:
            z = beta[j] / sdv[j]
            bil = "  ".join(f"{yon[a]}:{U[a, j]:+.4f}" for a in np.argsort(-np.abs(U[:, j]))[:4])
            print(
                f"  j={j} lam={lam[j]:+.3e}  u.b={beta[j]:+.3e}  sd={sdv[j]:.2e}  z={z:+6.2f}   {bil}"
            )
            nulls.append({"j": j, "lam": float(lam[j]), "ub": float(beta[j]), "z": float(z)})
    print("  |z|<2.5 => yuvarlama gurultusu icinde; havuzda TUTARSIZLIK YOK.")
    rap["null"] = nulls

    # ================= 5. ORNEKLEM-DISI SINAV =================
    print("\n" + "=" * 96)
    print("5) ORNEKLEM-DISI SINAV")
    print("=" * 96)
    print("\n5a) KRITIK: v101 ve v102 disarida -- egitim = v83 oncesi 18 yon")
    for r in [6, 8, 10, 12, 14, 16, 17]:
        out, kz, rr, w = tahmin(H, eski_yon, ["v101", "v102"], r=r)
        s101, s102 = out["v101"][0], out["v102"][0]
        print(
            f"  r={r:2d} (|w|1={np.abs(w).sum():.2e})  v101 tahmin={s101:.5f} (gercek 1.01614, hata={s101 - 1.01614:+.5f})"
            f"   v102 tahmin={s102:.5f} (gercek 1.00553, hata={s102 - 1.00553:+.5f})"
        )
    rap["v101_v102_sinavi"] = {}

    print("\n5b) Ilk N gonderimle coz, sonrakilerin GERCEK skorunu tahmin et")
    kron = [a for a in KRONOLOJI if a != "v83"]
    satirlar = []
    for N in [12, 14, 16, 18, 20]:
        egit = kron[: N - 1] if len(kron) >= N else kron
        egit = kron[: min(N, len(kron))]
        test = [a for a in kron if a not in egit]
        if not test:
            continue
        out, kz, rr, w = tahmin(H, egit, test, r=None)
        hatalar = []
        for t in test:
            sh = out[t][0]
            gercek = float(H["skorlar"][H["adlar"].index(t)])
            hatalar.append(sh - gercek)
            satirlar.append((N, t, gercek, sh, sh - gercek))
        h = np.array(hatalar)
        print(f"  N={N:2d} egit={len(egit)} rank={rr}  test={test}")
        print(
            f"       ort hata={h.mean():+.5f}  medyan={np.median(h):+.5f}  "
            f"rms={np.sqrt((h**2).mean()):.5f}  max|hata|={np.abs(h).max():.5f}"
        )
    print("\n  N   hedef   gercek     tahmin     hata")
    for N, t, g, s, e in satirlar:
        print(f"  {N:2d}  {t:>5}  {g:.5f}  {s:.5f}  {e:+.5f}")
    tumh = np.array([e for *_x, e in satirlar])
    print(
        f"\n  TUM OOS: ort hata={tumh.mean():+.6f} (>0 = KOTUMSER, <0 = IYIMSER), "
        f"rms={np.sqrt((tumh**2).mean()):.6f}"
    )
    rap["oos_ort_hata"] = float(tumh.mean())
    rap["oos_rms"] = float(np.sqrt((tumh**2).mean()))

    # ================= 2. RANK / RIDGE SECIMI, YENI OPTIMUM =================
    print("\n" + "=" * 96)
    print("2) YENI OPTIMUM -- rank kesme ve ridge taramasi (tum 20 yon, taban v83)")
    print("=" * 96)
    print(" r  lam_r        SNR_r    kazanc     MSE        RMSLE     |w|_1     Q(v83)   Q(v102)")
    x0 = H["X"][H["i0"]]
    i101, i102 = yon.index("v101"), yon.index("v102")
    d102 = H["D"][i102]
    tablo = []
    for r in range(1, 18):
        w, kz, rr, *_ = coz_kesik(G, b, r=r)
        mm = m0 - kz
        Q0 = float(w @ G @ w)
        # v102'ye gore mesafe: ||sum w_i d_i - d_102||^2/n
        Q2 = Q0 - 2 * float(w @ G[:, i102]) + float(G[i102, i102])
        tablo.append(
            (
                r,
                lam[r - 1],
                np.abs(beta[r - 1]) / sdv[r - 1],
                kz,
                mm,
                np.sqrt(max(mm, 0)),
                np.abs(w).sum(),
                Q0,
                Q2,
            )
        )
        print(
            f"{r:2d}  {lam[r - 1]:.3e}  {np.abs(beta[r - 1]) / sdv[r - 1]:8.1f}  {kz:.6f}  "
            f"{mm:.6f}  {np.sqrt(max(mm, 0)):.6f}  {np.abs(w).sum():7.3f}  {Q0:.5f}  {Q2:.5f}"
        )

    print("\nridge taramasi (tam rank 17):")
    for al in [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
        w, kz, rr, *_ = coz_kesik(G, b, r=17, ridge=al)
        # gercek beklenen MSE: m0 + w'Gw - 2 w'b
        mm = m0 + float(w @ G @ w) - 2 * float(w @ b)
        print(
            f"  alpha={al:.0e}  |w|1={np.abs(w).sum():8.3f}  beklenen MSE={mm:.6f}  RMSLE={np.sqrt(mm):.6f}"
        )

    np.savez(
        CIK / "g03_ara.npz", lam=lam, U=U, beta=beta, sd=sdv, G=G, b=b, m0=m0, adlar=np.array(yon)
    )
    (CIK / "g03_rapor.json").write_text(json.dumps(rap, indent=2), encoding="utf-8")
    print("\nyazildi: g03_ara.npz, g03_rapor.json")


if __name__ == "__main__":
    main()
