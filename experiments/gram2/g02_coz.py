"""G2 -- Gram/b cozumu, rank-SNR tablosu, null-uzay sinavi, eski cozumun teshisi.

Model:
    y   = gercek log1p hedef (bilinmiyor)
    x_i = i'nci gonderimin log1p vektoru
    p0  = taban (v83), m0 = skor(v83)^2
    d_i = x_i - x_0
    G_ij = <d_i, d_j>/n
    MSE_i = m0 + G_ii - 2 b_i   =>   b_i = (m0 + G_ii - MSE_i)/2
    w = G^{-1} b  =>  ulasilabilir MSE = m0 - b'G^{-1}b

Yuvarlama gurultusu: skorlar 5 ondaliga yuvarli.
    s ~ U(s_hat +- 5e-6) => sd(s) = 1e-5/sqrt(12) = 2.8868e-6
    m = s^2 => sd(m_i) = 2*s_i*sd(s)
    b_i = (m0 + G_ii - m_i)/2 => Cov(b) = (sd(m0)^2/4)*J + diag(sd(m_i)^2/4)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from g01_havuz import yukle  # noqa: E402

CIK = Path(__file__).resolve().parent
DELTA_S = 1e-5 / np.sqrt(12.0)  # 5 ondalik yuvarlamanin sd'si


def gram_kur(taban: str = "v83"):
    adlar, X, skorlar, ids = yukle()
    n = X.shape[1]
    i0 = adlar.index(taban)
    m0 = float(skorlar[i0] ** 2)
    yon_adlar = [a for a in adlar if a != taban]
    idx = [adlar.index(a) for a in yon_adlar]
    D = X[idx] - X[i0]  # (K-1) x n
    G = (D @ D.T) / n
    m = skorlar[idx] ** 2
    b = (m0 + np.diag(G) - m) / 2.0
    # gurultu kovaryansi
    sd_m0 = 2 * skorlar[i0] * DELTA_S
    sd_m = 2 * skorlar[idx] * DELTA_S
    C = (sd_m0**2 / 4.0) * np.ones((len(idx), len(idx))) + np.diag(sd_m**2 / 4.0)
    return dict(
        adlar=adlar,
        X=X,
        skorlar=skorlar,
        n=n,
        taban=taban,
        m0=m0,
        yon_adlar=yon_adlar,
        D=D,
        G=G,
        m=m,
        b=b,
        C=C,
        s=skorlar[idx],
    )


def ozbaz(H):
    lam, U = np.linalg.eigh(H["G"])
    sira = np.argsort(lam)[::-1]
    return lam[sira], U[:, sira]


def main() -> None:
    H = gram_kur("v83")
    G, b, C = H["G"], H["b"], H["C"]
    ad = H["yon_adlar"]
    K = len(ad)
    m0 = H["m0"]
    print("=" * 92)
    print(f"G2 -- GRAM COZUMU  taban={H['taban']}  m0={m0:.8f}  K={K} yon  n={H['n']}")
    print("=" * 92)

    lam, U = ozbaz(H)
    beta = U.T @ b
    sd = np.sqrt(np.einsum("ij,jk,ki->i", U.T, C, U))
    snr = np.abs(beta) / sd
    rank_sayisal = int((lam > lam[0] * 1e-12).sum())

    print(f"\nG rank (lam > lam_max*1e-12): {rank_sayisal} / {K}")
    print(f"kosul sayisi (rank icinde): {lam[0] / lam[rank_sayisal - 1]:.3e}")
    print(
        "\n j   ozdeger      beta=u.b      sd(u.b)     SNR      beta^2/lam   kum.kazanc  baskin yonler"
    )
    kum = 0.0
    for j in range(K):
        katki = beta[j] ** 2 / lam[j] if lam[j] > lam[0] * 1e-12 else np.nan
        if np.isfinite(katki):
            kum += katki
        agir = np.argsort(-np.abs(U[:, j]))[:3]
        et = " ".join(f"{ad[a]}({U[a, j]:+.2f})" for a in agir)
        print(
            f"{j:2d}  {lam[j]:.4e}  {beta[j]:+.3e}  {sd[j]:.2e}  {snr[j]:8.1f}  "
            f"{katki: .3e}  {kum: .4f}   {et}"
        )

    # ---- NULL UZAY SINAVI ----
    print("\n" + "-" * 92)
    print("NULL-UZAY TUTARLILIK SINAVI  (lam ~ 0 => sum u_i d_i = 0 => u.b = 0 olmali)")
    print("-" * 92)
    null_j = [j for j in range(K) if lam[j] <= lam[0] * 1e-10]
    if not null_j:
        print("sayisal null yon yok (G tam rank).")
    for j in null_j:
        z = beta[j] / sd[j]
        agir = np.argsort(-np.abs(U[:, j]))[:5]
        et = "  ".join(f"{ad[a]}: {U[a, j]:+.4f}" for a in agir)
        print(f"j={j:2d} lam={lam[j]:.3e}  u.b={beta[j]:+.3e}  sd={sd[j]:.2e}  z={z:+8.2f}")
        print(f"      {et}")

    # ---- ESKI COZUMUN TESHISI ----
    print("\n" + "=" * 92)
    print("ESKI COZUMUN TESHISI (v93 / reports+experiments/v93_denetim/coz.json)")
    print("=" * 92)
    eski = json.loads(
        (Path(__file__).resolve().parents[1] / "v93_denetim" / "coz.json").read_text(
            encoding="utf-8"
        )
    )
    eski_yon = eski["yonler"]
    print(f"eski yon kumesi ({len(eski_yon)}): {eski_yon}")
    print(
        f"eski rank={eski['rank']}, |w|_1={eski['w_l1']:.3e}, iddia edilen ulasilabilir RMSLE={eski['ulasilabilir_rmsle']}"
    )
    print(f"eski v93 on-kayit RMSLE={eski['v93_rmsle']:.6f}, v85 on-kayit={eski['v85_rmsle']:.6f}")

    # eski yon kumesiyle ayni cozumu bugunku havuzla tekrar kur (v101/v102 HARIC)
    idx_eski = [ad.index(a) for a in eski_yon if a in ad]
    Ge = G[np.ix_(idx_eski, idx_eski)]
    be = b[idx_eski]
    Ce = C[np.ix_(idx_eski, idx_eski)]
    le, Ue = np.linalg.eigh(Ge)
    sr = np.argsort(le)[::-1]
    le, Ue = le[sr], Ue[:, sr]
    bre = Ue.T @ be
    sde = np.sqrt(np.einsum("ij,jk,ki->i", Ue.T, Ce, Ue))
    print("\neski yon kumesinin oz-analizi (bugunku olcumlerle):")
    print(" j   ozdeger      beta        sd         SNR     beta^2/lam   kum.kazanc")
    kum = 0.0
    for j in range(len(le)):
        k = bre[j] ** 2 / le[j] if le[j] > le[0] * 1e-12 else np.nan
        if np.isfinite(k):
            kum += k
        print(
            f"{j:2d}  {le[j]:.4e}  {bre[j]:+.3e}  {sde[j]:.2e}  {np.abs(bre[j]) / sde[j]:8.1f}  {k: .3e}  {kum: .4f}"
        )
    print(f"\nm0 - toplam kazanc = {m0 - kum:.6f}  (negatifse model kendini yalanliyor)")

    np.savez(
        CIK / "g02_gram.npz",
        G=G,
        b=b,
        C=C,
        lam=lam,
        U=U,
        beta=beta,
        sd=sd,
        m0=m0,
        adlar=np.array(ad),
        m=H["m"],
        s=H["s"],
    )
    print(f"\nyazildi: {CIK / 'g02_gram.npz'}")


if __name__ == "__main__":
    main()
