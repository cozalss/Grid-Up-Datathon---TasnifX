"""v93 denetimi -- adim 2: Gram cozumunu SIFIRDAN yeniden kur ve v93'u sina."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BURA = Path(__file__).resolve().parent
ONB = BURA / "onbellek"
ENV = json.loads((BURA / "envanter.json").read_text(encoding="utf-8"))

TABAN = "v83"
OLCULENLER = [k for k, v in ENV.items() if v["skor"] is not None]


def yukle(etiket: str) -> np.ndarray:
    return np.load(ONB / f"{etiket}.npy")


def gram_kur(etiketler: list[str], taban: str = TABAN):
    """d_i = log1p(p_i) - log1p(p_taban) icin G, b dondur."""
    p0 = yukle(taban)
    n = p0.size
    digerleri = [e for e in etiketler if e != taban]
    D = np.empty((len(digerleri), n), dtype=np.float64)
    for i, e in enumerate(digerleri):
        D[i] = yukle(e) - p0
    G = (D @ D.T) / n
    m0 = ENV[taban]["skor"] ** 2
    m = np.array([ENV[e]["skor"] ** 2 for e in digerleri])
    b = (m0 + np.diag(G) - m) / 2.0
    return digerleri, D, G, b, m0, n


def coz(G: np.ndarray, b: np.ndarray, rank: int | None = None, lam: float = 0.0):
    """Rank-kisitli / ridge pseudo-inverse cozumu."""
    U, s, Vt = np.linalg.svd(G, hermitian=True)
    if rank is None:
        tol = s.max() * len(s) * np.finfo(float).eps
        rank = int((s > tol).sum())
    sinv = np.zeros_like(s)
    sinv[:rank] = s[:rank] / (s[:rank] ** 2 + lam**2) if lam > 0 else 1.0 / s[:rank]
    w = (Vt.T * sinv) @ (U.T @ b)
    return w, s, rank


def yansit(
    D: np.ndarray,
    G: np.ndarray,
    b: np.ndarray,
    d_hedef: np.ndarray,
    n: int,
    rank: int | None = None,
    lam: float = 0.0,
):
    """d_hedef yonunun beklenen dMSE'sini hesapla.

    dMSE = ||d||^2/n - 2 <t-p0,d>/n.
    <t-p0,d>/n yalnizca d'nin span{d_i} icindeki bileseninden BILINIR.
    d = D^T c + r  (r span'a dik) ise <t-p0,d>/n = c'b + <t-p0,r>/n (BILINMEYEN).
    """
    Q = float(d_hedef @ d_hedef) / n
    rhs = (D @ d_hedef) / n  # G-uzayinda d_hedef'in izdusum saglari
    c, s, r_used = coz(G, rhs, rank=rank, lam=lam)
    ic = float(c @ b)  # <t-p0, d_span>/n
    # artik (span'a dik bilesenin) normu
    span_kismi = D.T @ c
    artik = d_hedef - span_kismi
    artik_norm2 = float(artik @ artik) / n
    return {
        "Q": Q,
        "ic_carpim": ic,
        "dMSE_bilinen": Q - 2 * ic,
        "artik_norm2": artik_norm2,
        "artik_orani": float(np.sqrt(artik_norm2 / Q)) if Q > 0 else 0.0,
        "c": c,
        "rank": r_used,
    }


def main() -> None:
    digerleri, D, G, b, m0, n = gram_kur(OLCULENLER)
    print(f"n={n}  taban={TABAN}  m0={m0:.8f}  yon sayisi={len(digerleri)}")
    U, s, Vt = np.linalg.svd(G, hermitian=True)
    print("\nGram tekil degerleri:")
    for i, si in enumerate(s):
        print(f"  {i:2d}  {si:.6e}")
    tol = s.max() * len(s) * np.finfo(float).eps
    rank = int((s > tol).sum())
    print(f"\nsayisal rank (tol={tol:.3e}) = {rank}")
    print(f"kosul sayisi (rank icinde) = {s[0] / s[rank - 1]:.3e}")

    w, _, _ = coz(G, b, rank=rank)
    print(f"\ncozum |w|_1 = {np.abs(w).sum():.4f}   |w|_inf = {np.abs(w).max():.4f}")
    for e, wi in sorted(zip(digerleri, w), key=lambda t: -abs(t[1])):
        print(f"   {e:5s} {wi:+9.4f}")
    kalinti = G @ w - b
    print(f"normal denklem kalintisi = {np.abs(kalinti).max():.3e}")

    # tam izdusum: t-p0'nin span icindeki bileseni
    proj_norm2 = float(w @ b)  # ||P(t-p0)||^2/n
    print(
        f"\n||P(t-p0)||^2/n = {proj_norm2:.8f}  -> ulasilabilir en iyi MSE = {m0 - proj_norm2:.8f}"
        f"  RMSLE = {np.sqrt(max(m0 - proj_norm2, 0)):.6f}"
    )

    d93 = yukle("v93") - yukle(TABAN)
    r93 = yansit(D, G, b, d93, n, rank=rank)
    print("\n=== v93 ===")
    print(f"Q = ||d93||^2/n            = {r93['Q']:.8f}")
    print(f"<t-p0,d93>/n (span kismi)  = {r93['ic_carpim']:.8f}")
    print(
        f"span'a DIK artik ||r||^2/n = {r93['artik_norm2']:.3e}"
        f"  (||r||/||d93|| = {r93['artik_orani']:.4%})"
    )
    print(f"beklenen dMSE              = {r93['dMSE_bilinen']:+.8f}")
    mse93 = m0 + r93["dMSE_bilinen"]
    print(f"beklenen MSE               = {mse93:.8f}")
    print(f"beklenen RMSLE             = {np.sqrt(max(mse93, 0)):.6f}")

    d85 = yukle("v85") - yukle(TABAN)
    r85 = yansit(D, G, b, d85, n, rank=rank)
    mse85 = m0 + r85["dMSE_bilinen"]
    print("\n=== v85 (gonderilmemis, referans) ===")
    print(
        f"Q={r85['Q']:.8f} artik_orani={r85['artik_orani']:.4%} "
        f"beklenen RMSLE={np.sqrt(max(mse85, 0)):.6f}"
    )

    cikti = {
        "n": n,
        "m0": m0,
        "yonler": digerleri,
        "tekil_degerler": s.tolist(),
        "rank": rank,
        "w": {e: float(x) for e, x in zip(digerleri, w)},
        "w_l1": float(np.abs(w).sum()),
        "proj_norm2": proj_norm2,
        "ulasilabilir_rmsle": float(np.sqrt(max(m0 - proj_norm2, 0))),
        "v93": {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in r93.items()},
        "v93_rmsle": float(np.sqrt(max(mse93, 0))),
        "v85_rmsle": float(np.sqrt(max(mse85, 0))),
    }
    (BURA / "coz.json").write_text(
        json.dumps(cikti, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
