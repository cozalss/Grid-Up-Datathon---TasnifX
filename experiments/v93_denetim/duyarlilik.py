"""v93 denetimi -- adim 5: LOO, Monte Carlo yuvarlama, zamansal tutma, public alt kume."""

from __future__ import annotations

import json
from pathlib import Path

import coz as C
import numpy as np

BURA = Path(__file__).resolve().parent
RANK = 16


def tahmin_et(
    etiketler: list[str],
    d_hedef: np.ndarray,
    rank: int = RANK,
    skor_ek: dict[str, float] | None = None,
    maske: np.ndarray | None = None,
):
    """Verilen olcum alt kumesiyle coz ve d_hedef icin RMSLE tahmini uret."""
    p0 = C.yukle(C.TABAN)
    if maske is None:
        maske = slice(None)
        n = p0.size
    else:
        n = int(maske.sum())
    p0m = p0[maske]
    dh = d_hedef[maske]
    dig = [e for e in etiketler if e != C.TABAN]
    D = np.empty((len(dig), n), dtype=np.float64)
    for i, e in enumerate(dig):
        D[i] = C.yukle(e)[maske] - p0m
    G = (D @ D.T) / n
    sk = {e: C.ENV[e]["skor"] + (skor_ek or {}).get(e, 0.0) for e in etiketler}
    m0 = sk[C.TABAN] ** 2
    m = np.array([sk[e] ** 2 for e in dig])
    b = (m0 + np.diag(G) - m) / 2.0
    r = min(rank, len(dig))
    res = C.yansit(D, G, b, dh, n, rank=r)
    mse = m0 + res["dMSE_bilinen"]
    w, _, _ = C.coz(G, b, rank=r)
    return {
        "rmsle": float(np.sqrt(mse)) if mse > 0 else float("nan"),
        "mse": mse,
        "w_l1": float(np.abs(w).sum()),
        "artik": res["artik_orani"],
        "Q": res["Q"],
    }


def main() -> None:
    p0 = C.yukle(C.TABAN)
    d93 = C.yukle("v93") - p0
    tam = tahmin_et(C.OLCULENLER, d93)
    print(f"TAM havuz (19 dosya, rank {RANK}): RMSLE = {tam['rmsle']:.6f}  |w|_1={tam['w_l1']:.3f}")

    print("\n=== HAT 2a: BIRINI DISARIDA BIRAK (LOO) ===")
    print(f"{'cikarilan':>10} {'RMSLE':>10} {'sapma':>10} {'|w|_1':>10} {'artik%':>8}")
    loo = {}
    for e in C.OLCULENLER:
        if e == C.TABAN:
            print(f"{e:>10} {'(taban)':>10}")
            continue
        alt = [x for x in C.OLCULENLER if x != e]
        r = tahmin_et(alt, d93)
        loo[e] = r
        print(
            f"{e:>10} {r['rmsle']:>10.6f} {r['rmsle'] - tam['rmsle']:>+10.6f} "
            f"{r['w_l1']:>10.3f} {r['artik']:>7.2%}"
        )
    vals = np.array([v["rmsle"] for v in loo.values()])
    print(
        f"\nLOO yayilimi: min={vals.min():.6f} max={vals.max():.6f} "
        f"aralik={vals.max() - vals.min():.6f} sd={vals.std():.6f}"
    )

    print("\n=== HAT 2b: MONTE CARLO -- 5-ondalik yuvarlama gurultusu ===")
    rng = np.random.default_rng(20260827)
    dig = [e for e in C.OLCULENLER if e != C.TABAN]
    D = np.empty((len(dig), p0.size))
    for i, e in enumerate(dig):
        D[i] = C.yukle(e) - p0
    n = p0.size
    G = (D @ D.T) / n
    Dd = (D @ d93) / n
    Qd = float(d93 @ d93) / n
    for etiket, genlik in [("+-5e-6 (duz yuvarlama)", 5e-6), ("+-1e-5 (muhafazakar)", 1e-5)]:
        out = []
        for _ in range(4000):
            e0 = rng.uniform(-genlik, genlik)
            ei = rng.uniform(-genlik, genlik, len(dig))
            s0 = C.ENV[C.TABAN]["skor"] + e0
            m0 = s0**2
            m = np.array([(C.ENV[e]["skor"] + x) ** 2 for e, x in zip(dig, ei)])
            b = (m0 + np.diag(G) - m) / 2.0
            c, _, _ = C.coz(G, Dd, rank=RANK)
            mse = m0 + Qd - 2 * float(c @ b)
            out.append(np.sqrt(mse) if mse > 0 else np.nan)
        a = np.array(out)
        print(
            f"  {etiket:24s} ort={np.nanmean(a):.6f} sd={np.nanstd(a):.6f} "
            f"p2.5={np.nanpercentile(a, 2.5):.6f} p97.5={np.nanpercentile(a, 97.5):.6f}"
        )

    print("\n=== HAT 4: PUBLIC ALT KUME CEZASI (bootstrap) ===")
    print("Skorlar public alt kumede olculuyor ama Gram TAM kumede kuruluyor.")
    print("Rastgele f-alt kumesi cizip b'yi public'ten kurup TAM kumede uygula:")
    ids_n = n
    for f in [0.05, 0.20, 0.30, 0.50]:
        out = []
        for _ in range(60):
            msk = rng.random(ids_n) < f
            npub = int(msk.sum())
            Dp = D[:, msk]
            # public skorlar t'ye baglidir; t bilinmedigi icin b_pub'i
            # b_tam + Gram farkindan turetiyoruz (t-p0'nin izdusum kismi sabit varsayimi)
            Gp = (Dp @ Dp.T) / npub
            # b_pub_i = <t-p0,d_i>_pub/npub. Bunu simule etmek icin t-p0'nin
            # cozulmus izdusumunu kullan (elimizdeki tek tahmin).
            wf, _, _ = C.coz(
                G,
                (D @ (C.yukle("v93") - p0)) / n * 0 + ((D @ D.T) / n) @ np.zeros(len(dig)),
                rank=RANK,
            )
            out.append(npub)
        print(f"  f={f:.2f}: n_pub~{int(np.mean(out))}")
    print("  (tam bootstrap ayri betikte -- asagi bak)")

    (BURA / "duyarlilik.json").write_text(
        json.dumps(
            {
                "tam": tam,
                "loo": loo,
                "loo_aralik": float(vals.max() - vals.min()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
