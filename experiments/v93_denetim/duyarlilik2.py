"""v93 denetimi -- adim 5b: esik-tabanli rank ile LOO + zamansal tutma + public alt kume."""

from __future__ import annotations

import json
from pathlib import Path

import coz as C
import numpy as np

BURA = Path(__file__).resolve().parent
ESIK = 1e-8  # sv > ESIK * sv_max tutulur; 7.06e-5 ile 4.19e-12 arasini temiz keser


def rank_sec(s: np.ndarray, esik: float = ESIK) -> int:
    return int((s > esik * s[0]).sum())


def coz_alt(etiketler, d_hedef, maske=None, skor_ek=None, rank=None, lam=0.0):
    p0 = C.yukle(C.TABAN)
    if maske is None:
        n = p0.size
        sel = slice(None)
    else:
        sel = maske
        n = int(maske.sum())
    p0m = p0[sel]
    dh = d_hedef[sel]
    dig = [e for e in etiketler if e != C.TABAN]
    D = np.empty((len(dig), n))
    for i, e in enumerate(dig):
        D[i] = C.yukle(e)[sel] - p0m
    G = (D @ D.T) / n
    sk = {e: C.ENV[e]["skor"] + (skor_ek or {}).get(e, 0.0) for e in etiketler}
    m0 = sk[C.TABAN] ** 2
    m = np.array([sk[e] ** 2 for e in dig])
    b = (m0 + np.diag(G) - m) / 2.0
    s = np.linalg.svd(G, hermitian=True, compute_uv=False)
    r = rank if rank is not None else rank_sec(s)
    res = C.yansit(D, G, b, dh, n, rank=r, lam=lam)
    w, _, _ = C.coz(G, b, rank=r, lam=lam)
    mse = m0 + res["dMSE_bilinen"]
    return {
        "rmsle": float(np.sqrt(mse)) if mse > 0 else float("nan"),
        "mse": float(mse),
        "w_l1": float(np.abs(w).sum()),
        "artik": res["artik_orani"],
        "rank": r,
        "Q": res["Q"],
        "w": dict(zip(dig, w.tolist())),
        "b": b.tolist(),
        "dig": dig,
    }


def main() -> None:
    p0 = C.yukle(C.TABAN)
    d93 = C.yukle("v93") - p0
    tam = coz_alt(C.OLCULENLER, d93)
    print(f"TAM havuz: rank={tam['rank']} RMSLE={tam['rmsle']:.6f} |w|_1={tam['w_l1']:.3f}\n")

    print("=== HAT 2a: LOO (esik-tabanli rank) ===")
    print(f"{'cikarilan':>10} {'rank':>5} {'RMSLE':>10} {'sapma':>10} {'|w|_1':>9} {'artik%':>8}")
    loo = {}
    for e in C.OLCULENLER:
        if e == C.TABAN:
            continue
        r = coz_alt([x for x in C.OLCULENLER if x != e], d93)
        loo[e] = {k: r[k] for k in ("rmsle", "w_l1", "artik", "rank")}
        print(
            f"{e:>10} {r['rank']:>5} {r['rmsle']:>10.6f} {r['rmsle'] - tam['rmsle']:>+10.6f} "
            f"{r['w_l1']:>9.3f} {r['artik']:>7.2%}"
        )
    v = np.array([x["rmsle"] for x in loo.values()])
    print(
        f"\nLOO: min={v.min():.6f} max={v.max():.6f} ARALIK={v.max() - v.min():.6f} sd={v.std():.6f}"
    )

    print("\n=== HAT 5: ZAMANSAL TUTMA (ilk N gonderimle coz, sonrakileri TAHMIN et) ===")
    sirali = sorted(C.OLCULENLER, key=lambda e: C.ENV[e]["ref"])
    print("gonderim sirasi:", " ".join(sirali))
    zaman = []
    for N in (8, 10, 12, 14, 16):
        egit = sirali[:N]
        if C.TABAN not in egit:
            egit = egit + [C.TABAN]
        hedefler = [e for e in sirali[N:] if e != C.TABAN]
        if not hedefler:
            continue
        print(f"\n  N={N}  (egitim: {len(egit)} dosya, sinav: {len(hedefler)})")
        hat = []
        for h in hedefler:
            dh = C.yukle(h) - p0
            r = coz_alt(egit, dh)
            gercek = C.ENV[h]["skor"]
            print(
                f"    {h:>5}: tahmin={r['rmsle']:.5f} gercek={gercek:.5f} "
                f"hata={r['rmsle'] - gercek:+.5f}  artik={r['artik']:.1%}"
            )
            hat.append(r["rmsle"] - gercek)
        hat = np.array(hat)
        print(
            f"    -> ort hata={hat.mean():+.5f}  |hata| ort={np.abs(hat).mean():.5f} "
            f"  {int((hat < 0).sum())}/{len(hat)} ASIRI IYIMSER"
        )
        zaman.append(
            {
                "N": N,
                "ort_hata": float(hat.mean()),
                "mutlak": float(np.abs(hat).mean()),
                "iyimser": int((hat < 0).sum()),
                "adet": len(hat),
            }
        )

    print("\n=== HAT 4: PUBLIC ALT KUME -- KOSEGEN TERIMININ TAM HESABI ===")
    dig = tam["dig"]
    D = np.empty((len(dig), p0.size))
    for i, e in enumerate(dig):
        D[i] = C.yukle(e) - p0
    n = p0.size
    G = (D @ D.T) / n
    d2 = D**2
    Gii_sd_birim = d2.std(axis=1)  # std_j(d_ij^2)
    rng = np.random.default_rng(7)
    c_vec, _, _ = C.coz(G, (D @ d93) / n, rank=tam["rank"])
    print("eps_i = (G_ii^tam - G_ii^public)/2 ; tahmin hatasi = 2*c.eps")
    print(f"{'f':>6} {'n_pub':>8} {'sd(eps) tip':>12} {'sd(2c.eps)MSE':>14} {'sd RMSLE':>10}")
    pub = []
    for f in (0.05, 0.20, 0.30, 0.50):
        npub = f * n
        # analitik: sd(G_ii^pub) = sd_j(d_ij^2)/sqrt(npub) * sqrt(1-f)
        sd_eps = Gii_sd_birim / (2 * np.sqrt(npub)) * np.sqrt(1 - f)
        # 2*c.eps varyansi: eps'ler korelasyonlu -> ampirik bootstrap
        orn = []
        for _ in range(200):
            msk = rng.random(n) < f
            Gp = np.einsum("ij,ij->i", D[:, msk], D[:, msk]) / msk.sum()
            eps = (np.diag(G) - Gp) / 2
            orn.append(2 * float(c_vec @ eps))
        orn = np.array(orn)
        pub.append(
            {
                "f": f,
                "sd_mse": float(orn.std()),
                "sd_rmsle": float(orn.std() / (2 * 1.0083)),
                "sd_eps_tipik": float(np.median(sd_eps)),
            }
        )
        print(
            f"{f:>6.2f} {int(npub):>8} {np.median(sd_eps):>12.3e} "
            f"{orn.std():>14.3e} {orn.std() / (2 * 1.0083):>10.3e}"
        )

    (BURA / "duyarlilik2.json").write_text(
        json.dumps(
            {
                "tam": {k: tam[k] for k in ("rmsle", "w_l1", "rank", "artik")},
                "loo": loo,
                "loo_aralik": float(v.max() - v.min()),
                "zamansal": zaman,
                "public": pub,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
