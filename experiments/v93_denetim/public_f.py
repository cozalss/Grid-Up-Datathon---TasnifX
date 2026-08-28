"""v93 denetimi -- adim 6: public oranini AMPIRIK sinirla + uctan uca public cezasi."""

from __future__ import annotations

import json
from pathlib import Path

import coz as C
import numpy as np

BURA = Path(__file__).resolve().parent


def main() -> None:
    p0 = C.yukle(C.TABAN)
    n = p0.size
    dig = [e for e in C.OLCULENLER if e != C.TABAN]
    D = np.empty((len(dig), n))
    for i, e in enumerate(dig):
        D[i] = C.yukle(e) - p0
    G = (D @ D.T) / n
    m0 = C.ENV[C.TABAN]["skor"] ** 2
    m = np.array([C.ENV[e]["skor"] ** 2 for e in dig])
    b = (m0 + np.diag(G) - m) / 2.0
    U, s, Vt = np.linalg.svd(G, hermitian=True)
    d93 = C.yukle("v93") - p0
    Qf = float(d93 @ d93) / n
    cvec, _, _ = C.coz(G, (D @ d93) / n, rank=16)

    print("=== A) NULL YONLERI f'i SINIRLIYOR MU? ===")
    print("Null yonunde |b.v| yalnizca GURULTUdur. Iki gurultu kaynagi:")
    print("  (1) 5-ondalik yuvarlama  (2) public!=tam kume kosegen uyusmazligi")
    rng = np.random.default_rng(11)
    for j in (16, 17):
        v = Vt[j]
        gozlem = abs(float(b @ v))
        # yuvarlama katkisi (analitik, uniform +-5e-6 skor hatasi)
        ds = 5e-6
        sk = np.array([C.ENV[e]["skor"] for e in dig])
        # dm_i = 2 s_i ds_i ; eps_i = (dm0 - dm_i)/2
        var_round = ((2 * C.ENV[C.TABAN]["skor"] / 2) ** 2 * (ds**2 / 3)) * v.sum() ** 2 + np.sum(
            ((2 * sk / 2) ** 2 * (ds**2 / 3)) * v**2
        )
        sd_round = np.sqrt(var_round)
        print(f"\n  yon {j} (sv={s[j]:.2e}):  gozlemlenen |b.v| = {gozlem:.3e}")
        print(
            f"    yalnizca yuvarlama:  sd = {sd_round:.3e}  ->  gozlem = {gozlem / sd_round:.2f} sigma"
        )
        print(
            f"    {'f':>6} {'sd(kosegen)':>13} {'sd(toplam)':>12} {'gozlem/sd':>10} {'p(>=gozlem)':>12}"
        )
        for f in (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80):
            orn = []
            for _ in range(120):
                msk = rng.random(n) < f
                Gp = np.einsum("ij,ij->i", D[:, msk], D[:, msk]) / msk.sum()
                orn.append(float(v @ ((np.diag(G) - Gp) / 2)))
            orn = np.array(orn)
            sd_tot = np.sqrt(orn.std() ** 2 + sd_round**2)
            z = gozlem / sd_tot
            # gozlemin bu kadar KUCUK olma olasiligi (yarim-normal)
            from math import erf
            from math import sqrt as _s

            p_kucuk = erf(z / _s(2))
            print(
                f"    {f:>6.2f} {orn.std():>13.3e} {sd_tot:>12.3e} {z:>10.2f} {1 - p_kucuk:>12.3f}"
            )

    print("\n\n=== B) UCTAN UCA PUBLIC CEZASI (varsayimsiz) ===")
    print("Tahmin edilen public MSE - GERCEK public MSE =")
    print("   (Q^tam - Q^public) - c.(diagG^tam - diagG^public)")
    print(
        f"{'f':>6} {'ort hata(MSE)':>15} {'sd(MSE)':>12} {'sd(RMSLE)':>11} {'p95 RMSLE kaymasi':>18}"
    )
    tab = []
    for f in (0.05, 0.20, 0.30, 0.50):
        orn = []
        for _ in range(150):
            msk = rng.random(n) < f
            npub = int(msk.sum())
            Dp = D[:, msk]
            Gp = np.einsum("ij,ij->i", Dp, Dp) / npub
            dp = d93[msk]
            Qp = float(dp @ dp) / npub
            orn.append((Qf - Qp) - float(cvec @ (np.diag(G) - Gp)))
        orn = np.array(orn)
        sd_r = orn.std() / (2 * 1.0083)
        tab.append(
            {
                "f": f,
                "ort": float(orn.mean()),
                "sd_mse": float(orn.std()),
                "sd_rmsle": float(sd_r),
                "p95": float(np.percentile(np.abs(orn), 95) / (2 * 1.0083)),
            }
        )
        print(
            f"{f:>6.2f} {orn.mean():>+15.3e} {orn.std():>12.3e} {sd_r:>11.3e} "
            f"{np.percentile(np.abs(orn), 95) / (2 * 1.0083):>18.3e}"
        )

    print("\n=== C) ZAMANSAL BOLME (public = Nisan+Mayis) ===")
    ids = np.load(C.ONB / "_ids.npy", allow_pickle=True)
    ay = np.array([str(x).split("_")[1][:7] for x in ids])
    pub_msk = np.isin(ay, ["2026-04", "2026-05"])
    for ad, msk in [("PUBLIC=Nis+May", pub_msk), ("PRIVATE=Haz+Tem", ~pub_msk)]:
        npub = int(msk.sum())
        Dp = D[:, msk]
        Gp = (Dp @ Dp.T) / npub
        dp = d93[msk]
        Qp = float(dp @ dp) / npub
        # bu dilimde d93'un span icindeki payi
        cp, _, _ = C.coz(Gp, (Dp @ dp) / npub, rank=16)
        span = Dp.T @ cp
        art = dp - span
        print(
            f"  {ad}: n={npub} ({msk.mean():.1%})  Q_dilim={Qp:.6f}  "
            f"dik artik = {np.sqrt(float(art @ art) / npub / Qp):.2%}"
        )
    # tahmin, public dilimde kurulmus b ile ne olurdu?
    print("\n  Eger tum skorlar YALNIZ Nis+May'da olculduyse, cozum o dilime uyar.")
    print("  Haz+Tem'de v93'un sapmasi Q=0.011868 (public dilimin 1.71 kati).")
    print("  Bu dilimde <t-p0,d93> HIC olculmedi -> isaret bilinmiyor.")
    print("  En kotu durumda private MSE degisimi = +Q_priv = +0.01187 (RMSLE +0.0059),")
    print("  en iyi durumda -0.01187 (RMSLE -0.0059).")

    (BURA / "public_f.json").write_text(json.dumps({"uctan_uca": tab}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
