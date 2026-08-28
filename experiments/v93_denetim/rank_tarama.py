"""v93 denetimi -- adim 3: rank/ridge taramasi + null uzayin kimligi."""

from __future__ import annotations

import json
from pathlib import Path

import coz as C
import numpy as np

BURA = Path(__file__).resolve().parent


def main() -> None:
    digerleri, D, G, b, m0, n = C.gram_kur(C.OLCULENLER)
    U, s, Vt = np.linalg.svd(G, hermitian=True)
    d93 = C.yukle("v93") - C.yukle(C.TABAN)

    print("=== NULL UZAY KIMLIGI (en kucuk 3 ozvektor) ===")
    for j in (17, 16, 15):
        v = Vt[j]
        buyuk = sorted(zip(digerleri, v), key=lambda t: -abs(t[1]))[:6]
        print(f"sv[{j}]={s[j]:.3e}: " + "  ".join(f"{e}{x:+.3f}" for e, x in buyuk))
        # bu bagimliligin gercek log-uzay artigi
        kombi = D.T @ v
        print(f"          ||sum v_i d_i||^2/n = {float(kombi @ kombi) / n:.3e}")

    print("\n=== RANK TARAMASI ===")
    print(
        f"{'rank':>4} {'|w|_1':>14} {'proj^2':>12} {'ulas.RMSLE':>11} "
        f"{'v93 dMSE':>12} {'v93 RMSLE':>10} {'v93 artik%':>10}"
    )
    satirlar = []
    for r in range(4, 19):
        w, _, _ = C.coz(G, b, rank=r)
        pj = float(w @ b)
        res = C.yansit(D, G, b, d93, n, rank=r)
        mse = m0 + res["dMSE_bilinen"]
        satirlar.append(
            {
                "rank": r,
                "w_l1": float(np.abs(w).sum()),
                "proj2": pj,
                "ulas_rmsle": float(np.sqrt(m0 - pj)) if m0 - pj > 0 else None,
                "v93_dmse": res["dMSE_bilinen"],
                "v93_rmsle": float(np.sqrt(mse)) if mse > 0 else None,
                "v93_artik": res["artik_orani"],
            }
        )
        u = satirlar[-1]
        print(
            f"{r:>4} {u['w_l1']:>14.4f} {pj:>12.6f} "
            f"{(f'{u[chr(117) + chr(108) + chr(97) + chr(115) + chr(95) + chr(114) + chr(109) + chr(115) + chr(108) + chr(101)]:.6f}' if u['ulas_rmsle'] else 'IMKANSIZ'):>11} "
            f"{u['v93_dmse']:>+12.6f} "
            f"{(f'{u[chr(118) + chr(57) + chr(51) + chr(95) + chr(114) + chr(109) + chr(115) + chr(108) + chr(101)]:.6f}' if u['v93_rmsle'] else 'IMKANSIZ'):>10} "
            f"{u['v93_artik']:>9.3%}"
        )

    print("\n=== RIDGE TARAMASI (tam rank uzerinde) ===")
    print(f"{'lambda':>10} {'|w|_1':>12} {'ulas.RMSLE':>11} {'v93 RMSLE':>10}")
    ridge = []
    for lam in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]:
        w, _, _ = C.coz(G, b, rank=18, lam=lam)
        pj = float(w @ b)
        res = C.yansit(D, G, b, d93, n, rank=18, lam=lam)
        mse = m0 + res["dMSE_bilinen"]
        ridge.append(
            {
                "lam": lam,
                "w_l1": float(np.abs(w).sum()),
                "ulas_rmsle": float(np.sqrt(m0 - pj)) if m0 - pj > 0 else None,
                "v93_rmsle": float(np.sqrt(mse)) if mse > 0 else None,
            }
        )
        u = ridge[-1]
        a = f"{u['ulas_rmsle']:.6f}" if u["ulas_rmsle"] else "IMKANSIZ"
        c2 = f"{u['v93_rmsle']:.6f}" if u["v93_rmsle"] else "IMKANSIZ"
        print(f"{lam:>10.0e} {u['w_l1']:>12.4f} {a:>11} {c2:>10}")

    (BURA / "rank_tarama.json").write_text(
        json.dumps(
            {"tekil": s.tolist(), "rank_tarama": satirlar, "ridge": ridge},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
