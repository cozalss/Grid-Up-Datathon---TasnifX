"""PROB TASARIMI -- adim 3: TEST tarafi yon vektorleri + DIKLESTIRME.

Her aday yon:
  1. CV'den ogrenilmis grup ofset deseni test satirlarina yazilir (rejim maskeli).
  2. Rejim icinde merkezlenir (kuresel seviye zaten LB ile cozulmus).
  3. **Olculmus 18 LB yonune (d_i = log1p(v_i) - log1p(v83)) DIKLESTIRILIR.**
     Aksi halde prob zaten bilinen bir seyi olcer ve hak yanar.
  4. Q = ||d||^2/n, span icinde kalan pay, ve olcek secenekleri raporlanir.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
BURA = Path(__file__).resolve().parent
ONB = KOK / "experiments" / "v93_denetim" / "onbellek"
sys.path.insert(0, str(BURA))

OLCULEN = [
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
TABAN = "v83"
M0_V83 = 1.01318**2
DMSE_V93 = -0.009792283831693976  # experiments/v93_denetim/coz.json
M_V93 = M0_V83 + DMSE_V93
S_V93 = float(np.sqrt(M_V93))


def span_tabani() -> tuple[np.ndarray, int]:
    """Olculmus yonlerin ORTONORMAL tabani (satirlar), n ile normalize."""
    p0 = np.load(ONB / f"{TABAN}.npy")
    n = p0.size
    D = np.empty((len(OLCULEN), n), dtype=np.float64)
    for i, e in enumerate(OLCULEN):
        D[i] = np.load(ONB / f"{e}.npy") - p0
    # SVD ile ortonormal taban (satir uzayi)
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    tol = s.max() * max(D.shape) * np.finfo(float).eps
    r = int((s > tol).sum())
    return Vt[:r], n


def diklestir(v: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    """v'yi B'nin (ortonormal satirlar) span'ina dik bilesenine indir."""
    c = B @ v
    dik = v - B.T @ c
    icinde = float(np.sum(c**2) / np.sum(v**2)) if np.sum(v**2) > 0 else 0.0
    return dik, icinde


def main() -> None:
    from tavan import desil, test_cercevesi

    te = test_cercevesi()
    n = len(te)
    sicak = te["sicak"].to_numpy()
    soguk = ~sicak

    lgp = te["lgp"].to_numpy()
    sev_s = np.full(n, -1, dtype=int)
    sev_s[sicak] = desil(lgp[sicak], 10)
    sev_s20 = np.full(n, -1, dtype=int)
    sev_s20[sicak] = desil(lgp[sicak], 20)
    sev_c = np.full(n, -1, dtype=int)
    sev_c[soguk] = desil(lgp[soguk], 10)
    sev_c20 = np.full(n, -1, dtype=int)
    sev_c20[soguk] = desil(lgp[soguk], 20)

    des = json.loads((BURA / "desenler.json").read_text(encoding="utf-8"))
    ay_des = json.loads((BURA / "ay_deseni.json").read_text(encoding="utf-8"))

    ilce = te["ilce"].to_numpy()
    kova = te["kova"].to_numpy()
    ay = te["ay"].to_numpy()
    ilce_kova = (te["ilce"] + "|" + te["kova"]).to_numpy()

    def yaz(harita: dict, anahtar: np.ndarray, maske: np.ndarray) -> np.ndarray:
        v = pd.Series(anahtar).astype(str).map(harita).fillna(0.0).to_numpy(dtype="float64")
        v = np.where(maske, v, 0.0)
        # rejim ICINDE merkezle
        v[maske] -= v[maske].mean()
        return v

    adaylar = {
        "P1_sicak_ilce": yaz(des["sicak"]["ilce"], ilce, sicak),
        "P2_sicak_seviye10": yaz(des["sicak"]["seviye_desili10"], sev_s, sicak),
        "P2b_sicak_seviye20": yaz(des["sicak"]["seviye_desili20"], sev_s20, sicak),
        "P3_soguk_kva": yaz(des["soguk"]["kva_kovasi"], kova, soguk),
        "P4_soguk_seviye20": yaz(des["soguk"]["seviye_desili20"], sev_c20, soguk),
        "P4b_soguk_seviye10": yaz(des["soguk"]["seviye_desili10"], sev_c, soguk),
        "P5_ay_sicak": yaz(ay_des["sicak"], ay, sicak),
        "P5b_ay_soguk": yaz(ay_des["soguk"], ay, soguk),
        "P6_sicak_ilce_kova": yaz(des["sicak"]["ilce_x_kova"], ilce_kova, sicak),
        "P7_soguk_ilce_kova": yaz(des["soguk"]["ilce_x_kova"], ilce_kova, soguk),
    }

    B, n2 = span_tabani()
    assert n2 == n, f"{n2} != {n}"
    print(f"olculmus span rank = {B.shape[0]}  (18 yon, n={n:,})\n")

    sim = {
        (s["rejim"], s["yon"]): s
        for s in json.loads((BURA / "prob_simulasyon.json").read_text(encoding="utf-8"))
    }

    rapor = []
    print(f"{'yon':24s}{'Q_ham':>11s}{'span%':>9s}{'Q_dik':>11s}{'kaps%':>8s}{'CV kazanc':>12s}")
    print("-" * 76)
    for ad, v in adaylar.items():
        q_ham = float(v @ v) / n
        dik, icinde = diklestir(v, B)
        q_dik = float(dik @ dik) / n
        kaps = float((np.abs(dik) > 1e-12).mean())
        anahtar = {
            "P1_sicak_ilce": ("sicak", "ilce"),
            "P2_sicak_seviye10": ("sicak", "seviye_desili10"),
            "P2b_sicak_seviye20": ("sicak", "seviye_desili20"),
            "P3_soguk_kva": ("soguk", "kva_kovasi"),
            "P4_soguk_seviye20": ("soguk", "seviye_desili20"),
            "P4b_soguk_seviye10": ("soguk", "seviye_desili10"),
            "P5_ay_sicak": ("sicak", "ay"),
            "P5b_ay_soguk": ("soguk", "ay"),
            "P6_sicak_ilce_kova": ("sicak", "ilce_x_kova"),
            "P7_soguk_ilce_kova": ("soguk", "ilce_x_kova"),
        }[ad]
        cv = sim.get(anahtar, {}).get("kazanc_toplam", float("nan"))
        rapor.append(
            {
                "yon": ad,
                "Q_ham": q_ham,
                "span_pay": icinde,
                "Q_dik": q_dik,
                "kapsam": kaps,
                "cv_kazanc_toplam": cv,
                "grup": len(des[anahtar[0]].get(anahtar[1], {})) if anahtar[1] != "ay" else 4,
            }
        )
        print(
            f"{ad:24s}{q_ham:>11.6f}{icinde * 100:>8.2f}%{q_dik:>11.6f}"
            f"{kaps * 100:>7.1f}%{cv:>+12.6f}"
        )
        np.save(BURA / f"yon_{ad}.npy", dik)

    (BURA / "yon.json").write_text(
        json.dumps(rapor, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # yonler arasi diklik
    print("\nYONLER ARASI KOSINUS (dikleştirilmiş):")
    adlar = list(adaylar)
    M = np.array([np.load(BURA / f"yon_{a}.npy") for a in adlar])
    G = M @ M.T / n
    nrm = np.sqrt(np.diag(G))
    C = G / np.outer(nrm, nrm)
    print("      " + "".join(f"{a[:6]:>8s}" for a in adlar))
    for i, a in enumerate(adlar):
        print(f"{a[:6]:6s}" + "".join(f"{C[i, j]:>8.3f}" for j in range(len(adlar))))

    print(f"\nM_v93 (ongorulen) = {M_V93:.7f}   S_v93 = {S_V93:.6f}")


if __name__ == "__main__":
    main()
