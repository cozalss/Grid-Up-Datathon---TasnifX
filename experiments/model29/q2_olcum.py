"""Q2 -- aday olcumu: Q, kurtoz, %1 payi, kosinusler, bagimsiz pay, kapi denetimi."""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
import z1_ortak as Z  # noqa: E402

S = Z.S
M0 = 1.005688
R_KAL = 0.0641

REF = [
    ("y46_amnezik", "tuketim_y46_amnezik_kirpik.csv"),
    ("y45_mevsimsel", "tuketim_y45_mevsimsel_kirpik.csv"),
    ("z2_analog", "tuketim_z2_analog.csv"),
    ("g7_span", "tuketim_g7_span_tau3.csv"),
]


def logoku(a):
    return np.log1p(pd.read_csv(os.path.join(S, a)).tuketim.values)


def main(adaylar):
    A6 = Z.taban()
    V102 = logoku("tuketim_v102_kappa_optimum.csv")
    ss = pd.read_csv(os.path.join(Z.KOK, "data/raw/sample_submission.csv"))

    def nrm(v):
        return v / np.sqrt((v**2).mean())

    E1 = nrm(logoku("tuketim_m4_hava_capali.csv") - V102)  # m4-v102 ekseni
    E2 = nrm(A6 - V102 - ((A6 - V102) * E1).mean() * E1)
    B = [E1, E2]
    adB = ["m4v102", "m6v102"]
    for nm, dosya in REF:
        yol = os.path.join(S, dosya)
        if os.path.exists(yol):
            f = logoku(dosya) - A6
            if (f**2).mean() > 0:
                B.append(nrm(f))
                adB.append(nm)
    Bm = np.array(B)
    G = Bm @ Bm.T / Bm.shape[1]

    rap = {}
    for dosya in adaylar:
        nm = dosya.replace("tuketim_", "").replace(".csv", "")
        d = pd.read_csv(os.path.join(S, dosya))
        b = np.log1p(d.tuketim.values)
        f = b - A6
        o = Z.olcut(f)
        Q = o["Q"]
        u = nrm(f)
        c = Bm @ u / len(u)
        w = np.linalg.solve(G + 1e-10 * np.eye(len(G)), c)
        bagimsiz = max(0.0, float(1.0 - w @ c))
        kapi = dict(
            satir=int(len(d)),
            id_birebir=bool(len(d) == len(ss) and (d.id.values == ss.iloc[:, 0].values).all()),
            nan=int(d.tuketim.isna().sum()),
            negatif=int((d.tuketim < 0).sum()),
            baslik=list(d.columns) == ["id", "tuketim"],
        )
        rap[nm] = dict(
            dosya=dosya,
            **o,
            kosinus={a: float(ci) for a, ci in zip(adB, c)},
            bagimsiz_pay=bagimsiz,
            beklenen_dMSE_m4kalitesi=-(R_KAL**2) * bagimsiz * 1.0,
            basabas_skor=float(np.sqrt(M0 + Q)),
            dagilim=dict(
                log_ort=float(b.mean()),
                log_std=float(b.std()),
                maks=float(d.tuketim.max()),
                alt1kwh=float((d.tuketim.values < 1).mean()),
            ),
            kapi=kapi,
        )
        ok = all(
            [
                kapi["id_birebir"],
                not kapi["nan"],
                not kapi["negatif"],
                kapi["baslik"],
                kapi["satir"] == 714688,
            ]
        )
        print(
            f"{nm:22s} Q={Q:.5f} kurtoz={o['kurtoz']:6.1f} %1pay={100 * o['en_kotu_yuzde1_pay']:5.1f} "
            f"bagimsiz%={100 * bagimsiz:5.1f} "
            + " ".join(f"kos_{a}={ci:+.3f}" for a, ci in zip(adB, c))
            + f" logort={b.mean():.4f} std={b.std():.4f} <1kWh={rap[nm]['dagilim']['alt1kwh']:.4f}"
            f" kapi={'OK' if ok else 'HATA'}",
            flush=True,
        )
    json.dump(
        dict(referans=adB, adaylar=rap), open(os.path.join(BURA, "q1_nisanli.json"), "w"), indent=1
    )
    print("yazildi q1_nisanli.json")


if __name__ == "__main__":
    main(sys.argv[1:])
