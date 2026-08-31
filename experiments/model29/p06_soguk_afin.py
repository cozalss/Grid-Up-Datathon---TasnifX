"""p06: SOGUGA OZEL afin kalibrasyon -- egim bloklar arasi KARARLI mi?

p02'nin D3'u TUM satirlarda afin kalibrasyon yapmisti (beta = -0,0065,
yani egim ~1). Soguk taraf ayri bakilmadi. Burada soguk satirlarda
  y ~ a + b*(p - p_ort)
her blokta ayri kestirilir. b bloklar arasi kararliysa YAPISAL, degilse
p01'in tuzagi.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p02_duzeltme import blok  # noqa: E402


def rmsle(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def fit(d):
    p, y = d.p.values, d.y.values
    m = p.mean()
    X = np.c_[np.ones(len(p)), p - m]
    cf = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(m), float(cf[0]), float(cf[1])


def main():
    R = {}
    S = {b: blok(b)[lambda x: x.soguk_mu == 1] for b in ("yaz25", "guz25", "kis26")}
    for b, d in S.items():
        m, a, k = fit(d)
        R[b] = dict(n=int(len(d)), p_ort=round(m, 4), sabit=round(a, 4), egim=round(k, 4),
                    taban=round(rmsle(d.r), 5),
                    kahin_afin=round(rmsle(d.y.values - (a + k * (d.p.values - m))), 5))
        print(b, R[b], flush=True)

    dis = pd.concat([S["guz25"], S["kis26"]])
    m, a, k = fit(dis)
    yz = S["yaz25"]
    R["dis_fit"] = dict(p_ort=round(m, 4), sabit=round(a, 4), egim=round(k, 4))
    uy = {}
    # (1) tam afin (sabit dahil)  (2) yalniz EGIM (sabit yerine yaz25'in kendi p_ort'u -- yapisal)
    uy["tam_afin"] = round(rmsle(yz.y.values - (a + k * (yz.p.values - m))), 5)
    my = float(yz.p.values.mean())
    for kk in (k, 1.05, 1.1, 1.15):
        uy[f"yalniz_egim_b={round(kk, 3)}"] = round(
            rmsle(yz.y.values - (my + kk * (yz.p.values - my))), 5)
    uy["taban"] = round(rmsle(yz.r), 5)
    R["yaz25_uygulama"] = uy
    print(json.dumps(R["yaz25_uygulama"], indent=1))

    with open(os.path.join(BURA, "p06_soguk_afin.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
