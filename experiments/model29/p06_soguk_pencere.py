"""p06: PENCERE OZNITELIKLERI soguk trafolarda SIFIRI ongoruyor mu?

p06_soguk_sifir.py'nin bulgusu: yaz25'te TAM SIFIR soguk trafolarin medyan
p_gun_sayisi'si 3 (digerleri 37), p_yayilma'si 4 (digerleri 37,5).

Burada bunun BLOKLAR ARASI kararli olup olmadigini ve TEST'te ayni desenin
bulunup bulunmadigini olcuyoruz. Test'te hedef YOK -- yalnizca oznitelik
dagilimi karsilastirilir.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p02_duzeltme import DN  # noqa: E402
from p06_soguk_tani import hazirla  # noqa: E402

E = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
KEN = [1, 2, 3, 5, 8, 15, 30, 60, 100]


def main():
    R = {}
    for bad in ("yaz25", "guz25", "kis26"):
        d = hazirla(bad)
        s = d[d.soguk_mu == 1].copy()
        s["gun"] = E.loc[s.index, "p_gun_sayisi"].values
        s["yay"] = E.loc[s.index, "p_yayilma"].values
        s["sf"] = (s.tuketim <= 0).astype(int)
        s["kova"] = np.digitize(s.gun.values, KEN)
        rows = []
        for k, g in s.groupby("kova"):
            rows.append(
                dict(
                    kova=int(k),
                    gun_araligi=f"<={KEN[k] if k < len(KEN) else '+'}" if k < len(KEN) else f">{KEN[-1]}",
                    n_satir=int(len(g)),
                    n_trafo=int(g.tanim.nunique()),
                    sifir_orani=round(float(g.sf.mean()), 4),
                    p_ort=round(float(g.p.mean()), 3),
                    y_ort=round(float(g.y.mean()), 3),
                    rmsle=round(float(np.sqrt((g.r**2).mean())), 4),
                    kare_pay=round(float((g.r**2).sum() / (s.r**2).sum()), 4),
                )
            )
        R[bad] = rows
        print(f"--- {bad}")
        print(pd.DataFrame(rows).to_string(index=False), flush=True)

    # --- TEST dagilimi (hedef yok)
    tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
    ts = tp[tp.soguk_mu == 1]
    kova = np.digitize(ts.p_gun_sayisi.values, KEN)
    rows = []
    for k in sorted(set(kova)):
        m = kova == k
        rows.append(
            dict(
                kova=int(k),
                n_satir=int(m.sum()),
                n_trafo=int(ts.loc[m, "tanim"].nunique()),
                satir_pay=round(float(m.mean()), 4),
            )
        )
    R["test_soguk_dagilim"] = rows
    print("--- TEST soguk")
    print(pd.DataFrame(rows).to_string(index=False))
    R["test_soguk_toplam"] = dict(satir=int(len(ts)), trafo=int(ts.tanim.nunique()))
    R["kenarlar"] = KEN

    with open(os.path.join(BURA, "p06_soguk_pencere.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
