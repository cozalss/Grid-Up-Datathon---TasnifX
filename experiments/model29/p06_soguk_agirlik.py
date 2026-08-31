"""p06: SOGUK harmanda aile agirligi -- aday izgarasi, blok blok + guven araligi.

Karar zemini: Kural 10 (docs/45) yaz25'i TEST'in mevsimsel/geometrik ikizi
sayar ve soguk hukumlerin orada verilmesini sart kosar. kis26 (Ara-Mar)
test doneminden (Nis-Tem) mevsimce uzaktir.

Burada:
  - aday agirliklar her blokta olculur,
  - yaz25 soguk icin (lgbm agirligi) - (esit harman) farkinin TRAFO
    BAZLI onyukleme (bootstrap) guven araligi verilir,
  - toplam test-bilesimi skoru hesaplanir.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p02_duzeltme import DN  # noqa: E402

E = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
AILE = ("cat", "xgb", "lgbm")
HEDEF_SOGUK = 0.2216
SICAK_YAZ25 = 0.8031875554912884
ADAY = {
    "esit_(uretim)": (1 / 3, 1 / 3, 1 / 3),
    "cat_yok": (0.0, 0.5, 0.5),
    "lgbm_agirlikli_.1.2.7": (0.1, 0.2, 0.7),
    "lgbm_agirlikli_.15.25.6": (0.15, 0.25, 0.6),
    "lgbm_.0.25.75": (0.0, 0.25, 0.75),
    "yalniz_lgbm": (0.0, 0.0, 1.0),
    "yalniz_cat": (1.0, 0.0, 0.0),
}


def rmsle(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def toplam(soguk_rmsle):
    return float(np.sqrt(HEDEF_SOGUK * soguk_rmsle**2 + (1 - HEDEF_SOGUK) * SICAK_YAZ25**2))


def yukle(bad):
    blk = E[E._blok == bad]
    sog = blk[blk.soguk_mu == 1]
    z = np.load(os.path.join(DN, f"soguk_tahmin_{bad}.npz"))
    P = np.c_[[np.mean([z[k] for k in z.files if k.endswith("_" + a)], axis=0) for a in AILE]].T
    y = np.log1p(sog.tuketim.to_numpy(dtype="float64").clip(0))
    return sog, y, P


def main():
    R = {"aday": {k: list(v) for k, v in ADAY.items()}}
    for bad in ("yaz25", "guz25", "kis26"):
        sog, y, P = yukle(bad)
        R[bad] = {ad: round(rmsle(y - P @ np.array(w)), 5) for ad, w in ADAY.items()}
        print(bad, json.dumps(R[bad], indent=1), flush=True)

    sog, y, P = yukle("yaz25")
    R["yaz25_toplam_test_bilesimi"] = {ad: round(toplam(v), 5) for ad, v in R["yaz25"].items()}
    print("yaz25 TOPLAM:", json.dumps(R["yaz25_toplam_test_bilesimi"], indent=1))

    # --- trafo bazli onyukleme: yalniz_lgbm - esit farki
    rng = np.random.default_rng(0)
    tn = sog.tanim.to_numpy()
    utn, kod = np.unique(tn, return_inverse=True)
    yerler = [np.where(kod == i)[0] for i in range(len(utn))]
    e_r = y - P @ np.array(ADAY["esit_(uretim)"])
    l_r = y - P @ np.array(ADAY["yalniz_lgbm"])
    fark = []
    for _ in range(400):
        sec = rng.integers(0, len(utn), len(utn))
        idx = np.concatenate([yerler[i] for i in sec])
        fark.append(rmsle(e_r[idx]) - rmsle(l_r[idx]))
    fark = np.array(fark)
    R["onyukleme_lgbm_kazanci"] = dict(
        ort=round(float(fark.mean()), 5),
        std=round(float(fark.std()), 5),
        alt2_5=round(float(np.percentile(fark, 2.5)), 5),
        ust97_5=round(float(np.percentile(fark, 97.5)), 5),
        pozitif_oran=round(float((fark > 0).mean()), 4),
    )
    print(json.dumps(R["onyukleme_lgbm_kazanci"], indent=1))

    with open(os.path.join(BURA, "p06_soguk_agirlik.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
