"""p06: TEST icin SOGUK aile-agirligi DELTASI uret.

Bulgu (p06_soguk_agirlik.json): soguk harmanda cat/xgb/lgbm esit agirlikli
ama bu yaz25'te (test'in mevsimsel ikizi) ve guz25'te ZARARLI. guz25'te
SIZINTISIZ secilen agirlik (0,05 / 0,35 / 0,60) yaz25 soguk RMSLE'yi
1,43592 -> 1,41269 dusuruyor (test bilesimi 0,97932 -> 0,97180).

URETILEN SEY: uretim gonderimine EKLENECEK bir DELTA vektoru --
    delta = P_test @ (w_yeni - w_esit)
sadece SOGUK satirlarda sifirdan farkli. Boylece uretim hattinin geri
kalani (tohum sayisi, m112/m148 katmani) AYNEN korunur ve yalnizca aile
agirligi degisir.

Egitim: TUM egitim satirlari (uretimdeki test kosusu gibi), maske orani
1,00 (saf soguk uzman), cat icin depth 7 -- scripts/uret_soguk_tahmin.py
ile birebir ayni ayar.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
BURA = os.path.dirname(os.path.abspath(__file__))
CIKTI = (
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad"
)
sys.path.insert(0, BURA)
sys.path.insert(0, os.path.join(KOK, "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

AILE = ("cat", "xgb", "lgbm")
TOHUM = (1000, 1001, 1002)
W_YENI = np.array([0.05, 0.35, 0.60])
W_ESIT = np.array([1 / 3, 1 / 3, 1 / 3])
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def main():
    os.makedirs(CIKTI, exist_ok=True)
    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    soguk = (test.soguk_mu == 1).to_numpy()
    ts = test.loc[soguk]
    log(f"egitim {egitim.shape}  test {test.shape}  soguk {soguk.sum():,}  kolon {len(kol)}")

    P = {a: [] for a in AILE}
    for tohum in TOHUM:
        maskeli = d.soguk_maskele(egitim, kol, 1.00, tohum)
        for a in AILE:
            ust = {"depth": 7} if a == "cat" else {}
            P[a].append(di.egit_tahmin(a, maskeli, ts, kol, tohum, **ust))
            log(f"{tohum}_{a} hazir")
        del maskeli
    A = np.c_[[np.mean(P[a], axis=0) for a in AILE]].T
    np.save(os.path.join(CIKTI, "p06_test_soguk_aile.npy"), A)

    d_soguk = A @ (W_YENI - W_ESIT)
    delta = np.zeros(len(test))
    delta[soguk] = d_soguk
    np.save(os.path.join(CIKTI, "p06_test_delta_log.npy"), delta)
    np.save(os.path.join(CIKTI, "p06_test_soguk_maske.npy"), soguk)
    test[["tanim", "tarih"]].assign(delta=delta).to_parquet(
        os.path.join(CIKTI, "p06_test_delta.parquet"), index=False
    )

    R = dict(
        w_yeni=list(W_YENI),
        w_esit=list(W_ESIT),
        tohum=list(TOHUM),
        aile=list(AILE),
        n_test=int(len(test)),
        n_soguk=int(soguk.sum()),
        aile_ort={a: round(float(A[:, i].mean()), 4) for i, a in enumerate(AILE)},
        delta_ort=round(float(d_soguk.mean()), 5),
        delta_std=round(float(d_soguk.std()), 5),
        delta_mutlak_ort=round(float(np.abs(d_soguk).mean()), 5),
        cikti=CIKTI,
    )
    with open(os.path.join(BURA, "p06_soguk_test.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
