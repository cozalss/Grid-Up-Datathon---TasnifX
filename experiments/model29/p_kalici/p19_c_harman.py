"""p19-C: URETIM soguk harmani (YALNIZ CAT) ne kadar kotu?  BEDAVA OLCUM.

Butun veri data/interim/deney/soguk_tahmin_{blok}.npz icinde ZATEN var.
Hicbir egitim yapilmaz.

Sorulan: uretimin soguk tarafi cat-tekil. cat+xgb+lgbm ESIT harman daha mi iyi?
Bu, p06'nin sordugu soru DEGIL -- p06 esit harmani TABAN alip agirlik ariyordu.
Buradaki taban CAT-TEKIL (gercek uretim).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p11_agirlik import (
    PG_HAKIM,
    agirlik,
    ess,
    kova,
    onyukleme_w,
    rmsle,  # noqa: E402
    test_dagilimi,
    wrmsle,
)
from p11_ortak import (  # noqa: E402
    AILE,
    BLOKLAR,
    DN,
    egitim,
    kirp,
    sicak_rmsle,
    toplam,
)

#: On-kayit: bu agirlik kumesi olcumden ONCE sabitlendi.
HARMANLAR = {
    "URETIM_cat_tekil": (1.0, 0.0, 0.0),
    "esit": (1 / 3, 1 / 3, 1 / 3),
    "cat_xgb": (0.5, 0.5, 0.0),
    "cat_lgbm": (0.5, 0.0, 0.5),
    "xgb_lgbm": (0.0, 0.5, 0.5),
    "yalniz_xgb": (0.0, 1.0, 0.0),
    "yalniz_lgbm": (0.0, 0.0, 1.0),
    "cat_agir": (0.5, 0.25, 0.25),
    "p06": (0.05, 0.35, 0.60),
}


def main():
    T = pd.read_parquet(os.path.join(DN, "test.parquet"))
    q = test_dagilimi(T[T.soguk_mu == 1])
    del T
    E = egitim()

    R = {"aciklama": "TABAN = URETIM cat-tekil soguk uzman", "bloklar": {}, "kazanclar": {}}
    MW = {}
    for b in BLOKLAR:
        blk = E[E._blok == b]
        sog = blk[blk.soguk_mu == 1].reset_index(drop=True)
        y = np.log1p(sog.tuketim.to_numpy(dtype="float64").clip(0))
        w = agirlik(sog, q)
        m = kova(sog)[0] == PG_HAKIM
        sic = sicak_rmsle(b)
        z = np.load(os.path.join(DN, f"soguk_tahmin_{b}.npz"))
        P = np.c_[
            [np.mean([z[k] for k in z.files if k.endswith("_" + a)], axis=0) for a in AILE]
        ].T.astype(np.float64)
        R["bloklar"][b] = {
            "n_soguk": int(len(y)),
            "sicak": round(sic, 5),
            "ess": round(ess(w), 1),
            "seviye": {},
        }
        MW[b] = {}
        r0 = None
        for ad, ww in HARMANLAR.items():
            lg = P @ np.array(ww, dtype="float64")
            r = y - kirp(lg)
            MW[b][ad] = float(np.sum(w * r * r) / w.sum())
            R["bloklar"][b]["seviye"][ad] = dict(
                ham=round(rmsle(r), 5),
                agr=round(wrmsle(r, w), 5),
                pg=round(rmsle(r[m]), 5),
                bilesim_ham=round(toplam(rmsle(r), sic), 5),
                bilesim_agr=round(toplam(wrmsle(r, w), sic), 5),
            )
            if ad == "URETIM_cat_tekil":
                r0 = r
        for ad in HARMANLAR:
            if ad == "URETIM_cat_tekil":
                continue
            lg = P @ np.array(HARMANLAR[ad], dtype="float64")
            r1 = y - kirp(lg)
            R["kazanclar"].setdefault(ad, {})[b] = dict(
                ham=round(rmsle(r0) - rmsle(r1), 5),
                agr=round(wrmsle(r0, w) - wrmsle(r1, w), 5),
                pg=round(rmsle(r0[m]) - rmsle(r1[m]), 5),
                bilesim_agr=round(toplam(wrmsle(r0, w), sic) - toplam(wrmsle(r1, w), sic), 5),
                onyukleme_agr=onyukleme_w(sog.tanim.values, r0, r1, w, 500),
                onyukleme_pg=onyukleme_w(
                    sog.tanim.values[m], r0[m], r1[m], np.ones(int(m.sum())), 500
                ),
            )

    sec = {}
    for h in BLOKLAR:
        dis = [b for b in BLOKLAR if b != h]
        puan = {a: sum(MW[b][a] / MW[b]["URETIM_cat_tekil"] for b in dis) for a in HARMANLAR}
        s = min(puan, key=puan.get)
        sec[h] = dict(
            secilen=s,
            puan={k: round(v, 5) for k, v in puan.items()},
            hedefte_kazanc=(None if s == "URETIM_cat_tekil" else R["kazanclar"][s][h]),
        )
    R["blok_disi_secim"] = sec

    with open(os.path.join(BURA, "p19_c_harman.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, ensure_ascii=False)

    print(f"\n{'harman':20}{'blok':7}{'ham':>10}{'agr':>10}{'pg':>10}{'bil_agr':>10}")
    for b in BLOKLAR:
        for a, v in R["bloklar"][b]["seviye"].items():
            print(
                f"{a:20}{b:7}{v['ham']:>10.5f}{v['agr']:>10.5f}{v['pg']:>10.5f}"
                f"{v['bilesim_agr']:>10.5f}"
            )
    print(
        f"\nKAZANC (cat-tekile karsi, + = iyi)\n{'harman':20}{'blok':7}"
        f"{'ham':>10}{'agr':>10}{'pg':>10}{'bil_agr':>10}{'oy+':>8}{'oypg+':>8}"
    )
    for a, bb in R["kazanclar"].items():
        for b, v in bb.items():
            print(
                f"{a:20}{b:7}{v['ham']:>+10.5f}{v['agr']:>+10.5f}{v['pg']:>+10.5f}"
                f"{v['bilesim_agr']:>+10.5f}{v['onyukleme_agr']['pozitif_oran']:>8.3f}"
                f"{v['onyukleme_pg']['pozitif_oran']:>8.3f}"
            )
    print("\nBLOK-DISI SECIM")
    print(json.dumps(sec, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
