"""p19-F: 3/1/1 harmaninda UYE DEGISIMLERI -- kesif olcumu (on-kayit DISI, etiketli).

p19 on-kayitli olcut cat-tekil uretim harmaniydi. p20/p21 adayi soguk harmani
3/1/1'e (0.6/0.2/0.2) donduruyor. Bu betik 3/1/1 icinde uye degisimlerini olcer:

  B  = 3/1/1 standart (p21 adayi)          : 0.6 cat7 + 0.2 xgb + 0.2 lgbm
  C1 = lgbm -> huber(a=1.0)                : 0.6 cat7 + 0.2 xgb + 0.2 lgbm_hub
  C2 = cat -> derin8                       : 0.6 cat8 + 0.2 xgb + 0.2 lgbm
  C3 = ikisi birden                        : 0.6 cat8 + 0.2 xgb + 0.2 lgbm_hub

cat8 = p19_{b}_{t}_hp_derin8.npy, lgbm_hub = p11b_{b}_{t}_huber.npy.
Ortak tohum kumesi = uc kaynagin kesisimi. Egitim yok.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p11_agirlik import (PG_HAKIM, agirlik, kova, onyukleme_w, rmsle,  # noqa: E402
                         test_dagilimi, wrmsle)
from p11_ortak import BLOKLAR, DN, egitim, kirp, sicak_rmsle, toplam  # noqa: E402

W = (0.6, 0.2, 0.2)


def yol_d8(b, t):
    return os.path.join(BURA, f"p19_{b}_{t}_hp_derin8.npy")


def yol_hub(b, t):
    return os.path.join(BURA, f"p11b_{b}_{t}_huber.npy")


def main():
    T = pd.read_parquet(os.path.join(DN, "test.parquet"))
    q = test_dagilimi(T[T.soguk_mu == 1])
    del T
    E = egitim()

    R = {"aciklama": "3/1/1 uye degisimleri; taban B = p21 adayi (0.6 cat7 / 0.2 xgb / 0.2 lgbm)",
         "etiket": "KESIF -- p19 on-kayit disinda, ayni agirlikli rig ile",
         "bloklar": {}, "kazanclar": {}, "toplama": {}}

    bil = {}
    for b in BLOKLAR:
        blk = E[E._blok == b]
        sog = blk[blk.soguk_mu == 1].reset_index(drop=True)
        y = np.log1p(sog.tuketim.to_numpy(dtype="float64").clip(0))
        w = agirlik(sog, q)
        m = kova(sog)[0] == PG_HAKIM
        sic = sicak_rmsle(b)
        z = np.load(os.path.join(DN, f"soguk_tahmin_{b}.npz"))
        npz_ts = sorted({int(k.split("_")[0]) for k in z.files})
        ts = [t for t in npz_ts
              if os.path.exists(yol_d8(b, t)) and os.path.exists(yol_hub(b, t))]
        if not ts:
            R["bloklar"][b] = {"uyari": "ortak tohum yok"}
            continue

        def ort(f):
            return np.mean([f(t) for t in ts], axis=0)

        cat7 = ort(lambda t: z[f"{t}_cat"].astype(np.float64))
        xgb = ort(lambda t: z[f"{t}_xgb"].astype(np.float64))
        lgb = ort(lambda t: z[f"{t}_lgbm"].astype(np.float64))
        cat8 = ort(lambda t: np.load(yol_d8(b, t)).astype(np.float64))
        hub = ort(lambda t: np.load(yol_hub(b, t)).astype(np.float64))

        Y = {
            "B_std": W[0] * cat7 + W[1] * xgb + W[2] * lgb,
            "C1_lgbm_huber": W[0] * cat7 + W[1] * xgb + W[2] * hub,
            "C2_cat_derin8": W[0] * cat8 + W[1] * xgb + W[2] * lgb,
            "C3_ikisi": W[0] * cat8 + W[1] * xgb + W[2] * hub,
        }

        def olc(lg):
            r = y - kirp(lg)
            return dict(ham=round(rmsle(r), 5), agr=round(wrmsle(r, w), 5),
                        pg=round(rmsle(r[m]), 5),
                        bilesim_agr=round(toplam(wrmsle(r, w), sic), 5))

        R["bloklar"][b] = {"ortak_tohum": ts, "sicak_rmsle": round(sic, 5),
                           "seviye": {k: olc(v) for k, v in Y.items()}}

        rB = y - kirp(Y["B_std"])
        for k in ("C1_lgbm_huber", "C2_cat_derin8", "C3_ikisi"):
            rC = y - kirp(Y[k])
            kz = dict(
                ortak_tohum=ts,
                agr=round(wrmsle(rB, w) - wrmsle(rC, w), 5),
                pg=round(rmsle(rB[m]) - rmsle(rC[m]), 5),
                bilesim_agr=round(toplam(wrmsle(rB, w), sic)
                                  - toplam(wrmsle(rC, w), sic), 5),
                onyukleme_agr=onyukleme_w(sog.tanim.values, rB, rC, w, 500),
            )
            R["kazanclar"].setdefault(k, {})[b] = kz
            bil.setdefault(k, {})[b] = kz["bilesim_agr"]

    for k, v in bil.items():
        if len(v) == 3:
            a = np.array([v[b] for b in BLOKLAR])
            R["toplama"][k] = dict(blok=v, ort=round(float(a.mean()), 5),
                                   pozitif_blok=int((a > 0).sum()))

    yol = os.path.join(BURA, "p19_f_311_bilesim.json")
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, ensure_ascii=False)

    print(f"\n{'yapik':16}{'blok':7}{'ham':>9}{'agr':>9}{'pg':>9}{'bil':>9}")
    for b in BLOKLAR:
        sv = R["bloklar"][b].get("seviye", {})
        for k, v in sv.items():
            print(f"{k:16}{b:7}{v['ham']:>9.5f}{v['agr']:>9.5f}{v['pg']:>9.5f}"
                  f"{v['bilesim_agr']:>9.5f}")
    print("\nKAZANC (B_std'ye karsi, + = degisim IYI)")
    for k, bb in R["kazanclar"].items():
        for b, v in bb.items():
            print(f"{k:16}{b:7} agr={v['agr']:+.5f} pg={v['pg']:+.5f} "
                  f"bil={v['bilesim_agr']:+.5f} oy+={v['onyukleme_agr']['pozitif_oran']:.3f}")
    print("\nTOPLAMA")
    print(json.dumps(R["toplama"], indent=1, ensure_ascii=False))
    print("\nyazildi", yol)


if __name__ == "__main__":
    main()
