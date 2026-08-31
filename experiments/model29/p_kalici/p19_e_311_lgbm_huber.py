"""p19-E: YENI SOGUK HARMAN 3/1/1 (p20 adayi) + lgbm uyesini HUBER yapmak.

SORU (devir gorevi #5): p20 adayi soguk harmani 3/1/1'e dondurdu
(cat 0.6 / xgb 0.2 / lgbm 0.2). p14'un soguk lgbm huber bulgusu ESIT harmanda
olculmustu ve "uretimde lgbm yok" diye gecersiz sayilmisti. 3/1/1'de lgbm
GERI GELDI (pay 0.2). lgbm uyesini huber yapmak bilesik soguk RMSLE'yi ne yapar?

EGITIM YOK. Girdiler:
  data/interim/deney/soguk_tahmin_{blok}.npz  anahtar {tohum}_{aile}
  scratchpad p11b_{blok}_{tohum}_{aday}.npy   (soguk lgbm log tahmini, float32)

Olcut: p19 on-kaydi ile ayni (kohort agirlikli wrmsle, PG(75,90], test bilesimi).
Karsilastirma ORTAK tohum kumesinde: B = 3/1/1 standart lgbm,
C = 3/1/1 lgbm->huber. Kazanc = B - C yonunde degil; kazanc = olc(B) - olc(C)
(pozitif = huber IYI). A = cat-tekil (eski uretim) yalniz baglam icin.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p11_agirlik import (PG_HAKIM, agirlik, ess, kova, onyukleme_w, rmsle,  # noqa: E402
                         test_dagilimi, wrmsle)
from p11_ortak import BLOKLAR, DN, HEDEF_SOGUK, egitim, kirp, sicak_rmsle, toplam  # noqa: E402

W311 = {"cat": 0.6, "xgb": 0.2, "lgbm": 0.2}
HUBER_ADAYLAR = ("huber", "huber_a05")  # a=1.0 (p14 mansedi), a=0.5 (genis izgara secimi)


def p11b(b, t, a):
    return os.path.join(BURA, f"p11b_{b}_{t}_{a}.npy")


def npz_yukle(b):
    return np.load(os.path.join(DN, f"soguk_tahmin_{b}.npz"))


def main():
    T = pd.read_parquet(os.path.join(DN, "test.parquet"))
    q = test_dagilimi(T[T.soguk_mu == 1])
    del T
    E = egitim()

    R = {"soru": "3/1/1 harmaninda lgbm uyesi huber olursa ne olur?",
         "harman": W311, "bloklar": {}, "kazanclar": {}, "toplama": {}}

    bil = {a: {} for a in HUBER_ADAYLAR}
    for b in BLOKLAR:
        blk = E[E._blok == b]
        sog = blk[blk.soguk_mu == 1].reset_index(drop=True)
        y = np.log1p(sog.tuketim.to_numpy(dtype="float64").clip(0))
        w = agirlik(sog, q)
        m = kova(sog)[0] == PG_HAKIM
        sic = sicak_rmsle(b)
        z = npz_yukle(b)
        npz_tohum = sorted({k.split("_")[0] for k in z.files})

        def olc(lg):
            r = y - kirp(lg)
            return dict(ham=round(rmsle(r), 5), agr=round(wrmsle(r, w), 5),
                        pg=round(rmsle(r[m]), 5),
                        bilesim_agr=round(toplam(wrmsle(r, w), sic), 5))

        Rb = {"n_soguk": int(len(y)), "sicak_rmsle": round(sic, 5),
              "agirlik_ess": round(ess(w), 1), "npz_tohum": npz_tohum,
              "seviye": {}}
        R["bloklar"][b] = Rb

        # referans: tam tohum kumesiyle A ve B
        def aile_ort(a, ts):
            return np.mean([z[f"{t}_{a}"].astype(np.float64) for t in ts], axis=0)

        A_tam = aile_ort("cat", npz_tohum)
        B_tam = sum(W311[a] * aile_ort(a, npz_tohum) for a in ("cat", "xgb", "lgbm"))
        Rb["seviye"]["A_cat_tekil_tam_tohum"] = olc(A_tam)
        Rb["seviye"]["B_311_std_tam_tohum"] = olc(B_tam)

        for ha in HUBER_ADAYLAR:
            ts = [t for t in npz_tohum if os.path.exists(p11b(b, t, ha))]
            if not ts:
                continue
            # dogruluk: p11b TABAN npz lgbm ile birebir mi (float32 payi)
            dog = {}
            for t in ts:
                yol = p11b(b, t, "TABAN")
                if os.path.exists(yol):
                    mx = float(np.max(np.abs(
                        np.load(yol).astype(np.float64) - z[f"{t}_lgbm"].astype(np.float64))))
                    dog[t] = mx
                    assert mx < 1e-5, f"p11b TABAN != npz lgbm ({b} t={t} maxabs={mx})"

            hub = np.mean([np.load(p11b(b, t, ha)).astype(np.float64) for t in ts], axis=0)
            cat_o = aile_ort("cat", ts)
            xgb_o = aile_ort("xgb", ts)
            lgb_o = aile_ort("lgbm", ts)
            A = cat_o
            B = W311["cat"] * cat_o + W311["xgb"] * xgb_o + W311["lgbm"] * lgb_o
            C = W311["cat"] * cat_o + W311["xgb"] * xgb_o + W311["lgbm"] * hub

            rA, rB, rC = (y - kirp(A)), (y - kirp(B)), (y - kirp(C))
            Rb["seviye"][f"A_cat_tekil_ortak_{ha}"] = dict(tohum=ts, **olc(A))
            Rb["seviye"][f"B_311_std_ortak_{ha}"] = dict(tohum=ts, **olc(B))
            Rb["seviye"][f"C_311_{ha}"] = dict(tohum=ts, **olc(C))

            # tohum bazli delta (agr): B_t vs C_t tek tohumla
            tb = {}
            for t in ts:
                Bt = (W311["cat"] * z[f"{t}_cat"].astype(np.float64)
                      + W311["xgb"] * z[f"{t}_xgb"].astype(np.float64)
                      + W311["lgbm"] * z[f"{t}_lgbm"].astype(np.float64))
                Ct = (W311["cat"] * z[f"{t}_cat"].astype(np.float64)
                      + W311["xgb"] * z[f"{t}_xgb"].astype(np.float64)
                      + W311["lgbm"] * np.load(p11b(b, t, ha)).astype(np.float64))
                tb[str(t)] = round(wrmsle(y - kirp(Bt), w) - wrmsle(y - kirp(Ct), w), 5)

            kz = dict(
                ortak_tohum=ts, n_tohum=len(ts),
                p11b_taban_dogrulama_maxabs={str(k): round(v, 8) for k, v in dog.items()},
                ham=round(rmsle(rB) - rmsle(rC), 5),
                agr=round(wrmsle(rB, w) - wrmsle(rC, w), 5),
                pg=round(rmsle(rB[m]) - rmsle(rC[m]), 5),
                bilesim_agr=round(toplam(wrmsle(rB, w), sic) - toplam(wrmsle(rC, w), sic), 5),
                tohum_bazli_agr=tb,
                onyukleme_agr=onyukleme_w(sog.tanim.values, rB, rC, w, 500),
                onyukleme_pg=onyukleme_w(sog.tanim.values[m], rB[m], rC[m],
                                         np.ones(int(m.sum())), 500),
                C_vs_cat_tekil_agr=round(wrmsle(rA, w) - wrmsle(rC, w), 5),
                C_vs_cat_tekil_bilesim_agr=round(
                    toplam(wrmsle(rA, w), sic) - toplam(wrmsle(rC, w), sic), 5),
            )
            R["kazanclar"].setdefault(ha, {})[b] = kz
            bil[ha][b] = kz["bilesim_agr"]

    for ha in HUBER_ADAYLAR:
        if len(bil[ha]) < 3:
            continue
        v = np.array([bil[ha][b] for b in BLOKLAR])
        R["toplama"][ha] = dict(
            blok={b: bil[ha][b] for b in BLOKLAR},
            ort=round(float(v.mean()), 5),
            pozitif_blok=int((v > 0).sum()),
            not_="bilesim_agr = test bilesimi RMSLE kazanci (B_311_std - C_311_huber), pozitif = huber IYI")

    yol = os.path.join(BURA, "p19_e_311_lgbm_huber.json")
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, ensure_ascii=False)

    print(f"\n{'yapik':28}{'blok':7}{'ham':>9}{'agr':>9}{'pg':>9}{'bil_agr':>9}")
    for b in BLOKLAR:
        for a, v in R["bloklar"][b]["seviye"].items():
            print(f"{a:28}{b:7}{v['ham']:>9.5f}{v['agr']:>9.5f}{v['pg']:>9.5f}"
                  f"{v['bilesim_agr']:>9.5f}")
    print(f"\nKAZANC C-B (pozitif = huber IYI)")
    print(f"{'aday':12}{'blok':7}{'nt':>3}{'ham':>9}{'agr':>9}{'pg':>9}{'bil':>9}"
          f"{'oy+':>7}{'oypg+':>7}")
    for a, bb in R["kazanclar"].items():
        for b, v in bb.items():
            print(f"{a:12}{b:7}{v['n_tohum']:>3}{v['ham']:>+9.5f}{v['agr']:>+9.5f}"
                  f"{v['pg']:>+9.5f}{v['bilesim_agr']:>+9.5f}"
                  f"{v['onyukleme_agr']['pozitif_oran']:>7.3f}"
                  f"{v['onyukleme_pg']['pozitif_oran']:>7.3f}")
    print("\nTOPLAMA")
    print(json.dumps(R["toplama"], indent=1, ensure_ascii=False))
    print("\nyazildi", yol)


if __name__ == "__main__":
    main()
