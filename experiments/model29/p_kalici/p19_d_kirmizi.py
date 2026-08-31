"""p19-D: p19-C harman bulgusunun KIRMIZI TAKIM sinamasi.

Sinanan iddia: URETIM soguk harmani (cat-tekil) yaz25 ve guz25'te
cesitlendirilmis harmandan BELIRGIN kotu; kis26'da iyi.

Sinamalar:
  1) TOHUM GURULTU TABANI -- tekil tohumlarda ayni karsilastirma.
  2) TRAFO BAZINDA AYRISTIRMA -- kazancin kacta kaci tek/ilk-bes trafodan?
     (projenin kendi kalici dersi: 1.223 trafolu katta tek olu trafo t=13,71)
  3) SIFIR SATIRLARI -- kazanc tuketim=0 satirlarindan mi geliyor?
  4) BLOKLAR ARASI toplama: test bilesimi kazanci, ortalama ve SH.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p11_agirlik import PG_HAKIM, agirlik, kova, rmsle, test_dagilimi, wrmsle  # noqa: E402
from p11_ortak import AILE, BLOKLAR, DN, egitim, kirp, sicak_rmsle, toplam  # noqa: E402

ADAY = {"cat_agir": (0.5, 0.25, 0.25), "esit": (1 / 3, 1 / 3, 1 / 3), "cat_xgb": (0.5, 0.5, 0.0)}


def main():
    T = pd.read_parquet(os.path.join(DN, "test.parquet"))
    q = test_dagilimi(T[T.soguk_mu == 1])
    del T
    E = egitim()
    R = {"tohum_bazli": {}, "trafo_ayristirma": {}, "sifir_ayristirma": {}, "toplama": {}}

    bil = {a: {} for a in ADAY}
    for b in BLOKLAR:
        blk = E[E._blok == b]
        sog = blk[blk.soguk_mu == 1].reset_index(drop=True)
        y = np.log1p(sog.tuketim.to_numpy(dtype="float64").clip(0))
        w = agirlik(sog, q)
        m = kova(sog)[0] == PG_HAKIM
        sic = sicak_rmsle(b)
        z = np.load(os.path.join(DN, f"soguk_tahmin_{b}.npz"))
        tohumlar = sorted({k.split("_")[0] for k in z.files})

        # --- 1) tohum bazli
        R["tohum_bazli"][b] = {}
        for t in tohumlar:
            P = np.c_[[z[f"{t}_{a}"] for a in AILE]].T.astype(np.float64)
            r0 = y - kirp(P[:, 0])
            satir = {}
            for ad, ww in ADAY.items():
                r1 = y - kirp(P @ np.array(ww))
                satir[ad] = dict(
                    agr=round(wrmsle(r0, w) - wrmsle(r1, w), 5),
                    pg=round(rmsle(r0[m]) - rmsle(r1[m]), 5),
                )
            R["tohum_bazli"][b][t] = satir

        # tam (tum tohum) tahminler
        P = np.c_[
            [np.mean([z[k] for k in z.files if k.endswith("_" + a)], axis=0) for a in AILE]
        ].T.astype(np.float64)
        r0 = y - kirp(P[:, 0])
        R["trafo_ayristirma"][b] = {}
        R["sifir_ayristirma"][b] = {}
        sifir = sog.tuketim.to_numpy() <= 0
        for ad, ww in ADAY.items():
            r1 = y - kirp(P @ np.array(ww))
            # --- 2) trafo bazinda agirlikli dMSE
            d = w * (r0**2 - r1**2)
            g = pd.Series(d).groupby(sog.tanim.values).sum().sort_values(ascending=False)
            tp = float(d.sum())
            R["trafo_ayristirma"][b][ad] = dict(
                toplam_dMSE_w=round(tp, 4),
                n_trafo=int(len(g)),
                en_buyuk_trafo=str(g.index[0]),
                en_buyuk_pay=round(float(g.iloc[0] / tp), 4) if tp else None,
                ilk5_pay=round(float(g.iloc[:5].sum() / tp), 4) if tp else None,
                pozitif_trafo_orani=round(float((g > 0).mean()), 4),
            )
            # --- 3) sifir / sifir-disi
            R["sifir_ayristirma"][b][ad] = dict(
                sifir_pay_taban=round(float((w * r0**2)[sifir].sum() / (w * r0**2).sum()), 4),
                dMSE_w_sifir=round(float(d[sifir].sum()), 4),
                dMSE_w_sifirdisi=round(float(d[~sifir].sum()), 4),
            )
            bil[ad][b] = round(toplam(wrmsle(r0, w), sic) - toplam(wrmsle(r1, w), sic), 5)

    for ad in ADAY:
        v = np.array([bil[ad][b] for b in BLOKLAR])
        R["toplama"][ad] = dict(
            blok=bil[ad],
            ort=round(float(v.mean()), 5),
            sh=round(float(v.std(ddof=1) / np.sqrt(3)), 5),
            pozitif_blok=int((v > 0).sum()),
        )

    with open(os.path.join(BURA, "p19_d_kirmizi.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, ensure_ascii=False)

    print("\n1) TOHUM BAZLI KAZANC (agr / pg)")
    for b in BLOKLAR:
        for t, s in R["tohum_bazli"][b].items():
            print(
                f"  {b:6} t={t} "
                + "  ".join(f"{a}:{s[a]['agr']:+.4f}/{s[a]['pg']:+.4f}" for a in ADAY)
            )
    print("\n2) TRAFO AYRISTIRMA")
    for b in BLOKLAR:
        for a, v in R["trafo_ayristirma"][b].items():
            print(
                f"  {b:6} {a:10} dMSE_w={v['toplam_dMSE_w']:+.3f} "
                f"en_buyuk={v['en_buyuk_pay']} ilk5={v['ilk5_pay']} "
                f"poz_trafo={v['pozitif_trafo_orani']}"
            )
    print("\n3) SIFIR AYRISTIRMA")
    for b in BLOKLAR:
        for a, v in R["sifir_ayristirma"][b].items():
            print(
                f"  {b:6} {a:10} taban_sifir_pay={v['sifir_pay_taban']} "
                f"dMSE_sifir={v['dMSE_w_sifir']:+.3f} "
                f"dMSE_sifirdisi={v['dMSE_w_sifirdisi']:+.3f}"
            )
    print("\n4) TOPLAMA (test bilesimi kazanci)")
    print(json.dumps(R["toplama"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
