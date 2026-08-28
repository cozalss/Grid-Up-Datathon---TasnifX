"""GRUP B -- UC KESME, ileri 122 gun. Capa POZITIF gecmis GEREKTIRMEZ."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import KOK, train

KESMELER = ("2025-06-30", "2025-08-31", "2025-11-30")
UFUK = 122
V83_LP_GRUPB = 5.4787  # v83'un gercek grup B'ye yazdigi ort log1p
V89_LP_GRUPB = 0.7836


def main() -> int:
    tr = train()
    tr["lp"] = np.log1p(tr["tuketim"].clip(lower=0))
    rap: dict = {"kesmeler": {}}

    for k in KESMELER:
        T = pd.Timestamp(k)
        yt = tr[tr["tarih"] < T]
        W = tr[(tr["tarih"] >= T) & (tr["tarih"] < T + pd.Timedelta(days=UFUK))]
        gy = yt.groupby("tanim")
        hepsi_sifir = gy["tuketim"].max() <= 0
        sayi = gy.size() >= 60
        son = gy["tarih"].max()
        kesilmis = son < T - pd.Timedelta(days=5)
        aday = set((hepsi_sifir & sayi & kesilmis).pipe(lambda s: s.index[s]))
        donen = aday & set(W["tanim"].unique())
        w = W[W["tanim"].isin(donen)]
        # kaybolan: donmeyenler (testte KARSILIGI YOK ama nufusu gostersin)
        blok = {
            "kesme": k,
            "aday_olu_kesilmis_trafo": len(aday),
            "GERI_DONEN_trafo": len(donen),
            "donmeyen_trafo": len(aday) - len(donen),
            "donus_satir": int(len(w)),
            "donus_sifir_orani": float((w["tuketim"] <= 0).mean()) if len(w) else float("nan"),
            "donus_ort_lp": float(w["lp"].mean()) if len(w) else float("nan"),
            "trafo_bazli_lp": {},
        }
        if len(w):
            per = w.groupby("tanim").agg(
                n=("lp", "size"),
                lp=("lp", "mean"),
                sifir=("tuketim", lambda s: float((s <= 0).mean())),
            )
            blok["trafo_bazli_lp"] = {
                t: {"n": int(r["n"]), "lp": float(r["lp"]), "sifir": float(r["sifir"])}
                for t, r in per.iterrows()
            }
            blok["trafo_cogunlukla_sifir_orani"] = float((per["sifir"] > 0.5).mean())
            # v83 seviyesine gore optimum ek kaydirma
            blok["delta_yildiz_v83e_gore"] = float(w["lp"].mean() - V83_LP_GRUPB)
            # katsayi taramasi: v83 tabanina delta ekle
            tara = {}
            taban = float(((w["lp"] - V83_LP_GRUPB) ** 2).mean())
            for dl in np.round(np.arange(-1.0, 1.01, 0.1), 2):
                tara[float(dl)] = float(((w["lp"] - V83_LP_GRUPB - dl) ** 2).mean() - taban)
            blok["tarama_v83_uzerine"] = tara
        rap["kesmeler"][k] = blok
        print(
            f"[{k}] aday(olu+kesilmis)={len(aday):3d}  GERI DONEN={len(donen):2d} "
            f"donmeyen={len(aday) - len(donen):3d} | satir={len(w):4d} "
            f"sifir%={100 * blok['donus_sifir_orani']:5.1f} lp={blok['donus_ort_lp']:6.3f} "
            f"delta*(v83'e gore)={blok.get('delta_yildiz_v83e_gore', float('nan')):+.3f}"
            if len(w)
            else f"[{k}] aday={len(aday)} GERI DONEN=0 -> OLCUM YOK"
        )
        if len(w):
            for t, d in blok["trafo_bazli_lp"].items():
                print(f"      {t}: n={d['n']:3d} lp={d['lp']:.3f} sifir%={100 * d['sifir']:.0f}")

    print(f"\nv83 grup B lp = {V83_LP_GRUPB:.4f} | v89 grup B lp = {V89_LP_GRUPB:.4f}")
    (KOK / "reports/g3f_kesmeler.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("yazildi: reports/g3f_kesmeler.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
