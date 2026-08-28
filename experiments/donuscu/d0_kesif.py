"""D0 -- temel kesif: takvim, test giris tarihleri, boslugun dagilimi,
v102'nin grup B uzerinde v83'e gore ne yaptigi (kappa=0.459 dogrulamasi)."""

from __future__ import annotations

import json

import numpy as np
from ortak import CIK, KOK, N_TEST, hizala, lp, test, trafo_ozet, train


def main() -> int:
    tr, te = train(), test()
    rap: dict = {}

    rap["train_tarih"] = [str(tr["tarih"].min().date()), str(tr["tarih"].max().date())]
    rap["test_tarih"] = [str(te["tarih"].min().date()), str(te["tarih"].max().date())]
    rap["train_satir"] = int(len(tr))
    rap["test_satir"] = int(len(te))
    rap["train_trafo"] = int(tr["tanim"].nunique())
    rap["test_trafo"] = int(te["tanim"].nunique())
    rap["gorulmemis_test_trafo"] = int(len(set(te["tanim"]) - set(tr["tanim"])))
    print("train", rap["train_tarih"], rap["train_satir"], rap["train_trafo"])
    print(
        "test ",
        rap["test_tarih"],
        rap["test_satir"],
        rap["test_trafo"],
        "gorulmemis:",
        rap["gorulmemis_test_trafo"],
    )

    # test giris tarihleri
    tg = te.groupby("tanim")["tarih"].min()
    rap["test_giris_dagilimi"] = {
        str(k.date()): int(v) for k, v in tg.value_counts().head(10).items()
    }
    print("test giris (ilk 10):", rap["test_giris_dagilimi"])

    # trafo ozeti (tam train)
    t = trafo_ozet(tr)
    t = t.reindex(sorted(set(te["tanim"]) & set(t.index)))
    t["test_giris"] = tg.reindex(t.index)
    t["test_satir"] = te.groupby("tanim").size().reindex(t.index)
    t["bosluk_gun"] = (t["test_giris"] - t["son_tarih"]).dt.days

    rap["testte_trafo_train_gecmisli"] = int(len(t))
    for esik in (0, 30, 60, 90, 120, 150, 200, 250, 300, 350):
        m = t["bosluk_gun"] >= esik
        rap[f"bosluk>={esik}g"] = {
            "trafo": int(m.sum()),
            "test_satir": int(t.loc[m, "test_satir"].sum()),
            "Q_delta1": float(t.loc[m, "test_satir"].sum() / N_TEST),
            "hic_pozitif_yok": int((m & (t["n_poz"] == 0)).sum()),
        }
        r = rap[f"bosluk>={esik}g"]
        print(
            f"  bosluk>={esik:3d}g  trafo={r['trafo']:5d}  satir={r['test_satir']:7d}  "
            f"Q(d=1)={r['Q_delta1']:.5f}  (hic-poz {r['hic_pozitif_yok']})"
        )

    # ---- v102 vs v83 grup B dogrulamasi
    B = set((KOK / "experiments/rotus_envanteri/grup_b.txt").read_text(encoding="utf-8").split())
    v83, v102 = hizala("tuketim_v83_sicak_optimum.csv"), hizala("tuketim_v102_kappa_optimum.csv")
    sel = te["tanim"].isin(B).to_numpy()
    d = lp(v102) - lp(v83)
    rap["v102_vs_v83"] = {
        "grupB_satir": int(sel.sum()),
        "grupB_ort_fark_lp": float(d[sel].mean()),
        "grupB_disi_ort_mutlak_fark": float(np.abs(d[~sel]).mean()),
        "grupB_disi_degisen_satir": int((np.abs(d[~sel]) > 1e-9).sum()),
        "v83_grupB_ort_ofs": float((lp(v83[sel]) - te.loc[sel, "lg"].to_numpy()).mean()),
        "v102_grupB_ort_ofs": float((lp(v102[sel]) - te.loc[sel, "lg"].to_numpy()).mean()),
        "Q_v102_vs_v83": float(d @ d / N_TEST),
    }
    print("v102 vs v83:", json.dumps(rap["v102_vs_v83"], indent=2))

    t.to_csv(CIK / "d0_trafo_ozet.csv", encoding="utf-8")
    (CIK / "d0_kesif.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("yazildi: experiments/donuscu/d0_kesif.json + d0_trafo_ozet.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
