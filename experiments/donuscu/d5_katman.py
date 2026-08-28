"""D5 -- NIHAI KATMANLAR: iki AYRIK donuscu katmani, her biri kendi delta'si ile.

T1 (cekirdek) : bosluk >= 90 gun  &  son60_ofs <= -2.0  &  n_kayit >= 30
T2 (genis)    : bosluk >= 180 gun &  son60_ofs <=  0.0  &  T1'de DEGIL

Her katman 9 kesmede olculur (kontrol-duzeltilmis). Ciktilar:
  * kesme kesme delta, ortalama/min/SE, isaret tutarliligi
  * Q, Q*delta^2, basa bas noktasi
  * duyarlilik: gercek delta 0.5/1.0/1.5/2.0/2.5 iken uygulanan adimin skoru
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import CIK, KOK, N_TEST, UFUK, hizala, lp, test, trafo_ozet, train

TEST_T = pd.Timestamp("2026-04-01")
TRAIN_SON = pd.Timestamp("2026-03-31")
KESMELER = (
    "2025-06-30",
    "2025-07-31",
    "2025-08-31",
    "2025-09-30",
    "2025-10-31",
    "2025-11-30",
    "2025-12-31",
    "2026-01-31",
    "2026-02-28",
)
V102_MSE = 1.011091
LIDER_MSE = 0.982834
GRUP_A = set((KOK / "experiments/rotus_envanteri/grup_a.txt").read_text(encoding="utf-8").split())
GRUP_B = set((KOK / "experiments/rotus_envanteri/grup_b.txt").read_text(encoding="utf-8").split())


def ozet(tr: pd.DataFrame, T: pd.Timestamp | None) -> pd.DataFrame:
    oz = trafo_ozet(tr, T)
    d = tr if T is None else tr[tr["tarih"] < T]
    s60 = d.sort_values(["tanim", "tarih"]).groupby("tanim").tail(60)
    oz["son60_ofs"] = s60.groupby("tanim")["ofs"].mean()
    return oz


def t1(oz: pd.DataFrame, T: pd.Timestamp) -> set[str]:
    b = (T - oz["son_tarih"]).dt.days
    return set(oz.index[(b >= 90) & (oz["son60_ofs"] <= -2.0) & (oz["n_kayit"] >= 30)])


def t2(oz: pd.DataFrame, T: pd.Timestamp) -> set[str]:
    b = (T - oz["son_tarih"]).dt.days
    g = set(oz.index[(b >= 180) & (oz["son60_ofs"] <= 0.0)])
    return g - t1(oz, T)


KATMAN = {"T1_cekirdek": t1, "T2_genis": t2}


def main() -> int:
    tr, te = train(), test()
    v102 = hizala("tuketim_v102_kappa_optimum.csv")
    te = te.copy()
    te["ofs102"] = lp(v102) - te["lg"].to_numpy()
    te_ofs_all = te.groupby("tanim")["ofs102"].mean()

    oz_test = ozet(tr, None)
    oz_k, W, kontrol = {}, {}, {}
    for k in KESMELER:
        T = pd.Timestamp(k)
        oz_k[k] = ozet(tr, T)
        son = min(T + pd.Timedelta(days=UFUK), TRAIN_SON + pd.Timedelta(days=1))
        W[k] = tr[(tr["tarih"] >= T) & (tr["tarih"] < son)]
        oz = oz_k[k]
        C = set(oz.index[(oz["son_tarih"] >= T - pd.Timedelta(days=5)) & (oz["n_kayit"] >= 120)])
        w = W[k][W[k]["tanim"].isin(C)]
        C2 = sorted(set(w["tanim"]) & set(te_ofs_all.index))
        kontrol[k] = float(
            w[w["tanim"].isin(C2)]["ofs"].mean() - te.loc[te["tanim"].isin(C2), "ofs102"].mean()
        )

    rap: dict = {"kontrol_kaymasi": kontrol, "katmanlar": {}}
    for ad, fn in KATMAN.items():
        A = sorted(fn(oz_test, TEST_T) & set(te_ofs_all.index))
        m = te["tanim"].isin(A).to_numpy()
        n_sat = int(m.sum())
        ofs102 = float(te.loc[m, "ofs102"].mean())
        Q = n_sat / N_TEST
        kes, hepsi_ofs, hepsi_trafo = {}, [], []
        for k in KESMELER:
            T = pd.Timestamp(k)
            Ak = fn(oz_k[k], T)
            w = W[k][W[k]["tanim"].isin(Ak)]
            if len(w) < 25:
                kes[k] = {
                    "aday_trafo": len(Ak),
                    "donen_trafo": int(w["tanim"].nunique()),
                    "satir": int(len(w)),
                    "not": "yetersiz (<25 satir)",
                }
                continue
            dk = float(w["ofs"].mean()) - ofs102 - kontrol[k]
            pt = w.groupby("tanim")["ofs"].mean() - ofs102 - kontrol[k]
            hepsi_ofs.extend((w["ofs"] - kontrol[k]).tolist())
            hepsi_trafo.extend(pt.tolist())
            kes[k] = {
                "aday_trafo": len(Ak),
                "donen_trafo": int(w["tanim"].nunique()),
                "donus_orani": round(w["tanim"].nunique() / max(len(Ak), 1), 3),
                "satir": int(len(w)),
                "sifir_orani": round(float((w["tuketim"] <= 0).mean()), 4),
                "W_ort_ofs": round(float(w["ofs"].mean()), 4),
                "W_ort_lp": round(float(w["lp"].mean()), 4),
                "delta": round(dk, 4),
                "trafo_bazli_poz_oran": round(float((pt > 0).mean()), 3),
            }
        dl = [v["delta"] for v in kes.values() if "delta" in v]
        d_ort, d_min = float(np.mean(dl)), float(np.min(dl))
        se = float(np.std(dl, ddof=1) / np.sqrt(len(dl)))
        pt = np.asarray(hepsi_trafo)
        se_t = float(pt.std(ddof=1) / np.sqrt(len(pt)))
        rap["katmanlar"][ad] = {
            "test_trafo": len(A),
            "test_satir": n_sat,
            "Q_delta1": Q,
            "v102_ort_ofs": ofs102,
            "v102_ort_lp": float(lp(v102)[m].mean()),
            "grupB_pay_trafo": len(set(A) & GRUP_B),
            "grupA_kirlilik_trafo": len(set(A) & GRUP_A),
            "kesmeler": kes,
            "delta_ort": d_ort,
            "delta_min": d_min,
            "delta_medyan": float(np.median(dl)),
            "SE_kesmeler": se,
            "SE_trafo_kumeli": se_t,
            "kullanilan_kesme": len(dl),
            "isaret_ayni": bool((np.sign(dl) == np.sign(dl[0])).all()),
            "pencere_trafo_gozlem": len(pt),
            "kazanc_delta_ort": Q * d_ort**2,
            "kazanc_delta_min": Q * d_min**2 * (1 if d_min > 0 else -1),
            "kazanc_muhafazakar_1SE": Q * max(d_ort - se, 0) ** 2,
        }
        r = rap["katmanlar"][ad]
        print(f"\n=== {ad} ===")
        print(
            f"  test: {r['test_trafo']} trafo / {r['test_satir']} satir  Q={Q:.5f}  "
            f"v102_ofs={ofs102:+.4f}  v102_lp={r['v102_ort_lp']:.4f}  "
            f"(grupB {r['grupB_pay_trafo']}, grupA kirlilik {r['grupA_kirlilik_trafo']})"
        )
        for k, v in kes.items():
            if "delta" not in v:
                print(
                    f"    {k}: aday={v['aday_trafo']:5d} donen={v['donen_trafo']:4d} "
                    f"satir={v['satir']:5d}  -- {v['not']}"
                )
            else:
                print(
                    f"    {k}: aday={v['aday_trafo']:5d} donen={v['donen_trafo']:4d} "
                    f"satir={v['satir']:5d} sifir%={100 * v['sifir_orani']:5.1f} "
                    f"W_ofs={v['W_ort_ofs']:+.4f} W_lp={v['W_ort_lp']:.3f} "
                    f"delta={v['delta']:+.4f} trafo+={v['trafo_bazli_poz_oran']:.2f}"
                )
        print(
            f"  delta ort={d_ort:+.4f} medyan={r['delta_medyan']:+.4f} min={d_min:+.4f} "
            f"SE(kesme)={se:.4f} SE(trafo)={se_t:.4f} isaret_ayni={r['isaret_ayni']} "
            f"kesme={len(dl)}"
        )
        print(
            f"  kazanc: ort={r['kazanc_delta_ort']:.5f}  min={r['kazanc_delta_min']:+.5f}  "
            f"1SE-muh={r['kazanc_muhafazakar_1SE']:.5f}"
        )

    # ---- toplam + duyarlilik
    K = rap["katmanlar"]
    for ad in ("SADECE_T1", "T1+T2"):
        kl = ["T1_cekirdek"] if ad == "SADECE_T1" else ["T1_cekirdek", "T2_genis"]
        top = sum(K[a]["kazanc_delta_ort"] for a in kl)
        muh = sum(K[a]["kazanc_muhafazakar_1SE"] for a in kl)
        mn = sum(K[a]["kazanc_delta_min"] for a in kl)
        rap[ad] = {
            "Q_toplam": sum(K[a]["Q_delta1"] for a in kl),
            "satir": sum(K[a]["test_satir"] for a in kl),
            "kazanc_ort": top,
            "kazanc_muh": muh,
            "kazanc_min": mn,
            "MSE_ort": V102_MSE - top,
            "RMSLE_ort": float(np.sqrt(V102_MSE - top)),
            "MSE_muh": V102_MSE - muh,
            "RMSLE_muh": float(np.sqrt(V102_MSE - muh)),
            "MSE_min": V102_MSE - mn,
            "RMSLE_min": float(np.sqrt(V102_MSE - mn)),
        }
        r = rap[ad]
        print(
            f"\n[{ad}] Q={r['Q_toplam']:.5f} satir={r['satir']} | "
            f"kazanc ort={top:.5f} -> RMSLE {r['RMSLE_ort']:.5f} | "
            f"1SE-muh={muh:.5f} -> {r['RMSLE_muh']:.5f} | "
            f"en-kotu-kesme={mn:+.5f} -> {r['RMSLE_min']:.5f}"
        )

    print(
        f"\nLIDER: MSE {LIDER_MSE} (RMSLE {np.sqrt(LIDER_MSE):.5f}); "
        f"gereken kazanc {V102_MSE - LIDER_MSE:.6f}"
    )

    (CIK / "d5_katman.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nyazildi: experiments/donuscu/d5_katman.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
