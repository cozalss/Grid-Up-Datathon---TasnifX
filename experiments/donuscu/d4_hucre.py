"""D4 -- HUCRE AYRISTIRMASI: tek bir buyuk nufus + tek delta yerine,
AYRIK hucreler (bosluk bandi x seviye bandi) ve hucreye ozel delta.

Kazanc = toplam_c Q_c * delta_c^2  >=  en iyi tek hucre.
Boylece "nufusu buyut" fikri, delta'yi seyreltmeden uygulanir.

Her hucre 9 kesmede olculur; yalniz
   * isaret tum kesmelerde AYNI
   * delta_ort > SIGMA_KAT * SE(kesmeler arasi)
olan hucreler KULLANILIR. Digerleri sifir adim alir.
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
BOSLUK_KENAR = [5, 60, 180, 10**9]
SEVIYE_KENAR = [-np.inf, -2.0, 0.5, np.inf]
ASGARI_KAYIT = 30
SIGMA_KAT = 1.0
ASGARI_KESME = 3
ASGARI_PENCERE_SATIR = 25

GRUP_A = set((KOK / "experiments/rotus_envanteri/grup_a.txt").read_text(encoding="utf-8").split())
GRUP_B = set((KOK / "experiments/rotus_envanteri/grup_b.txt").read_text(encoding="utf-8").split())


def ozet(tr: pd.DataFrame, T: pd.Timestamp | None) -> pd.DataFrame:
    oz = trafo_ozet(tr, T)
    d = tr if T is None else tr[tr["tarih"] < T]
    s60 = d.sort_values(["tanim", "tarih"]).groupby("tanim").tail(60)
    oz["son60_ofs"] = s60.groupby("tanim")["ofs"].mean()
    return oz


def hucrele(oz: pd.DataFrame, T: pd.Timestamp) -> pd.Series:
    """Her trafoya '<bosluk bandi>|<seviye bandi>' etiketi; uygun degilse NaN."""
    bosluk = (T - oz["son_tarih"]).dt.days
    uygun = (bosluk >= BOSLUK_KENAR[0]) & (oz["n_kayit"] >= ASGARI_KAYIT) & oz["son60_ofs"].notna()
    bb = pd.cut(
        bosluk,
        BOSLUK_KENAR,
        right=False,
        labels=[f"b{a}-{b}" for a, b in zip(BOSLUK_KENAR[:-1], BOSLUK_KENAR[1:])],
    )
    sb = pd.cut(oz["son60_ofs"], SEVIYE_KENAR, labels=["s<=-2", "s-2..0.5", "s>0.5"])
    et = bb.astype(str) + "|" + sb.astype(str)
    return et.where(uygun)


def main() -> int:
    tr, te = train(), test()
    v102 = hizala("tuketim_v102_kappa_optimum.csv")
    te = te.copy()
    te["ofs102"] = lp(v102) - te["lg"].to_numpy()

    oz_test = ozet(tr, None)
    et_test = hucrele(oz_test, TEST_T)
    te["hucre"] = te["tanim"].map(et_test)

    oz_k, W, kontrol = {}, {}, {}
    te_ofs_all = te.groupby("tanim")["ofs102"].mean()
    for k in KESMELER:
        T = pd.Timestamp(k)
        oz_k[k] = ozet(tr, T)
        son = min(T + pd.Timedelta(days=UFUK), TRAIN_SON + pd.Timedelta(days=1))
        W[k] = tr[(tr["tarih"] >= T) & (tr["tarih"] < son)].copy()
        W[k]["hucre"] = W[k]["tanim"].map(hucrele(oz_k[k], T))
        oz = oz_k[k]
        C = set(oz.index[(oz["son_tarih"] >= T - pd.Timedelta(days=5)) & (oz["n_kayit"] >= 120)])
        w = W[k][W[k]["tanim"].isin(C)]
        C2 = sorted(set(w["tanim"]) & set(te_ofs_all.index))
        w = w[w["tanim"].isin(C2)]
        kontrol[k] = float(w["ofs"].mean() - te.loc[te["tanim"].isin(C2), "ofs102"].mean())

    hucreler = sorted(te["hucre"].dropna().unique())
    sonuc = []
    for h in hucreler:
        m = te["hucre"] == h
        n_sat = int(m.sum())
        if n_sat < 300:
            continue
        trf = sorted(te.loc[m, "tanim"].unique())
        ofs102 = float(te.loc[m, "ofs102"].mean())
        Q = n_sat / N_TEST
        dl, nd, nr = [], [], []
        for k in KESMELER:
            w = W[k][W[k]["hucre"] == h]
            if len(w) < ASGARI_PENCERE_SATIR:
                continue
            dl.append(float(w["ofs"].mean()) - ofs102 - kontrol[k])
            nd.append(int(w["tanim"].nunique()))
            nr.append(int(len(w)))
        if len(dl) < ASGARI_KESME:
            sonuc.append(
                {
                    "hucre": h,
                    "test_trafo": len(trf),
                    "test_satir": n_sat,
                    "Q": Q,
                    "v102_ofs": ofs102,
                    "kesme": len(dl),
                    "KULLAN": False,
                    "neden": "yetersiz kesme",
                }
            )
            continue
        d_ort = float(np.mean(dl))
        se = float(np.std(dl, ddof=1) / np.sqrt(len(dl)))
        isaret = bool((np.sign(dl) == np.sign(dl[0])).all())
        kullan = bool(isaret and d_ort > 0 and d_ort > SIGMA_KAT * se)
        sonuc.append(
            {
                "hucre": h,
                "test_trafo": len(trf),
                "test_satir": n_sat,
                "Q": Q,
                "v102_ofs": ofs102,
                "grupB_pay": len(set(trf) & GRUP_B),
                "grupA_kirlilik": len(set(trf) & GRUP_A),
                "kesme": len(dl),
                "donen_trafo_top": int(sum(nd)),
                "pencere_satir_top": int(sum(nr)),
                "delta_kesmeler": [round(x, 3) for x in dl],
                "delta_ort": d_ort,
                "delta_min": float(np.min(dl)),
                "SE": se,
                "isaret_ayni": isaret,
                "KULLAN": kullan,
                "kazanc": Q * d_ort * d_ort if kullan else 0.0,
                "kazanc_muhafazakar": Q * max(d_ort - se, 0.0) ** 2 if kullan else 0.0,
            }
        )

    tab = pd.DataFrame(sonuc)
    print(
        f"{'hucre':>22}{'trafo':>7}{'satir':>8}{'Q':>9}{'v102':>8}{'d_ort':>8}{'d_min':>8}"
        f"{'SE':>7}{'kes':>4}{'isr':>5}{'KUL':>5}{'kazanc':>10}"
    )
    for _, r in tab.sort_values("test_satir", ascending=False).iterrows():
        if isinstance(r.get("neden"), str):
            print(
                f"{r['hucre']:>22}{r['test_trafo']:>7}{r['test_satir']:>8}{r['Q']:>9.5f}"
                f"{r['v102_ofs']:>8.3f}{'':>8}{'':>8}{'':>7}{r['kesme']:>4}{'':>5}{'HAYIR':>5}"
                f"{0.0:>10.5f}  ({r['neden']})"
            )
            continue
        print(
            f"{r['hucre']:>22}{r['test_trafo']:>7}{r['test_satir']:>8}{r['Q']:>9.5f}"
            f"{r['v102_ofs']:>8.3f}{r['delta_ort']:>8.3f}{r['delta_min']:>8.3f}{r['SE']:>7.3f}"
            f"{r['kesme']:>4}{str(r['isaret_ayni'])[:3]:>5}{('EVET' if r['KULLAN'] else 'hayir'):>5}"
            f"{r['kazanc']:>10.5f}"
        )

    kul = tab[tab["KULLAN"] == True]  # noqa: E712
    top = float(kul["kazanc"].sum()) if len(kul) else 0.0
    top_m = float(kul["kazanc_muhafazakar"].sum()) if len(kul) else 0.0
    Qtop = float(kul["Q"].sum()) if len(kul) else 0.0
    print(
        f"\nKULLANILAN hucre={len(kul)}  toplam Q={Qtop:.5f}  "
        f"toplam satir={int(kul['test_satir'].sum()) if len(kul) else 0}"
    )
    print(f"TOPLAM KAZANC (delta_ort)          = {top:.5f}")
    print(f"TOPLAM KAZANC (delta_ort - 1*SE)   = {top_m:.5f}")
    v102_mse = 1.011091
    print(f"v102 MSE {v102_mse:.6f} -> {v102_mse - top:.6f}  RMSLE {np.sqrt(v102_mse - top):.6f}")
    print(
        f"                          muh -> {v102_mse - top_m:.6f}  RMSLE {np.sqrt(v102_mse - top_m):.6f}"
    )
    print(f"LIDER MSE 0.982834 icin gereken kazanc = {v102_mse - 0.982834:.6f}")

    (CIK / "d4_hucre.json").write_text(
        json.dumps(
            {
                "kontrol_kaymasi": kontrol,
                "hucreler": sonuc,
                "toplam_kazanc": top,
                "toplam_kazanc_muhafazakar": top_m,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tab.to_csv(CIK / "d4_hucre.csv", index=False, encoding="utf-8")
    print("\nyazildi: experiments/donuscu/d4_hucre.json + d4_hucre.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
