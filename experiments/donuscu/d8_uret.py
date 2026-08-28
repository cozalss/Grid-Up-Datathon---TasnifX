"""D8 -- IKI GONDERIM DOSYASI URETIMI (Kaggle'a HICBIR SEY GONDERMEZ).

Taban: submissions/tuketim_v102_kappa_optimum.csv
Katmanlar (d5_katman.json'dan olculmus):
    T1 cekirdek: bosluk>=90g & son60_ofs<=-2.0 & n_kayit>=30   -> delta1
    T2 genis   : bosluk>=180g & son60_ofs<=0.0 & T1'de degil   -> delta2

v111 = tam delta        (olculen delta*)
v112 = delta'nin yarisi (muhafazakar)

CIFT SAYIM NOTU: delta, v102'nin O NUFUS UZERINDEKI GERCEK ofsetine gore
olculdu (T1 icin -0.0449). v102'de grup B'ye uygulanmis olan +0.377'lik adim
zaten bu tabanin icinde; bu yuzden EK bir cikarma YAPILMAZ.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import CIK, KOK, N_TEST, SUB, hizala, lp, test, trafo_ozet, train

TEST_T = pd.Timestamp("2026-04-01")
TABAN = "tuketim_v102_kappa_optimum.csv"
V102_MSE = 1.011091
LIDER_MSE = 0.982834
DELTA = {"T1": 1.4438, "T2": 0.6378}
DOSYALAR = [("tuketim_v111_donuscu.csv", 1.0), ("tuketim_v112_donuscu_yarim.csv", 0.5)]


def ozet(tr, T):  # noqa: ANN001, ANN201
    oz = trafo_ozet(tr, T)
    d = tr if T is None else tr[tr["tarih"] < T]
    s60 = d.sort_values(["tanim", "tarih"]).groupby("tanim").tail(60)
    oz["son60_ofs"] = s60.groupby("tanim")["ofs"].mean()
    return oz


def main() -> int:
    tr, te = train(), test()
    v102 = hizala(TABAN)
    oz = ozet(tr, None)
    b = (TEST_T - oz["son_tarih"]).dt.days
    T1 = set(oz.index[(b >= 90) & (oz["son60_ofs"] <= -2.0) & (oz["n_kayit"] >= 30)])
    T2 = set(oz.index[(b >= 180) & (oz["son60_ofs"] <= 0.0)]) - T1

    m1 = te["tanim"].isin(T1).to_numpy()
    m2 = te["tanim"].isin(T2).to_numpy()
    assert not (m1 & m2).any(), "katmanlar ayrik degil"
    adim = np.zeros(len(te))
    rap: dict = {"taban": TABAN, "delta": DELTA, "dosyalar": {}}
    print(f"T1: {len(T1)} trafo / {int(m1.sum())} satir  Q={m1.sum() / N_TEST:.5f}")
    print(f"T2: {len(T2)} trafo / {int(m2.sum())} satir  Q={m2.sum() / N_TEST:.5f}")
    print(
        f"taban sifir satir (secilenlerde): T1={int((v102[m1] == 0).sum())} "
        f"T2={int((v102[m2] == 0).sum())}"
    )

    ss = pd.read_csv(KOK / "data/raw/sample_submission.csv", usecols=["id"])
    for ad, olcek in DOSYALAR:
        adim[:] = 0.0
        adim[m1] = DELTA["T1"] * olcek
        adim[m2] = DELTA["T2"] * olcek
        yeni = np.expm1(lp(v102) + adim)
        yeni = np.clip(yeni, 0.0, None)
        cik = pd.DataFrame({"id": te["id"].to_numpy(), "tuketim": yeni})
        cik = ss.merge(cik, on="id", how="left", validate="one_to_one")
        assert not cik["tuketim"].isna().any()
        (SUB / ad).write_text(cik.to_csv(index=False, lineterminator="\n"), encoding="utf-8")

        d = lp(yeni) - lp(v102)
        Q = float(d @ d / N_TEST)

        # kazanc(uygulanan s, gercek delta*) = Q_nufus*(2*s*delta - s^2)
        def kaz(gercek1: float, gercek2: float) -> float:
            s1, s2 = DELTA["T1"] * olcek, DELTA["T2"] * olcek
            q1, q2 = m1.sum() / N_TEST, m2.sum() / N_TEST
            return q1 * (2 * s1 * gercek1 - s1**2) + q2 * (2 * s2 * gercek2 - s2**2)

        bek = kaz(DELTA["T1"], DELTA["T2"])
        duyar = {}
        for g in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
            k = kaz(g, g * DELTA["T2"] / DELTA["T1"])  # T2 orantili olcekle
            duyar[f"gercek_delta_T1={g:.1f}"] = {
                "kazanc": round(k, 5),
                "MSE": round(V102_MSE - k, 6),
                "RMSLE": round(float(np.sqrt(max(V102_MSE - k, 1e-9))), 6),
            }
        bb1 = DELTA["T1"] * olcek / 2
        bb2 = DELTA["T2"] * olcek / 2
        rap["dosyalar"][ad] = {
            "olcek": olcek,
            "adim_T1": DELTA["T1"] * olcek,
            "adim_T2": DELTA["T2"] * olcek,
            "degisen_satir": int((np.abs(d) > 1e-12).sum()),
            "Q_v102ye_gore": Q,
            "ort_log1p": float(lp(yeni).mean()),
            "taban_ort_log1p": float(lp(v102).mean()),
            "BEKLENEN_kazanc": bek,
            "ON_KAYITLI_MSE": V102_MSE - bek,
            "ON_KAYITLI_RMSLE": float(np.sqrt(V102_MSE - bek)),
            "basa_bas_T1_gercek_delta": bb1,
            "basa_bas_T2_gercek_delta": bb2,
            "olculen_T1_min_kesme": 1.0142,
            "olculen_T2_min_kesme": 0.1822,
            "duyarlilik": duyar,
        }
        r = rap["dosyalar"][ad]
        print(f"\n=== {ad} (olcek {olcek}) ===")
        print(f"  adim: T1={r['adim_T1']:+.4f}  T2={r['adim_T2']:+.4f}")
        print(f"  degisen satir={r['degisen_satir']}  Q(v102'ye gore)={Q:.5f}")
        print(f"  ort log1p {r['taban_ort_log1p']:.5f} -> {r['ort_log1p']:.5f}")
        print(
            f"  ON KAYITLI: kazanc={bek:.5f}  MSE={r['ON_KAYITLI_MSE']:.6f}  "
            f"RMSLE={r['ON_KAYITLI_RMSLE']:.6f}"
        )
        print(
            f"  BASA BAS: gercek delta T1 < {bb1:.4f} ise KAYBETTIRIR "
            f"(olculen kesme min {1.0142:.4f})"
        )
        print(
            f"            gercek delta T2 < {bb2:.4f} ise KAYBETTIRIR "
            f"(olculen kesme min {0.1822:.4f})"
        )
        print("  DUYARLILIK (gercek delta T1 -> LB skoru):")
        for k, v in duyar.items():
            print(f"    {k}: kazanc={v['kazanc']:+.5f}  MSE={v['MSE']:.6f}  RMSLE={v['RMSLE']:.6f}")

    rap["lider"] = {
        "MSE": LIDER_MSE,
        "RMSLE": float(np.sqrt(LIDER_MSE)),
        "gereken_kazanc": V102_MSE - LIDER_MSE,
    }
    (CIK / "d8_uret.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nLIDER icin gereken kazanc: {rap['lider']['gereken_kazanc']:.6f}")
    print("yazildi: experiments/donuscu/d8_uret.json + iki gonderim dosyasi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
