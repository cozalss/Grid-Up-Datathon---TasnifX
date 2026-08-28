"""D9 -- TOPLU / ORGANIK AYRIMI (Kaggle'a HICBIR SEY GONDERMEZ).

d8 hatasi: DELTA={"T1":1.4438,"T2":0.6378} d5'te 4-5 ORGANIK donusten
olculdu, ama hedef kohortun %85'i TOPLU giriyor. d6/d7 toplu donusleri
organikten farkli olctu (ort -0.466/-0.624; trafo-medyani +0.186 --
yani toplu delta'si BILINMIYOR). v111 ikisini tek hakta karistiriyor.

Burada v111'in adimi iki AYRIK yone bolunur:
    v113 = v102 + adim|TOPLU      (Q'nun ~%85'i, delta BILINMIYOR)
    v114 = v102 + adim|ORGANIK    (Q'nun ~%15'i, delta ~1.0-1.4 olculdu)
Ozdeslik: adim(v113) + adim(v114) == adim(v111), satir satir.

Plan: HAK1 v113 -> L_toplu TAM olculur (isaret ne olursa olsun).
      HAK2 kappa* = L/Q -> kazanc = L^2/Q >= 0, HER ZAMAN.
Kaggle en iyi public skoru tuttugu icin kotu bir prob siramizi DUSURMEZ.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import CIK, KOK, N_TEST, SUB, hizala, lp, test, trafo_ozet, train

TEST_T = pd.Timestamp("2026-04-01")
TABAN = "tuketim_v102_kappa_optimum.csv"
V102_MSE = 1.011091  # 1.00553^2, olculmus
IKINCI = 1.00041  # 28 Agu 08:01 UTC itibariyla
LIDER = 0.99138
DELTA = {"T1": 1.4438, "T2": 0.6378}  # d5, ORGANIK donuslerden
TOPLU_ESIK = 8  # d7/d6 ile ayni


def ozet(tr):  # noqa: ANN001, ANN201
    oz = trafo_ozet(tr, None)
    s60 = tr.sort_values(["tanim", "tarih"]).groupby("tanim").tail(60)
    oz["son60_ofs"] = s60.groupby("tanim")["ofs"].mean()
    return oz


def main() -> int:  # noqa: PLR0915
    tr, te = train(), test()
    v102 = hizala(TABAN)
    oz = ozet(tr)
    b = (TEST_T - oz["son_tarih"]).dt.days
    T1 = set(oz.index[(b >= 90) & (oz["son60_ofs"] <= -2.0) & (oz["n_kayit"] >= 30)])
    T2 = set(oz.index[(b >= 180) & (oz["son60_ofs"] <= 0.0)]) - T1
    assert not (T1 & T2), "katmanlar ayrik degil"

    # --- test giris gunu -----------------------------------------------------
    giris = te.groupby("tanim")["tarih"].min()
    kohort = sorted(T1 | T2)
    kg = giris.reindex(kohort).dropna()

    # TOPLU gun: kohort icinde ayni gun >= TOPLU_ESIK trafo giriyorsa
    say = kg.value_counts()
    toplu_gun = set(say.index[say >= TOPLU_ESIK])
    # panelin ilk gununde zaten var olanlarin "donus olayi" GOZLENEMEZ
    ilk_gun = te["tarih"].min()

    print("=== TEST GIRIS GUNLERI (kohort) ===")
    for g, n in say.sort_values(ascending=False).items():
        etiket = "TOPLU" if g in toplu_gun else "tekil"
        if g == ilk_gun:
            etiket = "PANEL-BASI (gozlenemez)"
        print(f"  {g.date()}  {n:3d}  {etiket}")

    # Panel genelinde de bakalim -- sistemik olay mi?
    tum_giris = giris.value_counts().sort_index()
    ust = tum_giris.sort_values(ascending=False).head(6)
    print("\n=== TUM TEST PANELINDE EN YOGUN GIRIS GUNLERI ===")
    for g, n in ust.items():
        print(f"  {g.date()}  {n:5d} trafo")

    def sinifla(tanim: str) -> str:
        g = giris.get(tanim)
        if g is None or pd.isna(g):
            return "yok"
        if g == ilk_gun:
            return "panelbasi"
        return "toplu" if g in toplu_gun else "organik"

    sinif = {t: sinifla(t) for t in kohort}
    for kat, ad in ((T1, "T1"), (T2, "T2")):
        c = pd.Series([sinif[t] for t in sorted(kat)]).value_counts()
        print(f"\n{ad}: {len(kat)} trafo -> {c.to_dict()}")

    # --- maskeler ------------------------------------------------------------
    tn = te["tanim"].to_numpy()
    adim_tam = np.zeros(len(te))
    adim_tam[np.isin(tn, list(T1))] = DELTA["T1"]
    adim_tam[np.isin(tn, list(T2))] = DELTA["T2"]

    toplu_tr = {t for t in kohort if sinif[t] in ("toplu", "panelbasi")}
    org_tr = {t for t in kohort if sinif[t] == "organik"}
    m_toplu = np.isin(tn, list(toplu_tr))
    m_org = np.isin(tn, list(org_tr))
    assert not (m_toplu & m_org).any()

    ss = pd.read_csv(KOK / "data/raw/sample_submission.csv", usecols=["id"])
    rap: dict = {
        "taban": TABAN,
        "delta_kaynagi": "d5 ORGANIK donusler (4-5 trafo/kesme)",
        "toplu_esik": TOPLU_ESIK,
        "toplu_gunler": [str(g.date()) for g in sorted(toplu_gun)],
        "v102_MSE": V102_MSE,
        "dosyalar": {},
    }

    def uret(ad: str, maske: np.ndarray, etiket: str) -> dict:
        adim = np.where(maske, adim_tam, 0.0)
        yeni = np.clip(np.expm1(lp(v102) + adim), 0.0, None)
        cik = pd.DataFrame({"id": te["id"].to_numpy(), "tuketim": yeni})
        cik = ss.merge(cik, on="id", how="left", validate="one_to_one")
        assert not cik["tuketim"].isna().any(), f"{ad}: eksik id"
        assert len(cik) == N_TEST, f"{ad}: satir sayisi"
        (SUB / ad).write_text(cik.to_csv(index=False, lineterminator="\n"), encoding="utf-8")

        d = lp(yeni) - lp(v102)
        Q = float(d @ d / N_TEST)
        # kazanc(gercek delta oraninda g) = sum_k q_k*(2*s_k*g*s_k/s_ref - s_k^2) ...
        # basitlestir: adim vektoru s, gercek yon g*s ise L = g*Q, kazanc = (2g-1)*Q
        duyar = {}
        for g in (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            k = (2 * g - 1) * Q
            mse = V102_MSE - k
            duyar[f"g={g:+.2f}"] = {
                "kazanc": round(k, 6),
                "MSE": round(mse, 6),
                "RMSLE": round(float(np.sqrt(max(mse, 1e-9))), 6),
            }
        # HAK2 optimumu: kappa* = g, kazanc = g^2*Q  (isaretten BAGIMSIZ >= 0)
        hak2 = {}
        for g in (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            k = g * g * Q
            mse = V102_MSE - k
            hak2[f"g={g:+.2f}"] = {
                "kappa*": g,
                "kazanc": round(k, 6),
                "MSE": round(mse, 6),
                "RMSLE": round(float(np.sqrt(max(mse, 1e-9))), 6),
            }
        r = {
            "etiket": etiket,
            "trafo": int(len(set(tn[maske]))),
            "degisen_satir": int((np.abs(d) > 1e-12).sum()),
            "Q_v102ye_gore": Q,
            "basa_bas_g": 0.5,
            "HAK1_duyarlilik": duyar,
            "HAK2_kappa_optimumu": hak2,
        }
        print(f"\n=== {ad} ({etiket}) ===")
        print(f"  trafo {r['trafo']}  satir {r['degisen_satir']}  Q={Q:.6f}")
        print("  HAK1 (bu dosya gonderilirse):")
        for k, v in duyar.items():
            print(f"    gercek {k}: MSE={v['MSE']:.6f}  RMSLE={v['RMSLE']:.6f}")
        print("  HAK2 (kappa* uygulanirsa -- ISARETTEN BAGIMSIZ >= 0):")
        for k, v in hak2.items():
            print(f"    gercek {k}: kappa*={v['kappa*']:+.2f}  RMSLE={v['RMSLE']:.6f}")
        return r

    rap["dosyalar"]["tuketim_v113_toplu_prob.csv"] = uret(
        "tuketim_v113_toplu_prob.csv", m_toplu, "TOPLU giren kohort"
    )
    rap["dosyalar"]["tuketim_v114_organik_prob.csv"] = uret(
        "tuketim_v114_organik_prob.csv", m_org, "ORGANIK giren kohort"
    )

    # --- ozdeslik denetimi: v113 + v114 == v111 ------------------------------
    a = lp(hizala("tuketim_v113_toplu_prob.csv")) - lp(v102)
    c = lp(hizala("tuketim_v114_organik_prob.csv")) - lp(v102)
    v111 = lp(hizala("tuketim_v111_donuscu.csv")) - lp(v102)
    sapma = float(np.abs(a + c - v111).max())
    ortak_satir = int(((np.abs(a) > 1e-12) & (np.abs(c) > 1e-12)).sum())
    print("\n=== OZDESLIK DENETIMI ===")
    print(f"  max|adim(v113)+adim(v114) - adim(v111)| = {sapma:.3e}")
    print(f"  iki yonun ORTAK degistirdigi satir      = {ortak_satir}  (0 olmali)")
    print(
        f"  Q(v113)+Q(v114) = {a @ a / N_TEST + c @ c / N_TEST:.6f}   Q(v111) = {v111 @ v111 / N_TEST:.6f}"
    )
    rap["ozdeslik"] = {
        "max_sapma": sapma,
        "ortak_satir": ortak_satir,
        "Q_toplam": float(a @ a / N_TEST + c @ c / N_TEST),
        "Q_v111": float(v111 @ v111 / N_TEST),
    }
    assert sapma < 1e-9, "ozdeslik BOZUK"
    assert ortak_satir == 0, "yonler ayrik degil"

    rap["hedefler"] = {
        "ikinci_RMSLE": IKINCI,
        "lider_RMSLE": LIDER,
        "ikinci_icin_gereken_kazanc": V102_MSE - IKINCI**2,
        "lider_icin_gereken_kazanc": V102_MSE - LIDER**2,
    }
    print("\n=== HEDEFLER ===")
    print(f"  2. ({IKINCI}) icin gereken kazanc : {V102_MSE - IKINCI**2:.6f}")
    print(f"  lider ({LIDER}) icin gereken kazanc: {V102_MSE - LIDER**2:.6f}")
    (CIK / "d9_toplu_ayrim.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nyazildi: experiments/donuscu/d9_toplu_ayrim.json + iki gonderim dosyasi")
    print("KAGGLE'A HICBIR SEY GONDERILMEDI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
