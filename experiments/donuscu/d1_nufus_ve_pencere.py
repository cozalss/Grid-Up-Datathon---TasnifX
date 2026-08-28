"""D1 -- NUFUS TANIMI + 5 KESME ILERI PENCERE OLCUMU.

Nufus semasi (her T kesmesinde AYNI kural, sizintisiz):
    aday(T; G, Z, M) = { trafo :  son_kayit_tarihi(T) <  T - G gun
                                  ve n_kayit(T)       >= M
                                  ve Z sifir-kosulu    saglanir }
    Z0 = kosul yok            (pozitif gecmisliler DAHIL)
    Z1 = son 60 kaydin hepsi sifir
    Z2 = train'de HIC pozitif yok

    gercek nufus  = aday(2026-04-01) & test'te mevcut
    ileri pencere = aday(T)          & W=[T, T+122) icinde kaydi olan

Iki kosullama da AYNI ("donmus olma"); bu yuzden karsilastirma gecerli.

delta* uc yoldan olculur (hepsi kapasite-normalize ofset uzayinda,
ofs = log1p(tuketim) - log1p(guc)):
    HAM      : ofs_gercek(W) - ofs_v102(test nufusu)
    DUZELT   : HAM - kontrol_kaymasi
               kontrol = kesintisiz raporlayan trafolar; ayni fark onlar icin
               mevsim/seviye kaymasini verir, cikarilir
    CAPA     : [ofs_gercek(W) - capa_ofs(W nufusu)]  +  [capa_ofs - ofs_v102](test)
               modelden bagimsiz; capasi olan trafolarda

Kazanc olcutu: Q = n_satir/714688 (delta=1 icin), kazanc = Q * delta*^2.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import CIK, KOK, N_TEST, UFUK, hizala, lp, test, trafo_ozet, train

TEST_T = pd.Timestamp("2026-04-01")
KESMELER = ("2025-06-30", "2025-08-31", "2025-10-31", "2025-11-30", "2025-12-31")

#: (etiket, G bosluk esigi gun, Z sifir kosulu, M asgari kayit)
NUFUSLAR = [
    ("N1  grupB (93 trafo)", None, "grupB", 0),
    ("N2a G>=5   Z2 M>=60", 5, "Z2", 60),
    ("N2b G>=30  Z2 M>=30", 30, "Z2", 30),
    ("N2c G>=30  Z1 M>=30", 30, "Z1", 30),
    ("N2d G>=60  Z1 M>=30", 60, "Z1", 30),
    ("N2e G>=120 Z1 M>=30", 120, "Z1", 30),
    ("N3a G>=200 Z0 M>=30", 200, "Z0", 30),
    ("N3b G>=150 Z0 M>=30", 150, "Z0", 30),
    ("N3c G>=120 Z0 M>=30", 120, "Z0", 30),
    ("N3d G>=90  Z0 M>=30", 90, "Z0", 30),
    ("N3e G>=60  Z0 M>=30", 60, "Z0", 30),
    ("N3f G>=45  Z0 M>=30", 45, "Z0", 30),
    ("N3g G>=30  Z0 M>=30", 30, "Z0", 30),
    ("N3h G>=20  Z0 M>=30", 20, "Z0", 30),
    ("N3i G>=10  Z0 M>=30", 10, "Z0", 30),
    ("N3j G>=60  Z0 M>=0 ", 60, "Z0", 0),
]

GRUP_B = set((KOK / "experiments/rotus_envanteri/grup_b.txt").read_text(encoding="utf-8").split())
GRUP_A = set((KOK / "experiments/rotus_envanteri/grup_a.txt").read_text(encoding="utf-8").split())


def adaylar(oz: pd.DataFrame, T: pd.Timestamp, G, Z: str, M: int) -> set[str]:
    if Z == "grupB":
        return set(oz.index) & GRUP_B
    m = (oz["son_tarih"] < T - pd.Timedelta(days=G)) & (oz["n_kayit"] >= M)
    if Z == "Z1":
        m &= oz["son60_maks"] <= 0
    elif Z == "Z2":
        m &= oz["maks_tuketim"] <= 0
    return set(oz.index[m])


def kontrol_adaylari(oz: pd.DataFrame, T: pd.Timestamp) -> set[str]:
    """Kesintisiz raporlayan trafolar: son kayit T'ye 5 gunden yakin, >=200 kayit."""
    m = (oz["son_tarih"] >= T - pd.Timedelta(days=5)) & (oz["n_kayit"] >= 200)
    return set(oz.index[m])


def main() -> int:
    tr, te = train(), test()
    v102 = hizala("tuketim_v102_kappa_optimum.csv")
    te = te.copy()
    te["ofs102"] = lp(v102) - te["lg"].to_numpy()

    print("=== trafo ozetleri hazirlaniyor ===")
    oz_test = trafo_ozet(tr)  # tam train (kesme = test baslangici)
    oz_kes = {k: trafo_ozet(tr, pd.Timestamp(k)) for k in KESMELER}
    W = {
        k: tr[
            (tr["tarih"] >= pd.Timestamp(k))
            & (tr["tarih"] < pd.Timestamp(k) + pd.Timedelta(days=UFUK))
        ]
        for k in KESMELER
    }
    for k in KESMELER:
        print(
            f"  W[{k}] satir={len(W[k]):7d} gun={W[k]['tarih'].nunique():3d} trafo={W[k]['tanim'].nunique()}"
        )

    te_g = te.groupby("tanim")
    te_ofs = te_g["ofs102"].mean()
    te_n = te_g.size()

    # ---- kontrol kaymasi (her kesme icin bir skaler)
    kontrol: dict[str, dict] = {}
    for k in KESMELER:
        T = pd.Timestamp(k)
        C = kontrol_adaylari(oz_kes[k], T)
        w = W[k][W[k]["tanim"].isin(C)]
        C2 = sorted(set(w["tanim"]) & set(te_ofs.index))
        w = w[w["tanim"].isin(C2)]
        kayma = float(w["ofs"].mean() - te.loc[te["tanim"].isin(C2), "ofs102"].mean())
        kontrol[k] = {
            "trafo": len(C2),
            "satir": int(len(w)),
            "W_ofs": float(w["ofs"].mean()),
            "test_ofs102": float(te.loc[te["tanim"].isin(C2), "ofs102"].mean()),
            "kayma": kayma,
        }
        print(
            f"[KONTROL {k}] trafo={len(C2):5d} W_ofs={kontrol[k]['W_ofs']:+.4f} "
            f"test_ofs={kontrol[k]['test_ofs102']:+.4f} kayma={kayma:+.4f}"
        )

    rap: dict = {"kontrol_kaymasi": kontrol, "nufuslar": {}}

    for etiket, G, Z, M in NUFUSLAR:
        # ---- GERCEK NUFUS (test)
        A = adaylar(oz_test, TEST_T, G, Z, M)
        P = sorted(A & set(te_ofs.index))
        n_tr, n_sat = len(P), int(te_n.reindex(P).sum())
        if n_sat == 0:
            continue
        Q = n_sat / N_TEST
        ofs102 = float(te.loc[te["tanim"].isin(P), "ofs102"].mean())
        capa_test = oz_test["capa_ofs"].reindex(P)
        kirlilik_A = len(set(P) & GRUP_A)
        icerir_B = len(set(P) & GRUP_B)
        aday_test = len(A)
        d: dict = {
            "aday_trafo": aday_test,
            "testte_trafo": n_tr,
            "donus_orani": n_tr / max(aday_test, 1),
            "test_satir": n_sat,
            "Q_delta1": Q,
            "v102_ort_ofs": ofs102,
            "grupA_kirliligi_trafo": kirlilik_A,
            "grupB_icerir_trafo": icerir_B,
            "capasi_olan_trafo": int(capa_test.notna().sum()),
            "kesmeler": {},
        }
        # capa - v102 koprusu (test tarafi), satir agirlikli
        m_capa = te["tanim"].isin(set(capa_test.dropna().index))
        if m_capa.any():
            kt = te.loc[m_capa].copy()
            kt["capa"] = kt["tanim"].map(capa_test)
            d["kopru_capa_eksi_v102"] = float((kt["capa"] - kt["ofs102"]).mean())
        else:
            d["kopru_capa_eksi_v102"] = None

        ham, duz, capa_l = [], [], []
        for k in KESMELER:
            T = pd.Timestamp(k)
            Ak = adaylar(oz_kes[k], T, G, Z, M)
            w = W[k][W[k]["tanim"].isin(Ak)]
            if w.empty:
                continue
            w_ofs = float(w["ofs"].mean())
            dh = w_ofs - ofs102
            dd = dh - kontrol[k]["kayma"]
            ham.append(dh)
            duz.append(dd)
            # capa temelli
            cp = oz_kes[k]["capa_ofs"].reindex(sorted(set(w["tanim"])))
            wc = w[w["tanim"].isin(set(cp.dropna().index))].copy()
            dc = None
            if len(wc) and d["kopru_capa_eksi_v102"] is not None:
                wc["capa"] = wc["tanim"].map(cp)
                dc = float((wc["ofs"] - wc["capa"]).mean()) + d["kopru_capa_eksi_v102"]
                capa_l.append(dc)
            d["kesmeler"][k] = {
                "aday_trafo": len(Ak),
                "donen_trafo": int(w["tanim"].nunique()),
                "donus_orani": w["tanim"].nunique() / max(len(Ak), 1),
                "satir": int(len(w)),
                "sifir_orani": float((w["tuketim"] <= 0).mean()),
                "W_ort_ofs": w_ofs,
                "W_ort_lp": float(w["lp"].mean()),
                "delta_HAM": dh,
                "delta_DUZELT": dd,
                "delta_CAPA": dc,
                "trafo_bazli_poz_oran": float((w.groupby("tanim")["ofs"].mean() > ofs102).mean()),
            }
        if not duz:
            continue
        d["delta_HAM_ort"] = float(np.mean(ham))
        d["delta_HAM_min"] = float(np.min(ham))
        d["delta_DUZELT_ort"] = float(np.mean(duz))
        d["delta_DUZELT_min"] = float(np.min(duz))
        d["delta_DUZELT_medyan"] = float(np.median(duz))
        d["delta_CAPA_ort"] = float(np.mean(capa_l)) if capa_l else None
        d["isaret_tum_kesmelerde_ayni"] = bool((np.sign(duz) == np.sign(duz[0])).all())
        for ad, val in (
            ("HAM", d["delta_HAM_ort"]),
            ("DUZELT", d["delta_DUZELT_ort"]),
            ("DUZELT_min", d["delta_DUZELT_min"]),
        ):
            d[f"kazanc_{ad}"] = Q * val * val * (1 if val > 0 else -1)
        rap["nufuslar"][etiket] = d

        print(
            f"\n[{etiket}]  trafo={n_tr:5d} satir={n_sat:7d} Q={Q:.5f} v102_ofs={ofs102:+.4f} "
            f"(grupA kirlilik {kirlilik_A}, grupB {icerir_B}, donus orani {d['donus_orani']:.2f})"
        )
        for k, r in d["kesmeler"].items():
            dc_s = "  n/a" if r["delta_CAPA"] is None else format(r["delta_CAPA"], "+.3f")
            print(
                f"    {k}: donen={r['donen_trafo']:5d}/{r['aday_trafo']:5d} satir={r['satir']:6d} "
                f"sifir%={100 * r['sifir_orani']:5.1f} W_ofs={r['W_ort_ofs']:+.4f} "
                f"dHAM={r['delta_HAM']:+.3f} dDUZ={r['delta_DUZELT']:+.3f} "
                f"dCAPA={dc_s} trafo+={r['trafo_bazli_poz_oran']:.2f}"
            )
        print(
            f"    => dDUZELT ort={d['delta_DUZELT_ort']:+.4f} min={d['delta_DUZELT_min']:+.4f} "
            f"isaret_ayni={d['isaret_tum_kesmelerde_ayni']}  "
            f"kazanc(ort)={d['kazanc_DUZELT']:+.5f}  kazanc(min)={d['kazanc_DUZELT_min']:+.5f}"
        )

    (CIK / "d1_nufus.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nyazildi: experiments/donuscu/d1_nufus.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
