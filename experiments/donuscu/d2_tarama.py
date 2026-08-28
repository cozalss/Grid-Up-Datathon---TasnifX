"""D2 -- (G, Z, M) IZGARASINDA nufus-delta odunlesimi + capa donusu sinavi.

A) kontrol kaymasi (M>=120 ile 2025-06-30 dahil hepsinde tanimli)
B) izgara: her nufus icin test Q, 5 kesmede kontrol-duzeltilmis delta*, Q*delta*^2
C) CAPA DONUSU: ileri pencerede donen trafo kendi eski POZITIF seviyesine mi
   donuyor? ofs_W ~ capa + son_seviye regresyonu, artik std'si
D) test tarafi: satir bazli acik = capa - ofs102; hedefli (satir bazli) atisin
   teorik tavani ve gurultu cezasi
E) 2026-05-11 kohortu ve toplu donus olayi
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import CIK, KOK, N_TEST, UFUK, hizala, lp, test, trafo_ozet, train

TEST_T = pd.Timestamp("2026-04-01")
KESMELER = ("2025-06-30", "2025-08-31", "2025-10-31", "2025-11-30", "2025-12-31")
G_LISTE = (5, 15, 30, 45, 60, 90, 120, 180)
Z_LISTE = ("Z0", "ZL", "Z1", "Z2")
M_LISTE = (0, 30, 60)

GRUP_A = set((KOK / "experiments/rotus_envanteri/grup_a.txt").read_text(encoding="utf-8").split())
GRUP_B = set((KOK / "experiments/rotus_envanteri/grup_b.txt").read_text(encoding="utf-8").split())


def maske(oz: pd.DataFrame, T: pd.Timestamp, G: int, Z: str, M: int) -> pd.Series:
    m = (oz["son_tarih"] < T - pd.Timedelta(days=G)) & (oz["n_kayit"] >= M)
    if Z == "ZL":  # son 60 kaydin ORTALAMA ofseti cok dusuk (yumusak olum)
        m &= oz["son60_ofs"] <= -2.0
    elif Z == "Z1":
        m &= oz["son60_maks"] <= 0
    elif Z == "Z2":
        m &= oz["maks_tuketim"] <= 0
    return m


def son_seviye_ekle(tr: pd.DataFrame, oz: pd.DataFrame, T: pd.Timestamp | None) -> pd.DataFrame:
    d = tr if T is None else tr[tr["tarih"] < T]
    s60 = d.sort_values(["tanim", "tarih"]).groupby("tanim").tail(60)
    oz = oz.copy()
    oz["son60_ofs"] = s60.groupby("tanim")["ofs"].mean()
    return oz


def main() -> int:
    tr, te = train(), test()
    v102 = hizala("tuketim_v102_kappa_optimum.csv")
    te = te.copy()
    te["ofs102"] = lp(v102) - te["lg"].to_numpy()

    oz_test = son_seviye_ekle(tr, trafo_ozet(tr), None)
    oz_kes = {
        k: son_seviye_ekle(tr, trafo_ozet(tr, pd.Timestamp(k)), pd.Timestamp(k)) for k in KESMELER
    }
    W = {
        k: tr[
            (tr["tarih"] >= pd.Timestamp(k))
            & (tr["tarih"] < pd.Timestamp(k) + pd.Timedelta(days=UFUK))
        ]
        for k in KESMELER
    }
    te_ofs = te.groupby("tanim")["ofs102"].mean()
    te_n = te.groupby("tanim").size()
    rap: dict = {}

    # ---------- A) KONTROL KAYMASI ----------
    kontrol = {}
    for k in KESMELER:
        T = pd.Timestamp(k)
        oz = oz_kes[k]
        C = set(oz.index[(oz["son_tarih"] >= T - pd.Timedelta(days=5)) & (oz["n_kayit"] >= 120)])
        w = W[k][W[k]["tanim"].isin(C)]
        C2 = sorted(set(w["tanim"]) & set(te_ofs.index))
        w = w[w["tanim"].isin(C2)]
        kontrol[k] = float(w["ofs"].mean() - te.loc[te["tanim"].isin(C2), "ofs102"].mean())
        print(f"[KONTROL {k}] trafo={len(C2):5d} satir={len(w):7d} kayma={kontrol[k]:+.4f}")
    rap["kontrol_kaymasi"] = kontrol

    # ---------- B) IZGARA ----------
    satirlar = []
    for G in G_LISTE:
        for Z in Z_LISTE:
            for M in M_LISTE:
                A = set(oz_test.index[maske(oz_test, TEST_T, G, Z, M)])
                P = sorted(A & set(te_ofs.index))
                n_sat = int(te_n.reindex(P).sum()) if P else 0
                if n_sat < 2000:
                    continue
                ofs102 = float(te.loc[te["tanim"].isin(P), "ofs102"].mean())
                Q = n_sat / N_TEST
                dl, n_don, n_row = [], [], []
                for k in KESMELER:
                    T = pd.Timestamp(k)
                    Ak = set(oz_kes[k].index[maske(oz_kes[k], T, G, Z, M)])
                    w = W[k][W[k]["tanim"].isin(Ak)]
                    if len(w) < 30:
                        continue
                    dl.append(float(w["ofs"].mean()) - ofs102 - kontrol[k])
                    n_don.append(int(w["tanim"].nunique()))
                    n_row.append(int(len(w)))
                if len(dl) < 3:
                    continue
                d_ort = float(np.mean(dl))
                d_min = float(np.min(dl))
                d_agr = float(np.average(dl, weights=n_row))
                satirlar.append(
                    {
                        "G": G,
                        "Z": Z,
                        "M": M,
                        "test_trafo": len(P),
                        "test_satir": n_sat,
                        "Q": Q,
                        "v102_ofs": ofs102,
                        "grupA_kirlilik": len(set(P) & GRUP_A),
                        "grupB_pay": len(set(P) & GRUP_B),
                        "kesme_sayisi": len(dl),
                        "pencere_trafo": n_don,
                        "pencere_satir": n_row,
                        "delta_kesmeler": [round(x, 4) for x in dl],
                        "delta_ort": d_ort,
                        "delta_agr": d_agr,
                        "delta_min": d_min,
                        "isaret_ayni": bool((np.sign(dl) == np.sign(dl[0])).all()),
                        "kazanc_ort": Q * d_ort * abs(d_ort),
                        "kazanc_agr": Q * d_agr * abs(d_agr),
                        "kazanc_min": Q * d_min * abs(d_min),
                    }
                )
    tab = pd.DataFrame(satirlar).sort_values("kazanc_ort", ascending=False)
    rap["izgara"] = satirlar
    print("\n=== IZGARA (kazanc_ort'a gore ilk 20) ===")
    print(
        f"{'G':>4}{'Z':>4}{'M':>4}{'trafo':>7}{'satir':>8}{'Q':>9}{'v102ofs':>9}"
        f"{'d_ort':>8}{'d_agr':>8}{'d_min':>8}{'kaz_ort':>10}{'kaz_min':>10}{'isaret':>8}{'kes':>4}"
    )
    for _, r in tab.head(20).iterrows():
        print(
            f"{r['G']:>4}{r['Z']:>4}{r['M']:>4}{r['test_trafo']:>7}{r['test_satir']:>8}"
            f"{r['Q']:>9.5f}{r['v102_ofs']:>9.4f}{r['delta_ort']:>8.3f}{r['delta_agr']:>8.3f}"
            f"{r['delta_min']:>8.3f}{r['kazanc_ort']:>10.5f}{r['kazanc_min']:>10.5f}"
            f"{str(r['isaret_ayni']):>8}{r['kesme_sayisi']:>4}"
        )

    # ---------- C) CAPA DONUSU SINAVI ----------
    print("\n=== C) CAPA DONUSU: donen trafo kendi eski pozitif seviyesine mi donuyor? ===")
    capa_rap = {}
    for k in KESMELER:
        T = pd.Timestamp(k)
        oz = oz_kes[k]
        A = set(oz.index[maske(oz, T, 30, "Z0", 30) & oz["capa_ofs"].notna()])
        w = W[k][W[k]["tanim"].isin(A)].copy()
        if len(w) < 50:
            continue
        w["capa"] = w["tanim"].map(oz["capa_ofs"])
        w["son60"] = w["tanim"].map(oz["son60_ofs"])
        r = (w["ofs"] - w["capa"]).to_numpy()
        X = np.column_stack([np.ones(len(w)), w["capa"].to_numpy(), w["son60"].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, w["ofs"].to_numpy(), rcond=None)
        art = w["ofs"].to_numpy() - X @ beta
        capa_rap[k] = {
            "trafo": int(w["tanim"].nunique()),
            "satir": int(len(w)),
            "ort(ofs_W - capa)": float(r.mean()),
            "std(ofs_W - capa)": float(r.std()),
            "medyan(ofs_W - capa)": float(np.median(r)),
            "regresyon_sabit": float(beta[0]),
            "kat_capa": float(beta[1]),
            "kat_son60": float(beta[2]),
            "artik_std": float(art.std()),
        }
        c = capa_rap[k]
        print(
            f"  {k}: n={c['satir']:6d} ort(W-capa)={c['ort(ofs_W - capa)']:+.4f} "
            f"std={c['std(ofs_W - capa)']:.3f} | reg: sabit={c['regresyon_sabit']:+.3f} "
            f"capa={c['kat_capa']:+.3f} son60={c['kat_son60']:+.3f} artik_std={c['artik_std']:.3f}"
        )
    rap["capa_donusu"] = capa_rap

    # ---------- D) SATIR BAZLI HEDEFLI ATIS TAVANI ----------
    print("\n=== D) satir bazli hedefli atis (delta_i = capa_i - ofs102_i) ===")
    hedefli = {}
    for G in (5, 30, 60, 120):
        A = set(oz_test.index[maske(oz_test, TEST_T, G, "Z0", 0) & oz_test["capa_ofs"].notna()])
        m = te["tanim"].isin(A)
        if m.sum() < 500:
            continue
        sub = te.loc[m].copy()
        sub["acik"] = sub["tanim"].map(oz_test["capa_ofs"]) - sub["ofs102"]
        a = sub["acik"].to_numpy()
        # yalniz POZITIF acik (model kendi capasinin ALTINDA yazmis)
        ap = np.where(a > 0, a, 0.0)
        art_std = float(np.mean([c["artik_std"] for c in capa_rap.values()]))
        hedefli[f"G>={G}"] = {
            "trafo": int(sub["tanim"].nunique()),
            "satir": int(len(sub)),
            "Q_satir": len(sub) / N_TEST,
            "ort_acik": float(a.mean()),
            "ort_acik_poz": float(ap.mean()),
            "TAVAN_sum(acik^2)/N": float((a * a).sum() / N_TEST),
            "TAVAN_poz_sum/N": float((ap * ap).sum() / N_TEST),
            "GURULTU_CEZASI_artikvar*n/N": float(art_std**2 * len(sub) / N_TEST),
            "NET_poz": float((ap * ap).sum() / N_TEST - art_std**2 * len(sub) / N_TEST),
        }
        h = hedefli[f"G>={G}"]
        print(
            f"  G>={G:3d}: trafo={h['trafo']:5d} satir={h['satir']:7d} ort_acik={h['ort_acik']:+.3f} "
            f"tavan={h['TAVAN_poz_sum/N']:.5f} ceza={h['GURULTU_CEZASI_artikvar*n/N']:.5f} "
            f"NET={h['NET_poz']:+.5f}"
        )
    rap["hedefli_atis"] = hedefli

    # ---------- E) 2026-05-11 KOHORTU + TOPLU DONUS ----------
    tg = te.groupby("tanim")["tarih"].min()
    k11 = set(tg.index[tg == pd.Timestamp("2026-05-11")])
    gorulmus = set(tr["tanim"].unique())
    rap["kohort_20260511"] = {
        "trafo": len(k11),
        "satir": int(te["tanim"].isin(k11).sum()),
        "train'de_gorulmus": len(k11 & gorulmus),
        "gorulmemis": len(k11 - gorulmus),
        "gorulmusler_ort_bosluk_gun": float(
            (
                pd.Timestamp("2026-05-11") - oz_test["son_tarih"].reindex(sorted(k11 & gorulmus))
            ).dt.days.mean()
        ),
        "v102_ofs_kohort": float(te.loc[te["tanim"].isin(k11), "ofs102"].mean()),
        "v102_ofs_diger": float(te.loc[~te["tanim"].isin(k11), "ofs102"].mean()),
    }
    print("\n=== E) 2026-05-11 kohortu ===")
    print(json.dumps(rap["kohort_20260511"], indent=2, ensure_ascii=False))

    # toplu donus olayi 2026-03-26 (train icinde)
    olay = pd.Timestamp("2026-03-26")
    oz_o = son_seviye_ekle(tr, trafo_ozet(tr, olay), olay)
    kes = set(oz_o.index[oz_o["son_tarih"] < olay - pd.Timedelta(days=30)])
    w = tr[(tr["tarih"] >= olay) & (tr["tanim"].isin(kes))]
    rap["olay_20260326"] = {
        "aday_kesik_trafo": len(kes),
        "donen_trafo": int(w["tanim"].nunique()),
        "satir": int(len(w)),
        "sifir_orani": float((w["tuketim"] <= 0).mean()) if len(w) else None,
        "ort_ofs": float(w["ofs"].mean()) if len(w) else None,
        "ayni_gun_kontrol_ort_ofs": float(
            tr.loc[(tr["tarih"] >= olay) & (~tr["tanim"].isin(kes)), "ofs"].mean()
        ),
    }
    print("=== E2) 2026-03-26 toplu donus ===")
    print(json.dumps(rap["olay_20260326"], indent=2, ensure_ascii=False))

    (CIK / "d2_tarama.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tab.to_csv(CIK / "d2_izgara.csv", index=False, encoding="utf-8")
    print("\nyazildi: experiments/donuscu/d2_tarama.json + d2_izgara.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
