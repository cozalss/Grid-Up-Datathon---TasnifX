"""D3 -- NUFUS-DELTA ODUNLESIM EGRISI (genisletilmis kesme kumesi).

D2 en iyi nufusu (G>=90, son60_ofs<=-2.0) yalniz 3 kesmede olcebildi.
Burada 9 kesme kullanilir; her nufus icin
  * kesme sayisi, donen trafo/satir
  * kontrol-duzeltilmis delta*, kesmeler arasi ortalama / min / SS
  * trafo-kumeli standart hata (tum kesmelerin havuzunda)
  * Q ve Q*delta*^2
Sonra odunlesim egrisi (nufus buyudukce Q buyur, delta kuculur) cizilir.
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
GRUP_A = set((KOK / "experiments/rotus_envanteri/grup_a.txt").read_text(encoding="utf-8").split())
GRUP_B = set((KOK / "experiments/rotus_envanteri/grup_b.txt").read_text(encoding="utf-8").split())

#: (etiket, G, seviye esigi L: son60_ofs <= L, M). L=None -> kosul yok
TANIMLAR = []
for G in (5, 30, 60, 90, 120, 180, 240):
    for L in (-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, None):
        TANIMLAR.append((G, L, 30))
TANIMLAR += [(G, L, 0) for G in (60, 90, 120, 180) for L in (-2.0, -1.0, 0.0, None)]


def ozet(tr: pd.DataFrame, T: pd.Timestamp | None) -> pd.DataFrame:
    oz = trafo_ozet(tr, T)
    d = tr if T is None else tr[tr["tarih"] < T]
    s60 = d.sort_values(["tanim", "tarih"]).groupby("tanim").tail(60)
    oz["son60_ofs"] = s60.groupby("tanim")["ofs"].mean()
    return oz


def sec(oz: pd.DataFrame, T: pd.Timestamp, G: int, L, M: int) -> set[str]:
    m = (oz["son_tarih"] < T - pd.Timedelta(days=G)) & (oz["n_kayit"] >= M)
    if L is not None:
        m &= oz["son60_ofs"] <= L
    return set(oz.index[m])


def main() -> int:
    tr, te = train(), test()
    v102 = hizala("tuketim_v102_kappa_optimum.csv")
    te = te.copy()
    te["ofs102"] = lp(v102) - te["lg"].to_numpy()
    te_ofs = te.groupby("tanim")["ofs102"].mean()
    te_n = te.groupby("tanim").size()

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
        C2 = sorted(set(w["tanim"]) & set(te_ofs.index))
        w = w[w["tanim"].isin(C2)]
        kontrol[k] = float(w["ofs"].mean() - te.loc[te["tanim"].isin(C2), "ofs102"].mean())
        print(
            f"[{k}] W gun={W[k]['tarih'].nunique():3d} satir={len(W[k]):7d}  kontrol kayma={kontrol[k]:+.4f}"
        )

    satirlar = []
    for G, L, M in TANIMLAR:
        A = sorted(sec(oz_test, TEST_T, G, L, M) & set(te_ofs.index))
        n_sat = int(te_n.reindex(A).sum()) if A else 0
        if n_sat < 1500:
            continue
        ofs102 = float(te.loc[te["tanim"].isin(A), "ofs102"].mean())
        Q = n_sat / N_TEST
        dl, nd, nr, per_trafo = [], [], [], []
        for k in KESMELER:
            T = pd.Timestamp(k)
            Ak = sec(oz_k[k], T, G, L, M)
            w = W[k][W[k]["tanim"].isin(Ak)]
            if len(w) < 30:
                continue
            dl.append(float(w["ofs"].mean()) - ofs102 - kontrol[k])
            nd.append(int(w["tanim"].nunique()))
            nr.append(int(len(w)))
            pt = w.groupby("tanim")["ofs"].mean() - ofs102 - kontrol[k]
            per_trafo.extend(pt.tolist())
        if len(dl) < 3:
            continue
        d_ort, d_min = float(np.mean(dl)), float(np.min(dl))
        se_kesme = float(np.std(dl, ddof=1) / np.sqrt(len(dl)))
        pt = np.asarray(per_trafo, dtype=float)
        se_trafo = float(pt.std(ddof=1) / np.sqrt(len(pt))) if len(pt) > 1 else np.nan
        satirlar.append(
            {
                "G": G,
                "L": L,
                "M": M,
                "test_trafo": len(A),
                "test_satir": n_sat,
                "Q": Q,
                "v102_ofs": ofs102,
                "grupB_pay": len(set(A) & GRUP_B),
                "grupA_kirlilik": len(set(A) & GRUP_A),
                "kesme": len(dl),
                "donen_trafo_top": int(sum(nd)),
                "pencere_satir_top": int(sum(nr)),
                "delta_kesmeler": [round(x, 3) for x in dl],
                "delta_ort": d_ort,
                "delta_min": d_min,
                "delta_medyan": float(np.median(dl)),
                "SE_kesmeler_arasi": se_kesme,
                "SE_trafo_kumeli": se_trafo,
                "isaret_ayni": bool((np.sign(dl) == np.sign(dl[0])).all()),
                "poz_kesme_orani": float((np.asarray(dl) > 0).mean()),
                "kazanc_ort": Q * d_ort * abs(d_ort),
                "kazanc_min": Q * d_min * abs(d_min),
                "kazanc_muhafazakar": Q * max(d_ort - se_kesme, 0.0) ** 2,
            }
        )

    tab = pd.DataFrame(satirlar).sort_values("kazanc_ort", ascending=False)
    tab.to_csv(CIK / "d3_odunlesim.csv", index=False, encoding="utf-8")
    (CIK / "d3_odunlesim.json").write_text(
        json.dumps(satirlar, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== ODUNLESIM (kazanc_ort'a gore ilk 25) ===")
    bas = (
        f"{'G':>4}{'L':>7}{'M':>4}{'trafo':>7}{'satir':>8}{'Q':>9}{'v102':>8}"
        f"{'d_ort':>8}{'d_med':>8}{'d_min':>8}{'SEk':>7}{'SEt':>7}{'+kes':>6}{'kes':>4}"
        f"{'kaz_ort':>10}{'kaz_muh':>10}"
    )
    print(bas)
    for _, r in tab.head(25).iterrows():
        print(
            f"{r['G']:>4}{str(r['L']):>7}{r['M']:>4}{r['test_trafo']:>7}{r['test_satir']:>8}"
            f"{r['Q']:>9.5f}{r['v102_ofs']:>8.3f}{r['delta_ort']:>8.3f}{r['delta_medyan']:>8.3f}"
            f"{r['delta_min']:>8.3f}{r['SE_kesmeler_arasi']:>7.3f}{r['SE_trafo_kumeli']:>7.3f}"
            f"{r['poz_kesme_orani']:>6.2f}{r['kesme']:>4}{r['kazanc_ort']:>10.5f}"
            f"{r['kazanc_muhafazakar']:>10.5f}"
        )

    print("\n=== ODUNLESIM EGRISI: her Q bandinda en iyi ===")
    print(
        f"{'Q bandi':>16}{'G':>5}{'L':>7}{'trafo':>7}{'satir':>8}{'Q':>9}{'d_ort':>8}{'kaz_ort':>10}"
    )
    kenar = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 1.0]
    for a, b in zip(kenar[:-1], kenar[1:]):
        s = tab[(tab["Q"] >= a) & (tab["Q"] < b)]
        if s.empty:
            continue
        r = s.iloc[0]
        print(
            f"{f'[{a:.3f},{b:.3f})':>16}{r['G']:>5}{str(r['L']):>7}{r['test_trafo']:>7}"
            f"{r['test_satir']:>8}{r['Q']:>9.5f}{r['delta_ort']:>8.3f}{r['kazanc_ort']:>10.5f}"
        )

    print("\nyazildi: experiments/donuscu/d3_odunlesim.csv + .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
