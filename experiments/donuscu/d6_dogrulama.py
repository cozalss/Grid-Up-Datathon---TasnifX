"""D6 -- KOSULLAMA DOGRULAMASI ve TOPLU-DONUS RISKI.

Uyari: ileri pencerede donus orani ~%4, testte ~%60. Test paneli 2026-05-11'de
TOPLU yenilenmis. Toplu donen trafo, organik donenle ayni mi davraniyor?

1) T1/T2 nufusunun test'e GIRIS tarihleri (toplu mu, organik mi)
2) train icindeki TOPLU donus gunlerinde (>=20 trafo ayni gun donuyor)
   T1-benzeri adaylarin gercek ofseti -- kontrolle kiyas
3) T1/T2 ileri pencere orneklemi: hangi trafolar, hangi gun dondu
4) organik (tekil) vs toplu donus ayrimi
"""

from __future__ import annotations

import json

import pandas as pd
from ortak import CIK, UFUK, hizala, lp, test, trafo_ozet, train

TEST_T = pd.Timestamp("2026-04-01")
TRAIN_SON = pd.Timestamp("2026-03-31")
KESMELER = ("2025-09-30", "2025-10-31", "2025-11-30", "2025-12-31", "2025-07-31", "2025-08-31")


def ozet(tr, T):  # noqa: ANN001, ANN201
    oz = trafo_ozet(tr, T)
    d = tr if T is None else tr[tr["tarih"] < T]
    s60 = d.sort_values(["tanim", "tarih"]).groupby("tanim").tail(60)
    oz["son60_ofs"] = s60.groupby("tanim")["ofs"].mean()
    return oz


def t1(oz, T):  # noqa: ANN001, ANN201
    b = (T - oz["son_tarih"]).dt.days
    return set(oz.index[(b >= 90) & (oz["son60_ofs"] <= -2.0) & (oz["n_kayit"] >= 30)])


def t2(oz, T):  # noqa: ANN001, ANN201
    b = (T - oz["son_tarih"]).dt.days
    return set(oz.index[(b >= 180) & (oz["son60_ofs"] <= 0.0)]) - t1(oz, T)


def main() -> int:
    tr, te = train(), test()
    v102 = hizala("tuketim_v102_kappa_optimum.csv")
    te = te.copy()
    te["ofs102"] = lp(v102) - te["lg"].to_numpy()
    oz_test = ozet(tr, None)
    rap: dict = {}

    # ---- 1) test'e giris tarihleri
    tg = te.groupby("tanim")["tarih"].min()
    for ad, fn in (("T1", t1), ("T2", t2)):
        A = sorted(fn(oz_test, TEST_T) & set(tg.index))
        vc = tg.reindex(A).value_counts()
        rap[f"{ad}_test_giris"] = {str(k.date()): int(v) for k, v in vc.head(8).items()}
        rap[f"{ad}_toplu_0511_pay"] = float(vc.get(pd.Timestamp("2026-05-11"), 0) / max(len(A), 1))
        print(f"[{ad}] testte {len(A)} trafo; giris tarihleri: {rap[f'{ad}_test_giris']}")
        print(f"      2026-05-11 toplu girisin payi: {rap[f'{ad}_toplu_0511_pay']:.2f}")

    # ---- 2) train icindeki TOPLU donus gunleri
    d = tr.sort_values(["tanim", "tarih"])
    d["onceki"] = d.groupby("tanim")["tarih"].shift()
    d["bosluk"] = (d["tarih"] - d["onceki"]).dt.days
    donus = d[d["bosluk"] >= 90].copy()
    gun_say = donus["tarih"].value_counts()
    toplu_gun = set(gun_say.index[gun_say >= 20])
    donus["toplu"] = donus["tarih"].isin(toplu_gun)
    rap["toplu_donus_gunleri"] = {str(k.date()): int(v) for k, v in gun_say.head(10).items()}
    print("\n[toplu donus gunleri, bosluk>=90g] ", rap["toplu_donus_gunleri"])

    # donus gununden sonraki 122 gunun gercek ofseti (donus tipine gore)
    kayit = []
    for _, r in donus.iterrows():
        kayit.append((r["tanim"], r["tarih"], bool(r["toplu"])))
    dd = pd.DataFrame(kayit, columns=["tanim", "donus_gunu", "toplu"])
    dd = dd.drop_duplicates("tanim", keep="first")
    tr_i = tr.set_index("tanim")
    parca = []
    for _, r in dd.iterrows():
        s = tr_i.loc[[r["tanim"]]]
        s = s[
            (s["tarih"] >= r["donus_gunu"])
            & (s["tarih"] < r["donus_gunu"] + pd.Timedelta(days=UFUK))
        ]
        if len(s):
            parca.append(
                pd.DataFrame(
                    {
                        "tanim": r["tanim"],
                        "ofs": s["ofs"].to_numpy(),
                        "tuketim": s["tuketim"].to_numpy(),
                        "toplu": r["toplu"],
                        "tarih": s["tarih"].to_numpy(),
                    }
                )
            )
    P = pd.concat(parca, ignore_index=True)
    # ayni takvim gunlerinde kesintisiz raporlayanlarin ofseti = kontrol
    kontrol_gun = tr.groupby("tarih")["ofs"].mean()
    P["kontrol"] = P["tarih"].map(kontrol_gun)
    P["fazla"] = P["ofs"] - P["kontrol"]
    for tp, g in P.groupby("toplu"):
        et = "TOPLU (>=20 trafo ayni gun)" if tp else "TEKIL (organik)"
        rap[f"donus_{'toplu' if tp else 'tekil'}"] = {
            "trafo": int(g["tanim"].nunique()),
            "satir": int(len(g)),
            "sifir_orani": float((g["tuketim"] <= 0).mean()),
            "ort_ofs": float(g["ofs"].mean()),
            "ort_gun_kontrolu": float(g["kontrol"].mean()),
            "ort_FAZLA(ofs-kontrol)": float(g["fazla"].mean()),
        }
        r = rap[f"donus_{'toplu' if tp else 'tekil'}"]
        print(
            f"  {et:32s} trafo={r['trafo']:4d} satir={r['satir']:6d} "
            f"sifir%={100 * r['sifir_orani']:5.1f} ofs={r['ort_ofs']:+.4f} "
            f"kontrol={r['ort_gun_kontrolu']:+.4f} FAZLA={r['ort_FAZLA(ofs-kontrol)']:+.4f}"
        )

    # ---- 2b) ayni ayrim, YALNIZ olu-seviyeli (son60_ofs<=-2) donenlerde
    print("\n[yalniz OLU seviyeli (son60_ofs<=-2) donenler]")
    ayr = []
    for _, r in dd.iterrows():
        gecmis = tr[(tr["tanim"] == r["tanim"]) & (tr["tarih"] < r["donus_gunu"])]
        if len(gecmis) < 30:
            continue
        s60 = gecmis.sort_values("tarih").tail(60)
        ayr.append((r["tanim"], bool(r["toplu"]), float(s60["ofs"].mean())))
    AY = pd.DataFrame(ayr, columns=["tanim", "toplu", "son60_ofs"])
    olu = set(AY.loc[AY["son60_ofs"] <= -2.0, "tanim"])
    Po = P[P["tanim"].isin(olu)]
    for tp, g in Po.groupby("toplu"):
        et = "TOPLU" if tp else "TEKIL"
        rap[f"olu_donus_{'toplu' if tp else 'tekil'}"] = {
            "trafo": int(g["tanim"].nunique()),
            "satir": int(len(g)),
            "sifir_orani": float((g["tuketim"] <= 0).mean()),
            "ort_ofs": float(g["ofs"].mean()),
            "ort_FAZLA": float(g["fazla"].mean()),
        }
        r = rap[f"olu_donus_{'toplu' if tp else 'tekil'}"]
        print(
            f"  {et:8s} trafo={r['trafo']:4d} satir={r['satir']:6d} sifir%={100 * r['sifir_orani']:5.1f} "
            f"ofs={r['ort_ofs']:+.4f} FAZLA={r['ort_FAZLA']:+.4f}"
        )

    # ---- 3) ileri pencere orneklemi (hangi trafolar)
    ornek = {}
    for k in KESMELER:
        T = pd.Timestamp(k)
        oz = ozet(tr, T)
        son = min(T + pd.Timedelta(days=UFUK), TRAIN_SON + pd.Timedelta(days=1))
        W = tr[(tr["tarih"] >= T) & (tr["tarih"] < son)]
        for ad, fn in (("T1", t1), ("T2", t2)):
            w = W[W["tanim"].isin(fn(oz, T))]
            if not len(w):
                continue
            g = w.groupby("tanim").agg(
                n=("ofs", "size"),
                ofs=("ofs", "mean"),
                ilk=("tarih", "min"),
                sifir=("tuketim", lambda s: float((s <= 0).mean())),
            )
            ornek[f"{k}|{ad}"] = [
                {
                    "tanim": i,
                    "satir": int(r["n"]),
                    "ort_ofs": round(float(r["ofs"]), 3),
                    "donus_gunu": str(r["ilk"].date()),
                    "sifir_orani": round(r["sifir"], 3),
                }
                for i, r in g.iterrows()
            ]
    rap["ileri_pencere_orneklemi"] = ornek
    print("\n[ileri pencere orneklemi -- 2025-11-30 T1]")
    for e in ornek.get("2025-11-30|T1", []):
        print("   ", e)

    (CIK / "d6_dogrulama.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nyazildi: experiments/donuscu/d6_dogrulama.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
