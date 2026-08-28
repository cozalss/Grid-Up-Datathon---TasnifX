"""DURUST olcum: kesimler-arasi ORTUSEN trafolar atilir.

m7'deki kesimler-arasi kurulum kirli: 2025-09-30 soguk kumesinin %89'u 2025-10-31
soguk kumesinde de var ve hedef pencereleri buyuk olcude ORTUSUYOR. LGBM 'id_deger'
uzerinden trafoyu EZBERLEYEBILIR. Gercek testte (kesim 2026-03-31) soguk trafolarin
HICBIRI daha once gorulmemistir. Dolayisiyla tek gecerli olcum AYRIK (disjoint) alt kume.
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
import lightgbm as lgb
from m1_geriteste import yukle
from m7_soguk_olu import KESIMLER, auc, ozellik_tablosu, oznitelikler, pr_auc

SONUC = {}
# ezber riski tasiyan ham kimlik ozellikleri
KIMLIK = ["id_deger", "id_basamak", "id_blok3", "id_blok4", "id_blok5"]


def clf(Xtr, ytr, Xte, seed=0):
    m = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.04,
        num_leaves=7,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=5.0,
        verbose=-1,
        random_state=seed,
    )
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], m


def rgr(Xtr, ytr, wtr, Xte, seed=0):
    m = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=5.0,
        verbose=-1,
        random_state=seed,
    )
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m.predict(Xte), m


def main():
    tr = yukle()
    tab = {}
    for k in KESIMLER:
        gec, hed, f = ozellik_tablosu(tr, k)
        tab[k] = dict(gec=gec, hed=hed, f=f)
    ozn_tam = oznitelikler(tab[KESIMLER[0]]["f"])
    ozn_tam = [c for c in ozn_tam if all(c in tab[k]["f"].columns for k in KESIMLER)]
    ozn_temiz = [c for c in ozn_tam if c not in KIMLIK]

    # ---- test senaryolari: egitim kesim(ler)i -> test kesimi, SADECE AYRIK trafolar
    senaryolar = [
        (["2025-08-31"], "2025-11-30"),
        (["2025-08-31", "2025-09-30"], "2025-11-30"),
        (["2025-08-31"], "2025-10-31"),
        (["2025-08-31", "2025-09-30", "2025-10-31"], "2025-11-30"),
    ]

    print("=" * 84)
    print("A) AYRIK (ortusmeyen) SOGUK TRAFOLARDA SINIFLANDIRMA")
    print("   -- gercek testin karsiligi: hic gorulmemis soguk trafo --")
    A = {}
    for egit, test in senaryolar:
        ftr = pd.concat([tab[k]["f"] for k in egit])
        ftr = ftr[~ftr.index.duplicated(keep="first")]
        fte = tab[test]["f"]
        ayr = ~fte.index.isin(set(ftr.index))
        fa = fte[ayr]
        rec = {}
        for ad, ozn in [("tam", ozn_tam), ("kimliksiz", ozn_temiz)]:
            ps = []
            for s in range(5):
                p, _ = clf(
                    ftr[ozn].astype(float).values,
                    ftr.olu.values,
                    fa[ozn].astype(float).values,
                    seed=s,
                )
                ps.append(p)
            p = np.mean(ps, 0)
            rec[ad] = dict(
                auc=auc(fa.olu.values, p),
                prauc=pr_auc(fa.olu.values, p),
                taban_pr=float(fa.olu.mean()),
                p_maks=float(p.max()),
                p_ort=float(p.mean()),
            )
            fa = fa.copy()
            fa["_p_" + ad] = p
        A[f"{'+'.join(x[5:] for x in egit)}->{test}"] = dict(
            n_egit=int(len(ftr)),
            olu_egit=int(ftr.olu.sum()),
            n_ayrik=int(len(fa)),
            olu_ayrik=int(fa.olu.sum()),
            **rec,
        )
        print(
            f"  {'+'.join(x[5:] for x in egit):>17s} -> {test} | ayrik n={len(fa):4d} olu={int(fa.olu.sum()):3d} "
            f"(%{100 * fa.olu.mean():.1f}) | AUC tam {rec['tam']['auc']:.3f} / kimliksiz {rec['kimliksiz']['auc']:.3f} "
            f"| PR-AUC {rec['kimliksiz']['prauc']:.3f} (taban {rec['kimliksiz']['taban_pr']:.3f})"
        )
        tab[test].setdefault("ayrik", {})[tuple(egit)] = fa
    SONUC["ayrik_siniflandirma"] = A

    print("\n" + "=" * 84)
    print("B) AYRIK trafolarda TEKIL OZELLIK AUC (test kesim 2025-11-30, egitimde gorulmemisler)")
    ftr = pd.concat([tab[k]["f"] for k in ["2025-08-31", "2025-09-30", "2025-10-31"]])
    fa = tab["2025-11-30"]["f"]
    fa = fa[~fa.index.isin(set(ftr.index))]
    tek = {}
    for c in ozn_tam:
        tek[c] = auc(fa.olu.values, fa[c].values)
    sir = sorted(tek.items(), key=lambda kv: -abs((kv[1] if kv[1] == kv[1] else 0.5) - 0.5))
    print(f"  ayrik kume: n={len(fa)}, olu={int(fa.olu.sum())}")
    print("  en ayirt edici 12 tekil ozellik:")
    for c, v in sir[:12]:
        print(f"    {c:26s} AUC {v:.3f}")
    SONUC["ayrik_tekil_auc"] = tek
    # ayni olcum 10-31 ayrik kumesinde (capraz dogrulama)
    ftr2 = tab["2025-08-31"]["f"]
    fa2 = tab["2025-10-31"]["f"]
    fa2 = fa2[~fa2.index.isin(set(ftr2.index))]
    tek2 = {c: auc(fa2.olu.values, fa2[c].values) for c in ozn_tam}
    print(
        f"\n  ayni ozellikler 10-31 ayrik kumesinde (n={len(fa2)}, olu={int(fa2.olu.sum())}) -- ISARET TUTUYOR MU?"
    )
    for c, v in sir[:12]:
        print(
            f"    {c:26s} 11-30 AUC {v:.3f}   10-31 AUC {tek2[c]:.3f}   "
            f"{'AYNI YON' if (v - 0.5) * (tek2[c] - 0.5) > 0 else 'ISARET DONDU'}"
        )
    SONUC["ayrik_tekil_auc_1031"] = tek2

    print("\n" + "=" * 84)
    print("C) AYRIK trafolarda SEVIYE REGRESYONU (satir agirlikli R2)")
    C = {}
    for egit, test in senaryolar:
        ftr = pd.concat([tab[k]["f"] for k in egit])
        ftr = ftr[~ftr.index.duplicated(keep="first")]
        fte = tab[test]["f"]
        fa = fte[~fte.index.isin(set(ftr.index))]
        gec = tab[test]["gec"].assign(ly=lambda d: np.log1p(d.tuketim))
        gucm = gec.groupby("guc").ly.mean()
        g0 = float(gec.ly.mean())
        w = fa.n.values.astype(float)
        y = fa.y.values
        mu = np.average(y, weights=w)
        var = np.average((y - mu) ** 2, weights=w)

        def sk(p):
            return (
                float(np.sqrt(np.average((p - y) ** 2, weights=w))),
                float(1 - np.average((p - y) ** 2, weights=w) / var),
            )

        taban = fa.guc.map(gucm).fillna(g0).values
        rec = dict(n=int(len(fa)), seviye_std=float(np.sqrt(var)))
        rec["taban_rmse"], rec["taban_r2"] = sk(taban)
        for ad, ozn in [("tam", ozn_tam), ("kimliksiz", ozn_temiz)]:
            ps = [
                rgr(
                    ftr[ozn].astype(float).values,
                    ftr.y.values,
                    ftr.n.values.astype(float),
                    fa[ozn].astype(float).values,
                    s,
                )[0]
                for s in range(5)
            ]
            p = np.mean(ps, 0)
            rec[ad + "_rmse"], rec[ad + "_r2"] = sk(p)
        C[f"{'+'.join(x[5:] for x in egit)}->{test}"] = rec
        print(
            f"  {'+'.join(x[5:] for x in egit):>17s} -> {test} ayrik n={rec['n']:4d} | "
            f"taban R2 {rec['taban_r2']:+.4f} | LGBM tam R2 {rec['tam_r2']:+.4f} | "
            f"kimliksiz R2 {rec['kimliksiz_r2']:+.4f}"
        )
    SONUC["ayrik_regresyon"] = C

    print("\n" + "=" * 84)
    print("D) SATIR DUZEYINDE RMSLE -- SADECE AYRIK SOGUK SATIRLAR (durust kazanc)")
    D = {}
    for egit, test in [
        (["2025-08-31"], "2025-10-31"),
        (["2025-08-31", "2025-09-30", "2025-10-31"], "2025-11-30"),
    ]:
        ftr = pd.concat([tab[k]["f"] for k in egit])
        ftr = ftr[~ftr.index.duplicated(keep="first")]
        fte, hed, gec = tab[test]["f"], tab[test]["hed"], tab[test]["gec"]
        fa = fte[~fte.index.isin(set(ftr.index))]
        gec = gec.assign(ly=lambda d: np.log1p(d.tuketim))
        gucm = gec.groupby("guc").ly.mean()
        g0 = float(gec.ly.mean())
        m = hed.tanim.isin(set(fa.index)).values
        h = hed[m]
        hly = np.log1p(h.tuketim.values)
        taban = h.guc.map(gucm).fillna(g0).values
        pr = [
            rgr(
                ftr[ozn_temiz].astype(float).values,
                ftr.y.values,
                ftr.n.values.astype(float),
                fa[ozn_temiz].astype(float).values,
                s,
            )[0]
            for s in range(5)
        ]
        pr = np.mean(pr, 0)
        pc = [
            clf(
                ftr[ozn_temiz].astype(float).values,
                ftr.olu.values,
                fa[ozn_temiz].astype(float).values,
                s,
            )[0]
            for s in range(5)
        ]
        pc = np.mean(pc, 0)
        fad = ftr[ftr.olu == 0]
        pd_ = [
            rgr(
                fad[ozn_temiz].astype(float).values,
                fad.y.values,
                fad.n.values.astype(float),
                fa[ozn_temiz].astype(float).values,
                s,
            )[0]
            for s in range(5)
        ]
        pd_ = np.mean(pd_, 0)
        S = pd.DataFrame({"reg": pr, "p": pc, "diri": pd_}, index=fa.index)

        def R(v):
            return float(np.sqrt(np.mean((v - hly) ** 2)))

        r_tab = R(taban)
        r_reg = R(h.tanim.map(S.reg).values)
        r_har = R((1 - h.tanim.map(S.p).values) * h.tanim.map(S.diri).values)
        r_tav = R(np.where(h.tanim.map(fa.olu).values == 1, 0.0, h.tanim.map(S.reg).values))
        egri = []
        for t in [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05]:
            sf = S.p > t
            pred = np.where(h.tanim.map(sf).values, 0.0, h.tanim.map(S.reg).values)
            tf = fa.loc[sf[sf].index]
            egri.append(
                dict(
                    esik=t,
                    trafo=int(len(tf)),
                    dogru=int((tf.olu == 1).sum()),
                    yanlis=int((tf.olu == 0).sum()),
                    rmsle=R(pred),
                    kazanc=r_tab - R(pred),
                )
            )
        D[f"{'+'.join(x[5:] for x in egit)}->{test}"] = dict(
            ayrik_trafo=int(len(fa)),
            ayrik_olu=int(fa.olu.sum()),
            ayrik_satir=int(m.sum()),
            taban=r_tab,
            regresyon=r_reg,
            harman=r_har,
            tavan=r_tav,
            esik_egrisi=egri,
        )
        print(
            f"\n  --- egit {'+'.join(x[5:] for x in egit)} -> test {test} | AYRIK soguk: "
            f"{len(fa)} trafo / {int(m.sum()):,} satir / {int(fa.olu.sum())} olu ---"
        )
        print(f"    taban (guc grubu)      RMSLE {r_tab:.4f}")
        print(f"    seviye regresyonu      RMSLE {r_reg:.4f}  kazanc {r_tab - r_reg:+.4f}")
        print(f"    yumusak harman         RMSLE {r_har:.4f}  kazanc {r_tab - r_har:+.4f}")
        print(f"    TAVAN (olu bilinseydi) RMSLE {r_tav:.4f}  kazanc {r_tab - r_tav:+.4f}")
        print(
            f"    {'esik':>5s} {'trafo':>6s} {'dogru':>6s} {'yanlis':>6s} {'RMSLE':>8s} {'kazanc':>8s}"
        )
        for d in egri:
            print(
                f"    {d['esik']:5.2f} {d['trafo']:6d} {d['dogru']:6d} {d['yanlis']:6d} "
                f"{d['rmsle']:8.4f} {d['kazanc']:+8.4f}"
            )
    SONUC["ayrik_satir_rmsle"] = D

    # ---------------------------------------------------------------- E) rastgele kontrol
    print("\n" + "=" * 84)
    print("E) SIFIR-HIPOTEZI KONTROLU: etiketler karistirilinca AUC ne oluyor?")
    ftr = pd.concat([tab[k]["f"] for k in ["2025-08-31", "2025-09-30", "2025-10-31"]])
    ftr = ftr[~ftr.index.duplicated(keep="first")]
    fte = tab["2025-11-30"]["f"]
    fa = fte[~fte.index.isin(set(ftr.index))]
    rng = np.random.default_rng(0)
    bos = []
    for s in range(20):
        yk = rng.permutation(ftr.olu.values)
        p, _ = clf(
            ftr[ozn_temiz].astype(float).values, yk, fa[ozn_temiz].astype(float).values, seed=s
        )
        bos.append(auc(fa.olu.values, p))
    p, _ = clf(
        ftr[ozn_temiz].astype(float).values, ftr.olu.values, fa[ozn_temiz].astype(float).values, 0
    )
    ger = auc(fa.olu.values, p)
    print(
        f"  gercek etiketle AUC {ger:.3f} | karistirilmis etiketle AUC ort {np.mean(bos):.3f} "
        f"std {np.std(bos):.3f} (min {np.min(bos):.3f} maks {np.max(bos):.3f})"
    )
    print(f"  -> z = {(ger - np.mean(bos)) / np.std(bos):+.2f}")
    SONUC["sifir_hipotezi"] = dict(
        gercek_auc=ger,
        bos_ort=float(np.mean(bos)),
        bos_std=float(np.std(bos)),
        z=float((ger - np.mean(bos)) / np.std(bos)),
        bos_min=float(np.min(bos)),
        bos_maks=float(np.max(bos)),
    )

    yol = os.path.join(BURA, "m8_durust.json")
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(SONUC, fh, ensure_ascii=False, indent=1, default=float)
    print("\nyazildi: m8_durust.json")


if __name__ == "__main__":
    main()
