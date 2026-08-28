"""SOGUK trafo OLU MU? -- test zamaninda bilinen bilgiyle tahmin edilebilir mi?

Sizinti YASAK: hedef penceredeki TUKETIM degerlerinden hicbir sey turetilmez.
Izinli: guc, lokasyon, tanim(ID), hedef penceredeki VARLIK deseni (tarihler),
        gecmisten turetilen komsu/grup istatistikleri.
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
from m1_geriteste import kes, yukle

KESIMLER = ["2025-08-31", "2025-09-30", "2025-10-31", "2025-11-30"]
UFUK = 4
ESIK_OLU = 0.5
SONUC = {}


# ---------------------------------------------------------------- yardimcilar
def sayisal_id(idx):
    return pd.to_numeric(pd.Series(np.asarray(idx).astype(str)), errors="coerce").values


def hedef_kodla(anahtar_seri, ref_anahtar, ref_deger, k=10.0):
    """Duzgunlestirilmis grup ortalamasi (target encoding), GECMISTEN."""
    onsel = float(np.mean(ref_deger))
    d = pd.DataFrame({"a": np.asarray(ref_anahtar), "v": np.asarray(ref_deger, float)})
    g = d.groupby("a").v.agg(["sum", "count"])
    kod = (g["sum"] + onsel * k) / (g["count"] + k)
    return pd.Series(np.asarray(anahtar_seri)).map(kod).fillna(onsel).values


def komsu_id_istat(hedef_id, ref_id, ref_deger, k):
    """ID ekseninde +-k komsunun ortalamasi (gecmisteki trafolardan)."""
    ok = ~np.isnan(ref_id)
    ids = ref_id[ok]
    vals = np.asarray(ref_deger, float)[ok]
    o = np.argsort(ids)
    ids, vals = ids[o], vals[o]
    kum = np.concatenate([[0.0], np.cumsum(vals)])
    n = len(ids)
    out = np.full(len(hedef_id), np.nan)
    pos = np.searchsorted(ids, np.nan_to_num(hedef_id, nan=-1e18))
    for i, p in enumerate(pos):
        if np.isnan(hedef_id[i]):
            continue
        lo, hi = max(0, p - k), min(n, p + k)
        if hi > lo:
            out[i] = (kum[hi] - kum[lo]) / (hi - lo)
    return out


# ---------------------------------------------------------------- ozellikler
def ozellik_tablosu(tr, kesim, ufuk_ay=UFUK):
    kes_ts = pd.Timestamp(kesim)
    son_ts = kes_ts + pd.DateOffset(months=ufuk_ay)
    gec, hed = kes(tr, kesim, ufuk_ay)
    gec = gec.copy()
    gec["ly"] = np.log1p(gec.tuketim)
    hed = hed.copy()
    hed["ly"] = np.log1p(hed.tuketim)

    # ---------- GECMIS referanslari (sizinti yok) ----------
    g4 = gec[gec.tarih > kes_ts - pd.DateOffset(months=ufuk_ay)]
    ref = g4.groupby("tanim").agg(
        lvl=("ly", "mean"),
        n=("ly", "size"),
        guc=("guc", "first"),
        il=("il", "first"),
        bolge=("bolge", "first"),
        ilce=("ilce", "first"),
    )
    ref["olu"] = (ref.lvl < ESIK_OLU).astype(float)
    ref_id = sayisal_id(ref.index)

    # gecmiste YENI devreye girenler (soguk trafonun en yakin analogu)
    ilk_gec = gec.groupby("tanim").tarih.min()
    yeni_ad = ilk_gec[ilk_gec > kes_ts - pd.DateOffset(months=ufuk_ay)].index
    yeni = ref.reindex(yeni_ad).dropna(subset=["lvl"])

    # ---------- SOGUK trafolar: etiket + varlik deseni ----------
    sg = hed[hed.soguk]
    tem = sg.groupby("tanim")
    f = tem.agg(
        y=("ly", "mean"),
        n=("ly", "size"),
        guc=("guc", "first"),
        il=("il", "first"),
        bolge=("bolge", "first"),
        ilce=("ilce", "first"),
        ilk=("tarih", "min"),
        sonn=("tarih", "max"),
    )
    f["olu"] = (f.y < ESIK_OLU).astype(int)

    # varlik deseni
    f["ilk_gun"] = (f.ilk - kes_ts).dt.days
    f["son_gun"] = (f.sonn - kes_ts).dt.days
    f["kuyruk"] = (son_ts - f.sonn).dt.days
    f["gun_araligi"] = (f.sonn - f.ilk).dt.days + 1
    f["yogunluk"] = f.n / f.gun_araligi
    f["eksik_gun"] = f.gun_araligi - f.n
    f["kesintisiz"] = (f.eksik_gun == 0).astype(int)
    f["pencere_payi"] = f.n / max(1, (son_ts - kes_ts).days)

    # bosluk deseni
    def bosluk(v):
        d = np.diff(np.sort(v.values.astype("datetime64[D]").astype(int)))
        if len(d) == 0:
            return pd.Series({"maks_bosluk": 0, "bosluk_say": 0, "bosluk_std": 0.0})
        return pd.Series(
            {
                "maks_bosluk": float(d.max()),
                "bosluk_say": float((d > 1).sum()),
                "bosluk_std": float(d.std()),
            }
        )

    b = tem.tarih.apply(bosluk).unstack()
    f = f.join(b)
    # hafta gunu deseni
    hg = sg.assign(hg=sg.tarih.dt.dayofweek)
    f["farkli_haftagunu"] = hg.groupby("tanim").hg.nunique()
    f["haftasonu_orani"] = hg.groupby("tanim").hg.apply(lambda v: (v >= 5).mean())

    # ---------- devreye alma DALGASI ----------
    dalga = f.groupby("ilk").size()
    f["dalga_boyu"] = f.ilk.map(dalga)
    f["dalga_payi"] = f.dalga_boyu / len(f)
    di = f.groupby(["ilk", "ilce"]).size()
    f["dalga_ilce"] = pd.MultiIndex.from_frame(f[["ilk", "ilce"]]).map(di).values
    dg = f.groupby(["ilk", "guc"]).size()
    f["dalga_guc"] = pd.MultiIndex.from_frame(f[["ilk", "guc"]]).map(dg).values

    # ---------- ID ----------
    sid = sayisal_id(f.index)
    f["id_sayisal"] = np.where(np.isnan(sid), 0, 1)
    f["id_deger"] = np.nan_to_num(sid, nan=-1)
    f["id_basamak"] = np.where(
        np.isnan(sid), 0, np.log10(np.clip(np.nan_to_num(sid, nan=1), 1, None))
    )
    for d in [3, 4, 5]:
        f[f"id_blok{d}"] = np.floor(np.nan_to_num(sid, nan=-1) / 10**d)

    # ID komsulugu (GECMISTEKI trafolardan)
    for k in [3, 10, 25]:
        f[f"komsu{k}_olu"] = komsu_id_istat(sid, ref_id, ref.olu.values, k)
        f[f"komsu{k}_lvl"] = komsu_id_istat(sid, ref_id, ref.lvl.values, k)
    # SOGUK kumede ID yakinligi (etiket kullanilmaz, sadece yogunluk)
    sok = np.sort(sid[~np.isnan(sid)])
    for r in [5, 50]:
        f[f"soguk_id_yakin{r}"] = [
            0
            if np.isnan(v)
            else int(np.searchsorted(sok, v + r, "right") - np.searchsorted(sok, v - r, "left") - 1)
            for v in sid
        ]

    # ---------- grup kodlamalari (gecmisten) ----------
    for anah in ["guc", "ilce", "bolge", "il"]:
        f[f"kod_{anah}_olu"] = hedef_kodla(f[anah], ref[anah], ref.olu.values)
        f[f"kod_{anah}_lvl"] = hedef_kodla(f[anah], ref[anah], ref.lvl.values)
        if len(yeni) > 30:
            f[f"yeni_{anah}_olu"] = hedef_kodla(f[anah], yeni[anah], yeni.olu.values, k=5)
            f[f"yeni_{anah}_lvl"] = hedef_kodla(f[anah], yeni[anah], yeni.lvl.values, k=5)
    for d in [4, 5]:
        f[f"kod_blok{d}_olu"] = hedef_kodla(
            f[f"id_blok{d}"], np.floor(np.nan_to_num(ref_id, nan=-1) / 10**d), ref.olu.values
        )
        f[f"kod_blok{d}_lvl"] = hedef_kodla(
            f[f"id_blok{d}"], np.floor(np.nan_to_num(ref_id, nan=-1) / 10**d), ref.lvl.values
        )

    f["log_guc"] = np.log(f.guc.clip(lower=1))
    f["guc_sik"] = f.guc.map(ref.guc.value_counts()).fillna(0)
    f["ilce_boy"] = f.ilce.map(ref.ilce.value_counts()).fillna(0)
    f["ilce_soguk_boy"] = f.ilce.map(f.ilce.value_counts())
    f["ilce_soguk_orani"] = f.ilce_soguk_boy / (f.ilce_boy + f.ilce_soguk_boy)
    f["ayni_ilce_guc_soguk"] = (
        pd.MultiIndex.from_frame(f[["ilce", "guc"]]).map(f.groupby(["ilce", "guc"]).size()).values
    )

    f["kesim"] = kesim
    return gec, hed, f


OZ = None  # ozellik adlari (ilk tabloda belirlenecek)


def oznitelikler(f):
    at = [
        c
        for c in f.columns
        if c not in ("y", "olu", "guc", "il", "bolge", "ilce", "ilk", "sonn", "kesim", "n")
    ]
    return [c for c in at if pd.api.types.is_numeric_dtype(f[c])]


# ---------------------------------------------------------------- metrikler
def auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p, float)
    if len(np.unique(y)) < 2:
        return float("nan")
    r = pd.Series(p).rank().values
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pr_auc(y, p):
    from sklearn.metrics import average_precision_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def main():
    tr = yukle()
    tablolar, gecler, hedler = {}, {}, {}
    print("=" * 78)
    print("1) KESIM ENVANTERI")
    envanter = {}
    for k in KESIMLER:
        gec, hed, f = ozellik_tablosu(tr, k)
        tablolar[k], gecler[k], hedler[k] = f, gec, hed
        son_ts = pd.Timestamp(k) + pd.DateOffset(months=UFUK)
        tam = son_ts <= tr.tarih.max()
        envanter[k] = dict(
            pencere_son=str(son_ts.date()),
            pencere_tam=bool(tam),
            soguk_trafo=int(len(f)),
            olu_trafo=int(f.olu.sum()),
            olu_orani=float(f.olu.mean()),
            soguk_satir=int(f.n.sum()),
            olu_satir=int(f.n[f.olu == 1].sum()),
            olu_satir_payi=float(f.n[f.olu == 1].sum() / f.n.sum()),
            hedef_satir=int(len(hed)),
        )
        print(
            f"  {k} -> {son_ts.date()} tam={tam} | soguk {len(f):4d} trafo "
            f"({int(f.n.sum()):6,d} satir) | OLU {int(f.olu.sum()):3d} (%{100 * f.olu.mean():.1f}) "
            f"olu-satir payi %{100 * f.n[f.olu == 1].sum() / f.n.sum():.1f}"
        )
    SONUC["envanter"] = envanter

    # kesimler arasi trafo ortusmesi
    ort = {}
    for a in KESIMLER:
        for b in KESIMLER:
            if a < b:
                A, B = set(tablolar[a].index), set(tablolar[b].index)
                ort[f"{a}|{b}"] = dict(
                    ortak=len(A & B), b_boyut=len(B), b_ortak_pay=len(A & B) / len(B)
                )
    SONUC["kesim_ortusme"] = ort
    print("\n  kesimler arasi soguk-trafo ORTUSMESI (durustluk uyarisi):")
    for k, v in ort.items():
        print(
            f"    {k}: ortak {v['ortak']:4d} / test-kesim {v['b_boyut']:4d} = %{100 * v['b_ortak_pay']:.0f}"
        )

    ozn = oznitelikler(tablolar[KESIMLER[0]])
    ozn = [c for c in ozn if all(c in tablolar[k].columns for k in KESIMLER)]
    print(f"\n  ozellik sayisi: {len(ozn)}")

    # ------------------------------------------------------------ 2) tekil AUC
    print("\n" + "=" * 78)
    print("2) TEKIL OZELLIK AYIRT EDICILIGI (AUC, olu=1) -- her kesimde ayri")
    tekil = {}
    for c in ozn:
        a = [auc(tablolar[k].olu.values, tablolar[k][c].values) for k in KESIMLER]
        a = np.array(a, float)
        tekil[c] = dict(
            auc_kesim=[float(x) for x in a],
            ort=float(np.nanmean(a)),
            yon=float(np.nanmean(a) - 0.5),
        )
    sira = sorted(tekil.items(), key=lambda kv: -abs(kv[1]["ort"] - 0.5))
    print(f"  {'ozellik':26s} " + " ".join(f"{k[5:]:>7s}" for k in KESIMLER) + "   ORT")
    for c, v in sira[:22]:
        print(f"  {c:26s} " + " ".join(f"{x:7.3f}" for x in v["auc_kesim"]) + f"  {v['ort']:6.3f}")
    SONUC["tekil_auc"] = tekil

    # ------------------------------------------------------------ 3) siniflandirma
    print("\n" + "=" * 78)
    print("3) KESIMLER-ARASI SINIFLANDIRMA  (egit: kesim A, test: kesim B)")
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def lgbm_clf(Xtr, ytr, wtr, Xte):
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
            random_state=0,
        )
        m.fit(Xtr, ytr, sample_weight=wtr)
        return m.predict_proba(Xte)[:, 1], m

    def logreg(Xtr, ytr, Xte):
        s = StandardScaler().fit(np.nan_to_num(Xtr))
        m = LogisticRegression(max_iter=2000, C=0.3)
        m.fit(s.transform(np.nan_to_num(Xtr)), ytr)
        return m.predict_proba(s.transform(np.nan_to_num(Xte)))[:, 1]

    sinif = {}
    onsel_auc = {}
    for a in KESIMLER:
        for b in KESIMLER:
            if a == b:
                continue
            ftr, fte = tablolar[a], tablolar[b]
            Xtr, ytr = ftr[ozn].astype(float).values, ftr.olu.values
            Xte, yte = fte[ozn].astype(float).values, fte.olu.values
            if ytr.sum() < 10:
                continue
            p, mdl = lgbm_clf(Xtr, ytr, None, Xte)
            pl = logreg(Xtr, ytr, Xte)
            ayr = ~fte.index.isin(ftr.index)  # egitimde GORULMEMIS soguk trafolar
            rec = dict(
                n_tr=int(len(ftr)),
                olu_tr=int(ytr.sum()),
                n_te=int(len(fte)),
                olu_te=int(yte.sum()),
                lgb_auc=auc(yte, p),
                lgb_prauc=pr_auc(yte, p),
                lgb_taban_pr=float(yte.mean()),
                lr_auc=auc(yte, pl),
                lr_prauc=pr_auc(yte, pl),
                ayrik_n=int(ayr.sum()),
                ayrik_olu=int(yte[ayr].sum()),
                lgb_auc_ayrik=auc(yte[ayr], p[ayr]) if ayr.sum() > 10 else None,
                lgb_prauc_ayrik=pr_auc(yte[ayr], p[ayr])
                if ayr.sum() > 10 and yte[ayr].sum() > 0
                else None,
            )
            sinif[f"{a}->{b}"] = rec
            print(
                f"  {a} -> {b} | AUC {rec['lgb_auc']:.3f} (LR {rec['lr_auc']:.3f}) "
                f"PR-AUC {rec['lgb_prauc']:.3f} (taban {rec['lgb_taban_pr']:.3f}) | "
                f"AYRIK n={rec['ayrik_n']} olu={rec['ayrik_olu']} "
                f"AUC {rec['lgb_auc_ayrik'] if rec['lgb_auc_ayrik'] is None else round(rec['lgb_auc_ayrik'], 3)}"
            )
    SONUC["siniflandirma"] = sinif

    # ------- havuz egitim: gecmis kesimler -> son kesim ---------------------
    print("\n  HAVUZ egitim (onceki tum kesimler) -> test kesim:")
    havuz = {}
    for i, b in enumerate(KESIMLER):
        if i == 0:
            continue
        onc = KESIMLER[:i]
        ftr = pd.concat([tablolar[k] for k in onc])
        fte = tablolar[b]
        Xtr, ytr = ftr[ozn].astype(float).values, ftr.olu.values
        Xte, yte = fte[ozn].astype(float).values, fte.olu.values
        p, mdl = lgbm_clf(Xtr, ytr, None, Xte)
        ayr = ~fte.index.isin(set(ftr.index))
        # kalibrasyon
        kal = []
        q = pd.qcut(pd.Series(p), 5, duplicates="drop", labels=False)
        for g in sorted(pd.unique(q)):
            m = (q == g).values
            kal.append(
                dict(
                    bin=int(g),
                    n=int(m.sum()),
                    p_ort=float(p[m].mean()),
                    gercek=float(yte[m].mean()),
                )
            )
        onem = sorted(zip(ozn, mdl.feature_importances_), key=lambda t: -t[1])[:12]
        havuz[b] = dict(
            egit_kesimler=onc,
            n_tr=int(len(ftr)),
            olu_tr=int(ytr.sum()),
            auc=auc(yte, p),
            prauc=pr_auc(yte, p),
            taban_pr=float(yte.mean()),
            ayrik_n=int(ayr.sum()),
            ayrik_olu=int(yte[ayr].sum()),
            auc_ayrik=auc(yte[ayr], p[ayr]) if ayr.sum() > 10 else None,
            kalibrasyon=kal,
            onem=[(c, int(v)) for c, v in onem],
        )
        print(
            f"    {'+'.join(x[5:] for x in onc)} -> {b}: AUC {havuz[b]['auc']:.3f} "
            f"PR-AUC {havuz[b]['prauc']:.3f} (taban {havuz[b]['taban_pr']:.3f}) | "
            f"ayrik(n={int(ayr.sum())}) AUC {havuz[b]['auc_ayrik'] if havuz[b]['auc_ayrik'] is None else round(havuz[b]['auc_ayrik'], 3)}"
        )
        print(
            "      kalibrasyon: "
            + "  ".join(f"[{d['n']:3d}] p={d['p_ort']:.3f}/g={d['gercek']:.3f}" for d in kal)
        )
        print("      en onemli: " + ", ".join(c for c, _ in onem[:8]))
    SONUC["havuz_siniflandirma"] = havuz

    # ------------------------------------------------------------ 4) regresyon
    print("\n" + "=" * 78)
    print("4) SEVIYE REGRESYONU (hedef = ort log1p, kesimler-arasi, satir agirlikli)")
    reg = {}
    for i, b in enumerate(KESIMLER):
        if i == 0:
            continue
        onc = KESIMLER[:i]
        ftr = pd.concat([tablolar[k] for k in onc])
        fte = tablolar[b]
        Xtr, Xte = ftr[ozn].astype(float).values, fte[ozn].astype(float).values
        ytr, yte = ftr.y.values, fte.y.values
        wtr, wte = ftr.n.values.astype(float), fte.n.values.astype(float)
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
            random_state=0,
        )
        m.fit(Xtr, ytr, sample_weight=wtr)
        p = m.predict(Xte)
        # taban: gecmisteki guc-grubu ortalamasi
        gecb = gecler[b]
        gucm = gecb.assign(ly=np.log1p(gecb.tuketim)).groupby("guc").ly.mean()
        g0 = float(np.log1p(gecb.tuketim).mean())
        tab = fte.guc.map(gucm).fillna(g0).values

        def r2(pred):
            mu = np.average(yte, weights=wte)
            return float(
                1
                - np.average((pred - yte) ** 2, weights=wte)
                / np.average((yte - mu) ** 2, weights=wte)
            )

        def rmse(pred):
            return float(np.sqrt(np.average((pred - yte) ** 2, weights=wte)))

        reg[b] = dict(
            egit=onc,
            taban_rmse=rmse(tab),
            taban_r2=r2(tab),
            lgb_rmse=rmse(p),
            lgb_r2=r2(p),
            seviye_std=float(
                np.sqrt(np.average((yte - np.average(yte, weights=wte)) ** 2, weights=wte))
            ),
        )
        print(
            f"  {b}: taban(guc grubu) RMSE {rmse(tab):.4f} R2 {r2(tab):+.4f}  ->  "
            f"LGBM RMSE {rmse(p):.4f} R2 {r2(p):+.4f}   (seviye std {reg[b]['seviye_std']:.3f})"
        )
        tablolar[b]["_p_reg"] = p
        tablolar[b]["_p_taban"] = tab
    SONUC["regresyon"] = reg

    # -- diri-only regresyon (iki asamali harman icin m_alive)
    for i, b in enumerate(KESIMLER):
        if i == 0:
            continue
        onc = KESIMLER[:i]
        ftr = pd.concat([tablolar[k] for k in onc])
        ftr = ftr[ftr.olu == 0]
        fte = tablolar[b]
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
            random_state=0,
        )
        m.fit(ftr[ozn].astype(float).values, ftr.y.values, sample_weight=ftr.n.values.astype(float))
        tablolar[b]["_p_diri"] = m.predict(fte[ozn].astype(float).values)
        # siniflandirici olasiligi (havuz)
        Xtr = pd.concat([tablolar[k] for k in onc])
        pc, _ = lgbm_clf(
            Xtr[ozn].astype(float).values, Xtr.olu.values, None, fte[ozn].astype(float).values
        )
        tablolar[b]["_p_olu"] = pc

    # ------------------------------------------------------------ 5) satir RMSLE
    print("\n" + "=" * 78)
    print("5) SATIR DUZEYINDE RMSLE  (taban vs harman)")
    satir = {}
    for b in ["2025-10-31", "2025-11-30"]:
        gec, hed, f = gecler[b], hedler[b], tablolar[b]
        gec = gec.assign(ly=np.log1p(gec.tuketim))
        hly = np.log1p(hed.tuketim.values)
        g0 = float(gec.ly.mean())
        tm28 = gec[gec.tarih > pd.Timestamp(b) - pd.Timedelta(days=28)].groupby("tanim").ly.mean()
        tmall = gec.groupby("tanim").ly.mean()
        gucm = gec.groupby("guc").ly.mean()
        sicak_p = hed.tanim.map(tm28).fillna(hed.tanim.map(tmall)).fillna(g0).values
        sg = hed.soguk.values

        def rmsle_ile(soguk_log):
            p = np.where(sg, soguk_log, sicak_p)
            return float(np.sqrt(np.mean((p - hly) ** 2)))

        taban_soguk = hed.tanim.map(f._p_taban).fillna(hed.guc.map(gucm)).fillna(g0).values
        r_taban = rmsle_ile(taban_soguk)
        reg_soguk = hed.tanim.map(f._p_reg).fillna(g0).values
        r_reg = rmsle_ile(reg_soguk)
        p_olu = hed.tanim.map(f._p_olu).fillna(0.0).values
        p_diri = hed.tanim.map(f._p_diri).fillna(g0).values
        r_harman = rmsle_ile((1 - p_olu) * p_diri)
        # tavan: gercek olu bilgisi verilseydi
        gercek_olu = hed.tanim.map(f.olu).fillna(0).values
        r_tavan = rmsle_ile(np.where(gercek_olu == 1, 0.0, reg_soguk))

        # esik egrisi
        egri = []
        for t in [1.01, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
            sifirla = p_olu > t
            pred = np.where(sifirla, 0.0, reg_soguk)
            r = rmsle_ile(pred)
            tf = f[f._p_olu > t]
            egri.append(
                dict(
                    esik=t,
                    sifirlanan_trafo=int(len(tf)),
                    dogru=int((tf.olu == 1).sum()),
                    yanlis=int((tf.olu == 0).sum()),
                    sifirlanan_satir=int(sifirla.sum()),
                    rmsle=r,
                    kazanc=r_taban - r,
                )
            )
        satir[b] = dict(
            taban_rmsle=r_taban,
            regresyon_rmsle=r_reg,
            harman_rmsle=r_harman,
            tavan_rmsle=r_tavan,
            esik_egrisi=egri,
            en_iyi_esik=min(egri, key=lambda d: d["rmsle"]),
        )
        print(f"\n  --- {b} ---")
        print(f"    taban (guc grubu)          RMSLE {r_taban:.4f}")
        print(f"    seviye regresyonu          RMSLE {r_reg:.4f}   kazanc {r_taban - r_reg:+.4f}")
        print(
            f"    yumusak harman (1-p)*m     RMSLE {r_harman:.4f}   kazanc {r_taban - r_harman:+.4f}"
        )
        print(
            f"    TAVAN (olu bilinseydi)     RMSLE {r_tavan:.4f}   kazanc {r_taban - r_tavan:+.4f}"
        )
        print(
            f"    {'esik':>5s} {'sifirTrafo':>10s} {'dogru':>6s} {'yanlis':>6s} {'satir':>7s} {'RMSLE':>8s} {'kazanc':>8s}"
        )
        for d in egri:
            print(
                f"    {d['esik']:5.2f} {d['sifirlanan_trafo']:10d} {d['dogru']:6d} {d['yanlis']:6d} "
                f"{d['sifirlanan_satir']:7d} {d['rmsle']:8.4f} {d['kazanc']:+8.4f}"
            )
    SONUC["satir_rmsle"] = satir

    # ------------------------------------------------------------ 6) risk
    print("\n" + "=" * 78)
    print("6) RISK: bir DIRI trafoyu sifirlamanin / bir OLU'yu kacirmanin bedeli")
    risk = {}
    for b in ["2025-10-31", "2025-11-30"]:
        f = tablolar[b]
        d = f[f.olu == 0]
        o = f[f.olu == 1]
        # sifirlarsak diri trafonun satir basi kaybi = y^2 ; kacirirsak olu icin (tahmin)^2
        risk[b] = dict(
            diri_ort_seviye=float(np.average(d.y, weights=d.n)),
            diri_sifirlama_bedeli_satir=float(np.average(d.y**2, weights=d.n)),
            olu_kacirma_bedeli_satir=float(np.average(f.loc[o.index, "_p_reg"] ** 2, weights=o.n)),
            diri_satir=int(d.n.sum()),
            olu_satir=int(o.n.sum()),
            oran=float(
                np.average(f.loc[o.index, "_p_reg"] ** 2, weights=o.n)
                / np.average(d.y**2, weights=d.n)
            ),
        )
        print(
            f"  {b}: DIRI'yi sifirlamanin satir-bedeli {risk[b]['diri_sifirlama_bedeli_satir']:.2f} | "
            f"OLU'yu kacirmanin satir-bedeli {risk[b]['olu_kacirma_bedeli_satir']:.2f} | "
            f"oran {risk[b]['oran']:.2f}x"
        )
        p = np.sort(f._p_olu.values)
        print(
            f"     basabas olasilik esigi ~ {1 / (1 + risk[b]['oran']):.3f}  "
            f"(bunun ustunde p_olu olan trafo sayisi: {(f._p_olu > 1 / (1 + risk[b]['oran'])).sum()})"
        )
        risk[b]["basabas_esik"] = float(1 / (1 + risk[b]["oran"]))
        risk[b]["basabas_ustu_trafo"] = int((f._p_olu > risk[b]["basabas_esik"]).sum())
        risk[b]["p_olu_maks"] = float(f._p_olu.max())
        risk[b]["p_olu_p99"] = float(np.percentile(f._p_olu, 99))
    SONUC["risk"] = risk

    with open(os.path.join(BURA, "m7_soguk_olu.json"), "w", encoding="utf-8") as fh:
        json.dump(SONUC, fh, ensure_ascii=False, indent=1, default=float)
    print("\nyazildi: m7_soguk_olu.json")


if __name__ == "__main__":
    main()
