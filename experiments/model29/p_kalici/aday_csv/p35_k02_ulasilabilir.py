# -*- coding: utf-8 -*-
"""ULASILABILIR TAVAN: artigi YALNIZCA gozlenebilir oznitelklerden kestirmenin tavani.
   rho_ulasilabilir = < r , rhat/||rhat|| >_w   (rhat BLOK-DISI egitilmis)
   Ayrica SIZINTILI ust sinir: ayni blok icinde trafo-grupla capraz kestirim.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
sys.path.insert(0, os.path.join(KOK, "experiments/model29/p_kalici"))
import p27_ortak as P

CIK = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"
BLOKLAR = ("yaz25", "guz25", "kis26")
ATIL = {"tuketim", "id", "_blok", "tanim", "tarih", "lokasyon", "p", "y", "r", "w",
        "sog_cat", "sog_xgb", "sog_lgbm", "ay", "hg", "tanim_num"}


def kur(bad):
    d = P.blok(bad, soguk_harman="cat", son_islem=True).reset_index(drop=True)
    d["w"] = P.agirlik(d)
    return d


def oznitelikler(d):
    kol = []
    for c in d.columns:
        if c in ATIL:
            continue
        s = d[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            kol.append(c)
    return kol


def X(d, kol):
    return d[kol].astype("float32")


def rho_olc(d, u_ham, merkezle=False):
    """u_ham -> birim rms normlu yon; rho = agirlikli ic carpim. Kume-SE (trafo) ile."""
    w = d["w"].values; r = d["r"].values
    W = w.sum()
    u = np.asarray(u_ham, dtype=np.float64).copy()
    if merkezle:
        u -= np.sum(w * u) / W
    nrm = np.sqrt(np.sum(w * u * u) / W)
    if nrm <= 0:
        return dict(rho=0.0, se=0.0, oran=0.0, norm=0.0)
    u = u / nrm
    rho = float(np.sum(w * r * u) / W)
    v = w * (r * u - rho)
    gg = pd.DataFrame({"g": d["tanim"].values, "v": v}).groupby("g", observed=True)["v"].sum().values
    se = float(np.sqrt(np.sum(gg * gg)) / W)
    rms = float(np.sqrt(np.sum(w * r * r) / W))
    return dict(rho=rho, se=se, oran=rho / rms, norm=float(nrm))


PAR = dict(objective="regression", learning_rate=0.05, num_leaves=127,
           min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.7,
           bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=8, seed=7)
PAR_B = dict(PAR, objective="binary", metric="auc")


def egit(Xtr, ytr, wtr, par, tur=600):
    ds = lgb.Dataset(Xtr, label=ytr, weight=wtr, free_raw_data=False)
    return lgb.train(par, ds, num_boost_round=tur)


def main():
    t0 = time.time()
    D = {b: kur(b) for b in BLOKLAR}
    kol = oznitelikler(D["yaz25"])
    print("oznitelik sayisi:", len(kol)); sys.stdout.flush()
    sonuc = {"oznitelik_n": len(kol), "bloklar": {}}

    for hedef in BLOKLAR:
        dh = D[hedef]
        digerler = [b for b in BLOKLAR if b != hedef]
        tr = pd.concat([D[b] for b in digerler], ignore_index=True)
        Xtr, Xte = X(tr, kol), X(dh, kol)
        S = {}

        # --- A) GLOBAL ARTIK REGRESYONU (blok-disi) -> ulasilabilir tavan
        m = egit(Xtr, tr["r"].values, tr["w"].values, PAR, 700)
        rhat = m.predict(Xte)
        S["A_artik_regresyon"] = rho_olc(dh, rhat)
        S["A_artik_regresyon_merkezli"] = rho_olc(dh, rhat, merkezle=True)
        # sabit yon (global kayma) ayri
        S["A0_sabit"] = rho_olc(dh, np.ones(len(dh)))

        # --- B) SIFIR OLASILIGI (blok-disi siniflandirici)
        yb = (tr["tuketim"].values == 0).astype(np.int8)
        mb = egit(Xtr, yb, tr["w"].values, PAR_B, 500)
        q = mb.predict(Xte)
        yte = (dh["tuketim"].values == 0).astype(np.int8)
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(yte, q, sample_weight=dh["w"].values))
        S["B_sifir_auc"] = auc
        S["B_sifir_yumusak_q"] = rho_olc(dh, q)
        S["B_sifir_yumusak_q_merkezli"] = rho_olc(dh, q, merkezle=True)
        # esik: DIGER bloklarda en iyi |rho| veren esik (sizinti yok)
        esikler = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        skor = {e: [] for e in esikler}
        for b in digerler:
            db = D[b]
            qb = mb.predict(X(db, kol))  # NOT: mb b'yi de gordu -> sadece esik secimi icin
            for e in esikler:
                skor[e].append(abs(rho_olc(db, (qb > e).astype(float))["rho"]))
        en_iyi = max(esikler, key=lambda e: np.mean(skor[e]))
        S["B_sifir_esik_secilen"] = en_iyi
        S["B_sifir_sert_bayrak"] = rho_olc(dh, (q > en_iyi).astype(float))
        for e in (0.3, 0.5, 0.7, 0.9):
            S["B_sifir_bayrak_e%.1f" % e] = rho_olc(dh, (q > e).astype(float))
        # tahmin edilen sifir kumesinin kalitesi
        f = q > en_iyi
        S["B_sifir_kume"] = dict(n=int(f.sum()), pay=float(f.mean()),
                                 kesinlik=float(yte[f].mean()) if f.sum() else 0.0,
                                 duyarlilik=float(f[yte == 1].mean()))
        # KAHIN karsilastirmasi: gercek sifir bayragi
        S["B_sifir_KAHIN_bayrak"] = rho_olc(dh, yte.astype(float))

        # --- C) OLU TRAFO (4 ay hic uretmeyen) olasiligi, blok-disi
        def olu_maske(dd):
            tp = pd.DataFrame({"t": dd["tanim"].values, "v": dd["tuketim"].values}).groupby(
                "t", observed=True)["v"].max()
            return pd.Series(dd["tanim"].values).isin(set(tp.index[tp.values == 0])).values
        yo = olu_maske(tr).astype(np.int8)
        mo = egit(Xtr, yo, tr["w"].values, PAR_B, 500)
        qo = mo.predict(Xte)
        yo_te = olu_maske(dh).astype(np.int8)
        S["C_olu_auc"] = float(roc_auc_score(yo_te, qo, sample_weight=dh["w"].values))
        S["C_olu_yumusak"] = rho_olc(dh, qo)
        skor = {e: [] for e in esikler}
        for b in digerler:
            db = D[b]
            qb = mo.predict(X(db, kol))
            for e in esikler:
                skor[e].append(abs(rho_olc(db, (qb > e).astype(float))["rho"]))
        en_iyi_o = max(esikler, key=lambda e: np.mean(skor[e]))
        S["C_olu_esik_secilen"] = en_iyi_o
        S["C_olu_sert_bayrak"] = rho_olc(dh, (qo > en_iyi_o).astype(float))
        S["C_olu_KAHIN_bayrak"] = rho_olc(dh, yo_te.astype(float))

        # --- D) BUYUK ASAGI KACIRMA (r < -2) olasiligi
        yd = (tr["r"].values < -2).astype(np.int8)
        md = egit(Xtr, yd, tr["w"].values, PAR_B, 500)
        qd = md.predict(Xte)
        yd_te = (dh["r"].values < -2).astype(np.int8)
        S["D_asagi_auc"] = float(roc_auc_score(yd_te, qd, sample_weight=dh["w"].values))
        S["D_asagi_yumusak"] = rho_olc(dh, qd)
        S["D_asagi_bayrak_e0.5"] = rho_olc(dh, (qd > 0.5).astype(float))
        S["D_asagi_KAHIN_bayrak"] = rho_olc(dh, yd_te.astype(float))

        sonuc["bloklar"][hedef] = S
        print("[%s] bitti  t=%.0fs  A_rho=%.4f  B_auc=%.3f B_bayrak=%.4f  C_auc=%.3f C_bayrak=%.4f" % (
            hedef, time.time() - t0, S["A_artik_regresyon"]["rho"], S["B_sifir_auc"],
            S["B_sifir_sert_bayrak"]["rho"], S["C_olu_auc"], S["C_olu_sert_bayrak"]["rho"]))
        sys.stdout.flush()

    with open(os.path.join(CIK, "k02_ulasilabilir.json"), "w", encoding="utf-8") as f:
        json.dump(sonuc, f, indent=1)
    print("YAZILDI k02_ulasilabilir.json")


if __name__ == "__main__":
    main()
