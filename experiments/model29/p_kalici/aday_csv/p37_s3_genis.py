# -*- coding: utf-8 -*-
"""YON 4 s3: (A) TEST'te var olan TUM soguk gozlenebilirleri tek tek tara,
(B) blok-disi tahmin edilen GRUP kaymalari, (C) BETA ekseni supurmesi.
Parametre/esik secimi hedef bloktan YAPILMAZ."""
import json
import os
import numpy as np
import pandas as pd
from ortak import blok, ezber_maskesi, rho_olc, BLOKLAR, KOK
import p27_ortak as P

SP = os.path.dirname(os.path.abspath(__file__))
T = pd.read_parquet(os.path.join(KOK, "data/interim/deney/test.parquet"))
TEST_KOL = set(T.columns)

# TEST'te soguk satirlarda TANIMLI (NaN'siz) sayisal kolonlar
sgT = (T.soguk_mu.values == 1)
kullan = []
for c in T.columns:
    if c in ("id", "tanim", "tarih", "_blok", "soguk_mu", "tuketim"):
        continue
    if c.startswith("t_"):
        continue
    s = T[c]
    if not pd.api.types.is_numeric_dtype(s):
        continue
    v = s.values[sgT]
    if not np.isfinite(v).all():
        continue
    if np.nanstd(v) <= 0:
        continue
    kullan.append(c)
print("TEST soguk tarafta tanimli sayisal kolon sayisi:", len(kullan))


def hazirla(b):
    d = blok(b)
    w = P.agirlik(d)
    sg = (d.soguk_mu.values == 1)
    wc = w * sg
    return d, w, sg, wc


def mrk(x, sg, wc):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(len(x))
    m = np.sum(wc * x) / np.sum(wc)
    out[sg] = x[sg] - m
    return out


# ---------------------------------------------------------------- (A) TARAMA
R = {"A_tekil": {}}
D = {b: hazirla(b) for b in BLOKLAR}
EZ = {b: ezber_maskesi(b) for b in BLOKLAR}

for c in kullan:
    satir = {}
    ok = True
    for b in BLOKLAR:
        d, w, sg, wc = D[b]
        if c not in d.columns:
            ok = False
            break
        v = d[c].values.astype(np.float64)
        if not np.isfinite(v[sg]).all() or np.std(v[sg]) <= 0:
            ok = False
            break
        delta = mrk(v, sg, wc)
        o = rho_olc(d, delta, w)
        temiz = sg & (~EZ[b])
        ot = rho_olc(d, np.where(temiz, delta, 0.0), w)
        satir[b] = [round(o["rho"], 5), round(o["se"], 5), round(ot["rho"], 5)]
    if ok:
        R["A_tekil"][c] = satir

# isaret kararli + buyuk olanlari sirala (kis26 TEMIZ oldugu icin ana olcut)
tek = R["A_tekil"]


def satir_ozet(c):
    a = tek[c]
    return (a["yaz25"][0], a["yaz25"][1], a["guz25"][0], a["kis26"][0],
            a["yaz25"][2], a["guz25"][2], a["kis26"][2])


print("\n" + "=" * 112)
print("(A) TEKIL GOZLENEBILIR TARAMASI -- soguk-only, merkezli, kuresel rho")
print("Siralama: |kis26| (TEK %100 TEMIZ soguk blok) buyukten kucuge; ilk 25")
print("%-26s %9s %8s %9s %9s | %9s %9s" % ("kolon", "yaz25", "(SE)", "guz25", "kis26",
                                            "yaz25_tmz", "guz25_tmz"))
print("-" * 112)
sirali = sorted(tek.keys(), key=lambda c: -abs(tek[c]["kis26"][0]))
for c in sirali[:25]:
    y, se, g, k, yt, gt, kt = satir_ozet(c)
    print("%-26s %+9.4f %8.4f %+9.4f %+9.4f | %+9.4f %+9.4f" % (c, y, se, g, k, yt, gt))

print("\nISARET 3/3 KARARLI ve min|rho| >= 0.03 olanlar:")
kararli = []
for c in tek:
    v = [tek[c][b][0] for b in BLOKLAR]
    if (all(x > 0 for x in v) or all(x < 0 for x in v)) and min(abs(x) for x in v) >= 0.03:
        kararli.append(c)
for c in sorted(kararli, key=lambda c: -min(abs(tek[c][b][0]) for b in BLOKLAR)):
    y, se, g, k, yt, gt, kt = satir_ozet(c)
    print("  %-24s yaz25 %+.4f(+-%.4f) guz25 %+.4f kis26 %+.4f | temiz %+.4f/%+.4f"
          % (c, y, se, g, k, yt, gt))
if not kararli:
    print("  YOK")

# ------------------------------------------------- (B) BLOK-DISI GRUP KAYMASI
print("\n" + "=" * 112)
print("(B) BLOK-DISI GRUP KAYMASI (kayma katsayilari DIGER iki bloktan; hedef blokta olculur)")
print("    buzme: shift_g = (n_g/(n_g+LAM)) * agirlikli_ort_artik_g   [LAM sabit, hedeften secilmedi]")
gruplar = ["ilce_key", "bolge", "g_guc_kova", "tanim_on2", "tanim_on3", "ay", "hg", "il_key"]
LAMS = [20.0, 100.0, 500.0]
R["B_grup"] = {}
print("%-14s %6s | %-21s %-21s %-21s" % ("grup", "LAM", "yaz25", "guz25", "kis26"))
print("-" * 96)
for gk in gruplar:
    for LAM in LAMS:
        hucre = {}
        for b in BLOKLAR:
            d, w, sg, wc = D[b]
            if gk not in d.columns:
                hucre[b] = None
                continue
            # diger bloklarin SOGUK satirlarindan kayma
            par = []
            for ob in BLOKLAR:
                if ob == b:
                    continue
                do, wo, sgo, wco = D[ob]
                par.append(pd.DataFrame({
                    "g": do[gk].values[sgo], "r": do.r.values[sgo], "w": wo[sgo]}))
            pf = pd.concat(par, ignore_index=True)
            pf["wr"] = pf.w * pf.r
            ag = pf.groupby("g").agg(sw=("w", "sum"), swr=("wr", "sum"), n=("w", "size"))
            ort = ag.swr / ag.sw
            buz = ag.n / (ag.n + LAM)
            kay = (ort * buz)
            v = pd.Series(d[gk].values).map(kay).values.astype(np.float64)
            v = np.nan_to_num(v, nan=0.0)
            delta = mrk(v, sg, wc)
            if np.std(delta[sg]) <= 0:
                hucre[b] = None
                continue
            o = rho_olc(d, delta, w)
            hucre[b] = [round(o["rho"], 5), round(o["se"], 5)]
        R["B_grup"]["%s_L%d" % (gk, int(LAM))] = hucre
        f = lambda h: ("%+.4f+-%.4f" % (h[0], h[1])) if h else "--"
        print("%-14s %6.0f | %-21s %-21s %-21s"
              % (gk, LAM, f(hucre["yaz25"]), f(hucre["guz25"]), f(hucre["kis26"])))

# -------------------------------------------------------- (C) BETA EKSENI
print("\n" + "=" * 112)
print("(C) BETA EKSENI (kapasite ofset buzmesi). delta_beta = (BETA'-0.60)*(q-ort_q) soguk-only")
print("    Birim yon ayni: sign(+) = YAYILMA (beta buyur), sign(-) = BUZME (beta kucul)")
R["C_beta"] = {}
for b in BLOKLAR:
    d, w, sg, wc = D[b]
    q = d.p.values - np.log1p(d.guc.values.astype(np.float64))
    delta = mrk(q, sg, wc)
    o = rho_olc(d, delta, w)
    temiz = sg & (~EZ[b])
    ot = rho_olc(d, np.where(temiz, delta, 0.0), w)
    # dogrudan RMSLE: beta supurmesi
    tab = P.rmsle(d.y.values, d.p.values, w)
    sup = {}
    for bt in (0.0, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 1.2):
        pp = d.p.values.copy()
        sc = (bt / 0.60)
        m = np.sum(wc * q) / np.sum(wc)
        pp[sg] = m + sc * (q[sg] - m) + np.log1p(d.guc.values.astype(np.float64)[sg])
        sup["beta_%.2f" % bt] = round(P.rmsle(d.y.values, pp, w), 5)
    R["C_beta"][b] = dict(rho_yayilma=round(o["rho"], 5), se=round(o["se"], 5),
                          rho_yayilma_TEMIZ=round(ot["rho"], 5), taban_rmsle=round(tab, 5),
                          supurme=sup)
    print("  %s: rho(yayilma) %+.4f+-%.4f  TEMIZ %+.4f | RMSLE supurme %s"
          % (b, o["rho"], o["se"], ot["rho"], sup))

with open(os.path.join(SP, "s3_genis.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, indent=1)
print("\nyazildi: s3_genis.json")
