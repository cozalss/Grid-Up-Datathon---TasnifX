# -*- coding: utf-8 -*-
"""YON 4 s1: soguk-kohort aday yon taramasi. Hicbir parametre yaz25'ten secilmez."""
import json
import os
import numpy as np
import pandas as pd
from ortak import blok, ezber_maskesi, rho_olc, skor, BLOKLAR, KOK
import p27_ortak as P

SP = os.path.dirname(os.path.abspath(__file__))
GEREKEN = 0.12298


def adaylar(d):
    """cold-only delta sozlugu. TEST'te de hesaplanabilir olanlar 'T' etiketli."""
    n = len(d)
    sg = (d.soguk_mu.values == 1)
    p = d.p.values.astype(np.float64)
    guc = d.guc.values.astype(np.float64)
    lg = np.log1p(guc)
    w = P.agirlik(d)
    wc = w * sg

    def mrk(x):
        """soguk satirlarda agirlikli merkezleme, sicakta 0."""
        x = np.asarray(x, dtype=np.float64)
        m = np.sum(wc * x) / np.sum(wc)
        out = np.zeros(n)
        out[sg] = (x[sg] - m)
        return out

    A = {}
    q = p - lg  # kapasite ofset
    A["T_kapasite_BUZME"] = -mrk(q)
    A["T_kapasite_YAYILMA"] = mrk(q)
    A["T_soguk_SABIT"] = np.where(sg, 1.0, 0.0)
    A["T_guc_egim"] = mrk(lg)
    A["T_guc_egim_kare"] = mrk((lg - lg[sg].mean()) ** 2)
    A["T_p_seviye_egim"] = mrk(p)
    A["T_ilce_ort_BUZME"] = -mrk(p - d.g_ilce_log_ort.values.astype(np.float64))
    A["T_ilce_ort_YAYILMA"] = mrk(p - d.g_ilce_log_ort.values.astype(np.float64))
    A["T_kova_ort_BUZME"] = -mrk(p - d.g_kova_log_ort.values.astype(np.float64))
    A["T_cdd18"] = mrk(d.cdd18.values.astype(np.float64))
    A["T_ufuk"] = mrk(d.ufuk_gun.values.astype(np.float64))
    A["T_yas"] = mrk(np.log1p(np.maximum(d.yas.values.astype(np.float64), 0)))
    A["T_nufus_yog"] = mrk(np.log1p(d.ilce_nufus_yogunlugu.values.astype(np.float64)))
    A["T_guc_yuzdelik"] = mrk(d.guc_yuzdelik.values.astype(np.float64))
    A["T_hafta_sonu"] = mrk((d.hg.values >= 5).astype(np.float64))
    # CV-ozel (TEST'te aile tahmini yok) -- yalniz teshis
    cat = d.sog_cat.values.astype(np.float64)
    xgb = d.sog_xgb.values.astype(np.float64)
    lgb = d.sog_lgbm.values.astype(np.float64)
    ort3 = (cat + xgb + lgb) / 3.0
    ayr = np.zeros(n)
    ayr[sg] = np.std(np.c_[cat[sg], xgb[sg], lgb[sg]], axis=1)
    A["CV_aile_ayrisma"] = mrk(ayr)
    A["CV_aile_311_kayma"] = np.where(sg, 0.0, 0.0)
    tmp = np.zeros(n)
    tmp[sg] = ((3 * cat[sg] + xgb[sg] + lgb[sg]) / 5.0) - cat[sg]
    A["CV_aile_311_kayma"] = tmp
    tmp2 = np.zeros(n)
    tmp2[sg] = ort3[sg] - cat[sg]
    A["CV_aile_esit_kayma"] = tmp2
    # belirsizlik agirlikli buzme: ayrisma buyukse kapasite capasina daha cok buz
    z = np.zeros(n)
    if sg.sum():
        a = ayr[sg]
        a = (a - a.mean()) / (a.std() + 1e-12)
        z[sg] = -(q[sg] - np.sum(wc[sg] * q[sg]) / np.sum(wc[sg])) * a
    A["CV_ayrisma_x_buzme"] = z
    return A, w, sg


def main():
    R = {}
    ad_listesi = None
    for b in BLOKLAR:
        d = blok(b)
        A, w, sg = adaylar(d)
        ez = ezber_maskesi(b)
        temiz = sg & (~ez)
        kirli = sg & ez
        if ad_listesi is None:
            ad_listesi = list(A.keys())
        rb = {}
        for ad, delta in A.items():
            o = rho_olc(d, delta, w)
            # temiz/kirli ayrim: yalniz o alt kumeye kisitli yon (kuresel esdeger)
            ot = rho_olc(d, np.where(temiz, delta, 0.0), w)
            ok_ = rho_olc(d, np.where(kirli, delta, 0.0), w)
            rb[ad] = dict(
                rho=round(o["rho"], 5), se=round(o["se"], 5), t=round(o["t"], 2),
                rho_temiz=round(ot["rho"], 5), n_temiz=int(temiz.sum()),
                rho_kirli=round(ok_["rho"], 5), n_kirli=int(kirli.sum()),
            )
        R[b] = dict(
            n=len(d), n_soguk=int(sg.sum()),
            ezber_orani=round(float(ez[sg].mean()), 4),
            rmsle_w=round(P.rmsle(d.y.values, d.p.values, w), 5),
            adaylar=rb,
        )

    # tablo
    print("=" * 108)
    print("YON 4 -- SOGUK KOHORT ADAY TARAMASI  (rho, birim-rms normlu, kohort agirlikli, kuresel)")
    print("GEREKEN rho >= %.5f  (kabul 0.11436)" % GEREKEN)
    for b in BLOKLAR:
        print("  %s: n=%d soguk=%d (%.1f%%)  ezberlenebilir soguk orani=%.4f  rmsle_w=%.5f"
              % (b, R[b]["n"], R[b]["n_soguk"], 100 * R[b]["n_soguk"] / R[b]["n"],
                 R[b]["ezber_orani"], R[b]["rmsle_w"]))
    print("=" * 108)
    hd = "%-26s | %-22s | %-22s | %-22s" % ("YON", "yaz25 rho(+-SE,t)", "guz25", "kis26")
    print(hd)
    print("-" * 108)
    for ad in ad_listesi:
        huc = []
        for b in BLOKLAR:
            c = R[b]["adaylar"][ad]
            huc.append("%+.4f+-%.4f t%+.1f" % (c["rho"], c["se"], c["t"]))
        print("%-26s | %-22s | %-22s | %-22s" % (ad, huc[0], huc[1], huc[2]))
    print("=" * 108)
    print("TEMIZ / KIRLI AYRIM (p30 doktrini: yaz25/guz25 soguk olcumleri EZBER kirletir)")
    print("%-26s | %-28s | %-28s | %-20s" % ("YON", "yaz25 temiz/kirli", "guz25 temiz/kirli", "kis26 (%100 temiz)"))
    print("-" * 108)
    for ad in ad_listesi:
        s = []
        for b in BLOKLAR:
            c = R[b]["adaylar"][ad]
            s.append("%+.4f / %+.4f" % (c["rho_temiz"], c["rho_kirli"]))
        print("%-26s | %-28s | %-28s | %-20s" % (ad, s[0], s[1], s[2].split("/")[0].strip()))

    with open(os.path.join(SP, "s1_tarama.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1)
    print("\nyazildi: s1_tarama.json")


if __name__ == "__main__":
    main()
