"""Ayirt edemesek bile: OLU ORANINI biliyoruz. BUZME (shrinkage) ne kazandirir?

Soguk trafolarin ~%5-8'i olu. Kim oldugunu bilemiyoruz (m8/m10: AUC ~0.5).
Ama RMSLE'yi enkucuklten NOKTA tahmini kosullu ORTALAMA'dir:
    E[ly] = (1 - p_olu) * E[ly | diri]
Taban (guc grubu ortalamasi) DIRI gecmis trafolardan hesaplandigi icin SISTEMATIK
YUKSEK. Tek bir global buzme carpani k bile kazanc verebilir -- SIFIR ayirt edicilikle.
Burada k'yi EGITIM kesiminde ogrenip TEST kesiminde uyguluyoruz (siki, cakismasiz ciftler).
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
from m1_geriteste import yukle
from m7_soguk_olu import ozellik_tablosu

CIFTLER = [("2025-06-30", "2025-10-31"), ("2025-07-31", "2025-11-30"), ("2025-08-31", "2025-11-30")]
SONUC = {}


def soguk_taban(gec, hed, kesim):
    """Taban soguk tahmini: gecmisteki guc grubu log-ortalamasi."""
    g = gec.assign(ly=np.log1p(gec.tuketim))
    gucm = g.groupby("guc").ly.mean()
    g0 = float(g.ly.mean())
    return hed.guc.map(gucm).fillna(g0).values, gucm, g0


def main():
    tr = yukle()
    tab, gecl, hedl = {}, {}, {}
    for k in sorted({x for c in CIFTLER for x in c}):
        gec, hed, f = ozellik_tablosu(tr, k)
        tab[k], gecl[k], hedl[k] = f, gec, hed

    print("=" * 84)
    print("1) EN IYI GLOBAL BUZME CARPANI k  (soguk tahmin = k * guc-grubu-ortalamasi)")
    print("   -- k egitim kesiminde ogrenilir, test kesiminde UYGULANIR --")
    R = {}
    for a, b in CIFTLER:
        rec = {}
        # egitim kesiminde optimal k
        gec_a, hed_a = gecl[a], hedl[a]
        ha = hed_a[hed_a.soguk.values]
        pa, _, _ = soguk_taban(gec_a, ha, a)
        ya = np.log1p(ha.tuketim.values)
        k_egit = float(np.dot(pa, ya) / np.dot(pa, pa))  # en kucuk kareler carpani
        # test kesiminde uygula
        gec_b, hed_b = gecl[b], hedl[b]
        hb = hed_b[hed_b.soguk.values]
        pb, _, _ = soguk_taban(gec_b, hb, b)
        yb = np.log1p(hb.tuketim.values)
        Rf = lambda v: float(np.sqrt(np.mean((v - yb) ** 2)))
        k_test = float(np.dot(pb, yb) / np.dot(pb, pb))  # test'in KENDI optimumu (tavan)
        r0, r1, rstar = Rf(pb), Rf(k_egit * pb), Rf(k_test * pb)
        # olu orani ile teorik buzme
        p_olu_a = float(ha.tanim.map(tab[a].olu).mean())
        r_teor = Rf((1 - p_olu_a) * pb)
        rec = dict(
            k_egit=k_egit,
            k_test_optimum=k_test,
            olu_orani_egit=p_olu_a,
            rmsle_buzmesiz=r0,
            rmsle_k_egit=r1,
            rmsle_k_test_opt=rstar,
            rmsle_1eksi_p=r_teor,
            kazanc_k_egit=r0 - r1,
            kazanc_tavan=r0 - rstar,
            soguk_satir=int(len(hb)),
        )
        R[f"{a}->{b}"] = rec
        print(f"\n  {a} -> {b}  (test soguk {len(hb):,} satir)")
        print(f"    egitimde ogrenilen k = {k_egit:.4f} | testin kendi optimumu k* = {k_test:.4f}")
        print(f"    buzmesiz (k=1)          RMSLE {r0:.4f}")
        print(f"    k_egit uygulanmis       RMSLE {r1:.4f}   kazanc {r0 - r1:+.4f}")
        print(
            f"    (1 - olu_orani) = {1 - p_olu_a:.4f}  RMSLE {r_teor:.4f}   kazanc {r0 - r_teor:+.4f}"
        )
        print(f"    test-optimum k* (TAVAN) RMSLE {rstar:.4f}   kazanc {r0 - rstar:+.4f}")
    SONUC["global_buzme"] = R

    print("\n" + "=" * 84)
    print("2) ILCE-KOSULLU BUZME (tek hayatta kalan sinyal: kod_ilce_olu)")
    print("   tahmin = (1 - p_ilce) * guc-grubu-ortalamasi ; p_ilce GECMISTEN")
    R2 = {}
    for a, b in CIFTLER:
        fb = tab[b]
        gec_b, hed_b = gecl[b], hedl[b]
        hb = hed_b[hed_b.soguk.values]
        pb, _, _ = soguk_taban(gec_b, hb, b)
        yb = np.log1p(hb.tuketim.values)
        Rf = lambda v: float(np.sqrt(np.mean((v - yb) ** 2)))
        # egitim kesiminden: kod_ilce_olu -> gercek olu orani egimi (kalibrasyon)
        fa = tab[a]
        x = fa.kod_ilce_olu.values.astype(float)
        yl = fa.olu.values.astype(float)
        A = np.vstack([np.ones_like(x), x]).T
        c = np.linalg.lstsq(A, yl, rcond=None)[0]
        p_il = np.clip(c[0] + c[1] * fb.kod_ilce_olu.values.astype(float), 0.0, 0.6)
        pm = hb.tanim.map(pd.Series(p_il, index=fb.index)).fillna(float(np.mean(p_il))).values
        k_egit = R[f"{a}->{b}"]["k_egit"]
        r_glob = Rf(k_egit * pb)
        r_il = Rf((1 - pm) * pb)
        # global k ile birlikte (ilce sapmasini global k etrafinda uygula)
        pm2 = pm / max(1e-9, float(np.mean(pm))) * (1 - k_egit)
        r_il2 = Rf(np.clip(1 - pm2, 0.05, 1.2) * pb)
        R2[f"{a}->{b}"] = dict(
            kalibrasyon_kesme=float(c[0]),
            kalibrasyon_egim=float(c[1]),
            p_ilce_ort=float(np.mean(p_il)),
            p_ilce_maks=float(np.max(p_il)),
            rmsle_global_k=r_glob,
            rmsle_ilce=r_il,
            rmsle_ilce_global_k=r_il2,
            ilce_ek_kazanc=r_glob - r_il2,
        )
        print(
            f"  {a} -> {b}: global-k RMSLE {r_glob:.4f} | ilce-kosullu {r_il:.4f} | "
            f"ilce(global-k olcekli) {r_il2:.4f}  -> ilcenin EK kazanci {r_glob - r_il2:+.4f}"
        )
        print(f"      (p_ilce ort {np.mean(p_il):.3f}, maks {np.max(p_il):.3f}, egim {c[1]:+.3f})")
    SONUC["ilce_buzme"] = R2

    print("\n" + "=" * 84)
    print("3) TAM HEDEF KUMESINDE ETKI (soguk+sicak, m2 tabanina karsi)")
    R3 = {}
    for a, b in CIFTLER:
        gec, hed = gecl[b], hedl[b]
        g = gec.assign(ly=np.log1p(gec.tuketim))
        g0 = float(g.ly.mean())
        tm28 = g[g.tarih > pd.Timestamp(b) - pd.Timedelta(days=28)].groupby("tanim").ly.mean()
        tmall = g.groupby("tanim").ly.mean()
        gucm = g.groupby("guc").ly.mean()
        hly = np.log1p(hed.tuketim.values)
        sicak = hed.tanim.map(tm28).fillna(hed.tanim.map(tmall)).fillna(g0).values
        sg = hed.soguk.values
        base_cold = hed.guc.map(gucm).fillna(g0).values
        k = R[f"{a}->{b}"]["k_egit"]
        r0 = float(np.sqrt(np.mean((np.where(sg, base_cold, sicak) - hly) ** 2)))
        r1 = float(np.sqrt(np.mean((np.where(sg, k * base_cold, sicak) - hly) ** 2)))
        R3[f"{a}->{b}"] = dict(taban_rmsle=r0, buzmeli_rmsle=r1, kazanc=r0 - r1, k=k)
        print(
            f"  {a} -> {b}: TAM RMSLE taban {r0:.4f} -> buzmeli {r1:.4f}   kazanc {r0 - r1:+.4f}  (k={k:.4f})"
        )
    SONUC["tam_hedef"] = R3

    print("\n" + "=" * 84)
    print("4) k'nin KESIMLER ARASI KARARLILIGI (gercek teste tasinabilir mi?)")
    ks = {}
    for kk in sorted(tab):
        gec, hed = gecl[kk], hedl[kk]
        h = hed[hed.soguk.values]
        p, _, _ = soguk_taban(gec, h, kk)
        y = np.log1p(h.tuketim.values)
        ks[kk] = dict(
            k=float(np.dot(p, y) / np.dot(p, p)),
            olu_orani=float(tab[kk].olu.mean()),
            olu_satir_orani=float(h.tanim.map(tab[kk].olu).mean()),
        )
        print(
            f"  {kk}: optimal k = {ks[kk]['k']:.4f} | olu trafo orani %{100 * ks[kk]['olu_orani']:.1f} "
            f"| olu SATIR orani %{100 * ks[kk]['olu_satir_orani']:.1f}"
        )
    v = [x["k"] for x in ks.values()]
    print(f"  -> k araligi {min(v):.4f}..{max(v):.4f}, ort {np.mean(v):.4f}, std {np.std(v):.4f}")
    SONUC["k_kararlilik"] = dict(
        kesimler=ks,
        ort=float(np.mean(v)),
        std=float(np.std(v)),
        alt=float(min(v)),
        ust=float(max(v)),
    )

    with open(os.path.join(BURA, "m11_buzme.json"), "w", encoding="utf-8") as fh:
        json.dump(SONUC, fh, ensure_ascii=False, indent=1, default=float)
    print("\nyazildi: m11_buzme.json")


if __name__ == "__main__":
    main()
