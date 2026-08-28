"""SIKILASTIRILMIS kurulum: pencere cakismasi ve ID ezberi TAMAMEN kaldirilmis.

Koordinatorun iki uyarisi:
  (1) Kesimlerin HEDEF PENCERELERI cakisiyor -> ayni satirlar hem egitimde hem testte.
  (2) `tanim` ezber kanali -> ayni trafo iki kesimde de etiketli.

COZUM (ikisini de tek hamlede kapatir): kesimleri TAM 4 AY ayir.
  kesim A = X, hedef (X, X+4ay];  kesim B = X+4ay, hedef (X+4ay, X+8ay]
  -> pencereler KESISMEZ (satir cakismasi 0).
  -> B'de SOGUK olan trafo, tanimi geregi X+4ay'dan SONRA ilk kez gorunur; dolayisiyla
     A'nin ne gecmisinde ne hedef penceresinde bulunabilir. Trafo ortusmesi = 0 (kanitlanir).
Ayrica ozellik AILELERININ tek basina AUC'si ayri ayri raporlanir (ozellikle VARLIK DESENI).
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
from m7_soguk_olu import auc, ozellik_tablosu, oznitelikler, pr_auc
from m8_durust import clf, rgr

SONUC = {}

# --- kesin ayrik cift: hedef pencereler kesismiyor
CIFTLER = [("2025-06-30", "2025-10-31"), ("2025-07-31", "2025-11-30")]

AILE = {
    "VARLIK_DESENI": [
        "n",
        "ilk_gun",
        "son_gun",
        "kuyruk",
        "gun_araligi",
        "yogunluk",
        "eksik_gun",
        "kesintisiz",
        "pencere_payi",
        "maks_bosluk",
        "bosluk_say",
        "bosluk_std",
        "farkli_haftagunu",
        "haftasonu_orani",
    ],
    "DALGA_KOHORT": [
        "dalga_boyu",
        "dalga_payi",
        "dalga_ilce",
        "dalga_guc",
        "soguk_id_yakin5",
        "soguk_id_yakin50",
        "ayni_ilce_guc_soguk",
        "ilce_soguk_boy",
        "ilce_soguk_orani",
    ],
    "MEKAN": [
        "kod_ilce_olu",
        "kod_ilce_lvl",
        "kod_bolge_olu",
        "kod_bolge_lvl",
        "kod_il_olu",
        "kod_il_lvl",
        "yeni_ilce_olu",
        "yeni_ilce_lvl",
        "yeni_bolge_olu",
        "yeni_bolge_lvl",
        "yeni_il_olu",
        "yeni_il_lvl",
        "ilce_boy",
    ],
    "GUC": ["log_guc", "guc_sik", "kod_guc_olu", "kod_guc_lvl", "yeni_guc_olu", "yeni_guc_lvl"],
    "KIMLIK_ID": [
        "id_sayisal",
        "id_deger",
        "id_basamak",
        "id_blok3",
        "id_blok4",
        "id_blok5",
        "komsu3_olu",
        "komsu3_lvl",
        "komsu10_olu",
        "komsu10_lvl",
        "komsu25_olu",
        "komsu25_lvl",
        "kod_blok4_olu",
        "kod_blok4_lvl",
        "kod_blok5_olu",
        "kod_blok5_lvl",
    ],
}


def auc_gs(y, p, B=2000, seed=0):
    """AUC + onyukleme (bootstrap) %95 guven araligi."""
    rng = np.random.default_rng(seed)
    a = auc(y, p)
    n = len(y)
    v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        v.append(auc(y[i], p[i]))
    return float(a), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    tr = yukle()
    tab, gecl, hedl = {}, {}, {}
    kesimler = sorted({k for c in CIFTLER for k in c})
    print("=" * 86)
    print("0) SIKI KURULUM DOGRULAMASI (cakisma gercekten sifir mi?)")
    for k in kesimler:
        gec, hed, f = ozellik_tablosu(tr, k)
        tab[k], gecl[k], hedl[k] = f, gec, hed
    dog = {}
    for a, b in CIFTLER:
        A, B = tab[a], tab[b]
        # satir cakismasi: A'nin hedef penceresi ile B'nin hedef penceresi
        sa = pd.MultiIndex.from_frame(hedl[a][["tanim", "tarih"]])
        sb = pd.MultiIndex.from_frame(hedl[b][["tanim", "tarih"]])
        kes_satir = len(sa.intersection(sb))
        ort_trafo = len(set(A.index) & set(B.index))
        # B'nin soguklari A'nin gecmisinde var mi?
        b_in_a_gec = len(set(B.index) & set(gecl[a].tanim.unique()))
        b_in_a_hed = len(set(B.index) & set(hedl[a].tanim.unique()))
        dog[f"{a}->{b}"] = dict(
            hedef_satir_kesisimi=int(kes_satir),
            soguk_trafo_ortusmesi=int(ort_trafo),
            b_soguk_a_gecmisinde=int(b_in_a_gec),
            b_soguk_a_hedefinde=int(b_in_a_hed),
            a_n=int(len(A)),
            a_olu=int(A.olu.sum()),
            b_n=int(len(B)),
            b_olu=int(B.olu.sum()),
            a_pencere=f"{(pd.Timestamp(a) + pd.Timedelta(days=1)).date()}..{(pd.Timestamp(a) + pd.DateOffset(months=4)).date()}",
            b_pencere=f"{(pd.Timestamp(b) + pd.Timedelta(days=1)).date()}..{(pd.Timestamp(b) + pd.DateOffset(months=4)).date()}",
        )
        d = dog[f"{a}->{b}"]
        print(f"  {a} [{d['a_pencere']}] -> {b} [{d['b_pencere']}]")
        print(
            f"     hedef-satir kesisimi {kes_satir} | soguk-trafo ortusmesi {ort_trafo} | "
            f"B'nin soguklari A'nin gecmisinde {b_in_a_gec} / A'nin hedefinde {b_in_a_hed}"
        )
        print(
            f"     egitim: n={d['a_n']} olu={d['a_olu']} (%{100 * d['a_olu'] / d['a_n']:.1f}) | "
            f"test: n={d['b_n']} olu={d['b_olu']} (%{100 * d['b_olu'] / d['b_n']:.1f})"
        )
    SONUC["siki_kurulum_dogrulama"] = dog

    ozn = oznitelikler(tab[kesimler[0]])
    ozn = [c for c in ozn if all(c in tab[k].columns for k in kesimler)]

    print("\n" + "=" * 86)
    print("1) OZELLIK AILELERI -- TEK BASINA AUC (siki kurulum, egit A -> test B)")
    aile_sonuc = {}
    for a, b in CIFTLER:
        ftr, fte = tab[a], tab[b]
        y = fte.olu.values
        print(
            f"\n  --- egit {a} -> test {b} (test n={len(fte)}, olu={int(y.sum())}, "
            f"taban PR {y.mean():.3f}) ---"
        )
        # bos dagilim: etiket karistirilmis egitimle
        rng = np.random.default_rng(7)
        bos = []
        for s in range(25):
            yk = rng.permutation(ftr.olu.values)
            p, _ = clf(ftr[ozn].astype(float).values, yk, fte[ozn].astype(float).values, seed=s)
            bos.append(auc(y, p))
        bm, bs = float(np.mean(bos)), float(np.std(bos))
        print(f"    [BOS DAGILIM: karistirilmis etiketle AUC {bm:.3f} +- {bs:.3f}]")
        rec = {"bos_ort": bm, "bos_std": bs}
        for ad, kols in list(AILE.items()) + [
            ("HEPSI", ozn),
            ("HEPSI_kimliksiz", [c for c in ozn if c not in AILE["KIMLIK_ID"]]),
        ]:
            kols = [c for c in kols if c in ozn]
            ps = [
                clf(
                    ftr[kols].astype(float).values,
                    ftr.olu.values,
                    fte[kols].astype(float).values,
                    seed=s,
                )[0]
                for s in range(5)
            ]
            p = np.mean(ps, 0)
            A_, lo, hi = auc_gs(y, p)
            pr = pr_auc(y, p)
            z = (A_ - bm) / (bs + 1e-9)
            rec[ad] = dict(
                k=len(kols), auc=A_, auc_alt=lo, auc_ust=hi, prauc=pr, taban_pr=float(y.mean()), z=z
            )
            print(
                f"    {ad:18s} ({len(kols):2d} oz) AUC {A_:.3f} [%95 GA {lo:.3f}-{hi:.3f}] "
                f"PR-AUC {pr:.3f} (taban {y.mean():.3f})  z={z:+.2f}"
                f"{'   <-- BOS DISI' if abs(z) > 2 else ''}"
            )
        aile_sonuc[f"{a}->{b}"] = rec
    SONUC["aile_auc_siki"] = aile_sonuc

    print("\n" + "=" * 86)
    print("2) TEKIL OZELLIKLER: siki kurulumda IKI CIFTTE DE tutan var mi?")
    t1 = {c: auc(tab[CIFTLER[0][1]].olu.values, tab[CIFTLER[0][1]][c].values) for c in ozn}
    t2 = {c: auc(tab[CIFTLER[1][1]].olu.values, tab[CIFTLER[1][1]][c].values) for c in ozn}
    ort = {c: ((t1[c] - 0.5) + (t2[c] - 0.5)) / 2 for c in ozn if t1[c] == t1[c] and t2[c] == t2[c]}
    sir = sorted(ort.items(), key=lambda kv: -abs(kv[1]))
    print(f"  {'ozellik':24s} {'10-31':>7s} {'11-30':>7s}  tutarli?")
    tut = []
    for c, _ in sir[:16]:
        ok = (t1[c] - 0.5) * (t2[c] - 0.5) > 0 and min(abs(t1[c] - 0.5), abs(t2[c] - 0.5)) > 0.05
        if ok:
            tut.append(c)
        print(f"  {c:24s} {t1[c]:7.3f} {t2[c]:7.3f}  {'EVET' if ok else '-'}")
    SONUC["tekil_siki"] = dict(auc_1031=t1, auc_1130=t2, tutarli=tut)
    print(
        f"\n  Iki kesimde de ayni yonde ve |AUC-0.5|>0.05 olan ozellikler: {tut if tut else 'YOK'}"
    )

    # ILCE olu-orani sinyalinin ayri sinamasi (guven araligi ile)
    print("\n" + "=" * 86)
    print("3) TEK GERCEK ADAY: ILCE olu-orani (gecmisten) -- guven araligi ile")
    il = {}
    for a, b in CIFTLER:
        fte = tab[b]
        for c in ["kod_ilce_olu", "yeni_ilce_olu", "ilce_soguk_orani"]:
            A_, lo, hi = auc_gs(fte.olu.values, fte[c].values.astype(float), seed=3)
            il.setdefault(f"{a}->{b}", {})[c] = dict(auc=A_, alt=lo, ust=hi)
            print(
                f"  {b}  {c:20s} AUC {A_:.3f} [%95 GA {lo:.3f}-{hi:.3f}] "
                f"{'ANLAMLI (GA 0.5 icermiyor)' if lo > 0.5 or hi < 0.5 else 'ANLAMSIZ (GA 0.5 iceriyor)'}"
            )
    SONUC["ilce_sinyali"] = il

    print("\n" + "=" * 86)
    print("4) SIKI KURULUMDA SATIR-RMSLE (soguk satirlar; taban vs model)")
    sat = {}
    for a, b in CIFTLER:
        ftr, fte, hed, gec = tab[a], tab[b], hedl[b], gecl[b]
        gec = gec.assign(ly=lambda d: np.log1p(d.tuketim))
        gucm = gec.groupby("guc").ly.mean()
        g0 = float(gec.ly.mean())
        h = hed[hed.soguk.values]
        hly = np.log1p(h.tuketim.values)
        taban = h.guc.map(gucm).fillna(g0).values
        R = lambda v: float(np.sqrt(np.mean((v - hly) ** 2)))
        pr = np.mean(
            [
                rgr(
                    ftr[ozn].astype(float).values,
                    ftr.y.values,
                    ftr.n.values.astype(float),
                    fte[ozn].astype(float).values,
                    s,
                )[0]
                for s in range(5)
            ],
            0,
        )
        pc = np.mean(
            [
                clf(
                    ftr[ozn].astype(float).values, ftr.olu.values, fte[ozn].astype(float).values, s
                )[0]
                for s in range(5)
            ],
            0,
        )
        fad = ftr[ftr.olu == 0]
        pdd = np.mean(
            [
                rgr(
                    fad[ozn].astype(float).values,
                    fad.y.values,
                    fad.n.values.astype(float),
                    fte[ozn].astype(float).values,
                    s,
                )[0]
                for s in range(5)
            ],
            0,
        )
        S = pd.DataFrame({"reg": pr, "p": pc, "diri": pdd}, index=fte.index)
        r_tab, r_reg = R(taban), R(h.tanim.map(S.reg).values)
        r_har = R((1 - h.tanim.map(S.p).values) * h.tanim.map(S.diri).values)
        r_tav = R(np.where(h.tanim.map(fte.olu).values == 1, 0.0, h.tanim.map(S.reg).values))
        egri = []
        for t in [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
            sf = S.p > t
            pred = np.where(h.tanim.map(sf).values, 0.0, h.tanim.map(S.reg).values)
            tf = fte.loc[sf[sf].index]
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
        sat[f"{a}->{b}"] = dict(
            soguk_satir=int(len(h)),
            taban=r_tab,
            regresyon=r_reg,
            harman=r_har,
            tavan=r_tav,
            esik_egrisi=egri,
        )
        print(
            f"\n  --- {a} -> {b} | soguk {len(h):,} satir / {len(fte)} trafo / {int(fte.olu.sum())} olu ---"
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
    SONUC["satir_rmsle_siki"] = sat

    with open(os.path.join(BURA, "m10_sikilastir.json"), "w", encoding="utf-8") as fh:
        json.dump(SONUC, fh, ensure_ascii=False, indent=1, default=float)
    print("\nyazildi: m10_sikilastir.json")


if __name__ == "__main__":
    main()
