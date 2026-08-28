"""GUC TESTI: kesimler-arasi sinyal yoksa, HIC MI yok?

m8 kesimler-arasi AYRIK kumede AUC ~0.5 (hatta alti) buldu; ama o kumeler kucuk
(n=397..871, olu=16..37) ve gurultu std'si +-0.08. Bu yuzden 'sinyal yok' hukmunu
DAHA YUKSEK GUCLU bir testle dogruluyoruz:
  (1) TEK kesim icinde, ILCE-gruplu 5-kat CV (trafo ortusmesi yok, mekan ezberi yok).
      Bu kurulum kesimler-arasindan IYIMSER; yine de 0.5 cikarsa kapi kesin kapali.
  (2) Havuz: tum kesimlerin soguk trafolari, trafo bazinda TEKILLESTIRILMIS,
      ILCE-gruplu CV. En buyuk ornek.
  (3) Betimleyici: OLU vs DIRI soguk trafolarin profil farki (etkiyi gozle gor).
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
from m7_soguk_olu import KESIMLER, auc, ozellik_tablosu, oznitelikler, pr_auc
from m8_durust import KIMLIK, clf
from sklearn.model_selection import GroupKFold

SONUC = {}


def gruplu_cv(f, ozn, grup, k=5, seed_say=5):
    g = GroupKFold(n_splits=k)
    p = np.zeros(len(f))
    X = f[ozn].astype(float).values
    y = f.olu.values
    for tr_i, te_i in g.split(X, y, groups=grup):
        ps = [clf(X[tr_i], y[tr_i], X[te_i], s)[0] for s in range(seed_say)]
        p[te_i] = np.mean(ps, 0)
    return p


def main():
    tr = yukle()
    tab = {}
    for k in KESIMLER:
        gec, hed, f = ozellik_tablosu(tr, k)
        tab[k] = f
    ozn = oznitelikler(tab[KESIMLER[0]])
    ozn = [c for c in ozn if all(c in tab[k].columns for k in KESIMLER)]
    ozn_t = [c for c in ozn if c not in KIMLIK]

    print("=" * 80)
    print("1) TEK KESIM ICINDE, ILCE-GRUPLU 5-KAT CV  (iyimser ust sinir)")
    R = {}
    for k in KESIMLER:
        f = tab[k]
        for ad, oz in [("tam", ozn), ("kimliksiz", ozn_t)]:
            p = gruplu_cv(f, oz, f.ilce.fillna("YOK").astype(str).values)
            R.setdefault(k, {})[ad] = dict(
                auc=auc(f.olu.values, p),
                prauc=pr_auc(f.olu.values, p),
                taban=float(f.olu.mean()),
                n=int(len(f)),
                olu=int(f.olu.sum()),
            )
        # bos kontrol (etiket karistir)
        rng = np.random.default_rng(1)
        bos = []
        for s in range(10):
            fk = f.copy()
            fk["olu"] = rng.permutation(f.olu.values)
            pb = gruplu_cv(fk, ozn_t, f.ilce.fillna("YOK").astype(str).values, seed_say=1)
            bos.append(auc(fk.olu.values, pb))
        R[k]["bos_auc_ort"] = float(np.mean(bos))
        R[k]["bos_auc_std"] = float(np.std(bos))
        z = (R[k]["kimliksiz"]["auc"] - np.mean(bos)) / (np.std(bos) + 1e-9)
        R[k]["z"] = float(z)
        print(
            f"  {k} n={R[k]['tam']['n']:4d} olu={R[k]['tam']['olu']:3d} | "
            f"AUC tam {R[k]['tam']['auc']:.3f} kimliksiz {R[k]['kimliksiz']['auc']:.3f} | "
            f"PR-AUC {R[k]['kimliksiz']['prauc']:.3f} (taban {R[k]['kimliksiz']['taban']:.3f}) | "
            f"BOS AUC {np.mean(bos):.3f}+-{np.std(bos):.3f}  z={z:+.2f}"
        )
    SONUC["tek_kesim_gruplu_cv"] = R

    print("\n" + "=" * 80)
    print("2) HAVUZ (tum kesimler, trafo bazinda TEKILLESTIRILMIS), ILCE-GRUPLU 5-KAT CV")
    h = pd.concat([tab[k].assign(_k=k) for k in KESIMLER])
    h = h[~h.index.duplicated(keep="last")]  # her trafo bir kez (en gec kesim)
    p = gruplu_cv(h, ozn_t, h.ilce.fillna("YOK").astype(str).values)
    rng = np.random.default_rng(2)
    bos = []
    for s in range(10):
        hk = h.copy()
        hk["olu"] = rng.permutation(h.olu.values)
        bos.append(
            auc(
                hk.olu.values,
                gruplu_cv(hk, ozn_t, h.ilce.fillna("YOK").astype(str).values, seed_say=1),
            )
        )
    z = (auc(h.olu.values, p) - np.mean(bos)) / (np.std(bos) + 1e-9)
    SONUC["havuz_gruplu_cv"] = dict(
        n=int(len(h)),
        olu=int(h.olu.sum()),
        auc=auc(h.olu.values, p),
        prauc=pr_auc(h.olu.values, p),
        taban=float(h.olu.mean()),
        bos_ort=float(np.mean(bos)),
        bos_std=float(np.std(bos)),
        z=float(z),
    )
    v = SONUC["havuz_gruplu_cv"]
    print(
        f"  n={v['n']} olu={v['olu']} (%{100 * v['taban']:.1f}) | AUC {v['auc']:.3f} "
        f"PR-AUC {v['prauc']:.3f} (taban {v['taban']:.3f}) | BOS {v['bos_ort']:.3f}+-{v['bos_std']:.3f} z={z:+.2f}"
    )
    # kalibrasyon
    q = pd.qcut(pd.Series(p), 10, duplicates="drop", labels=False).values
    kal = [
        dict(
            bin=int(b),
            n=int((q == b).sum()),
            p=float(p[q == b].mean()),
            gercek=float(h.olu.values[q == b].mean()),
        )
        for b in sorted(set(q))
    ]
    SONUC["havuz_kalibrasyon"] = kal
    print("  kalibrasyon (10 dilim): " + " ".join(f"{d['p']:.3f}/{d['gercek']:.3f}" for d in kal))
    print(
        f"  en yuksek riskli %10 dilimde gercek olu orani {kal[-1]['gercek']:.3f} "
        f"(genel {v['taban']:.3f}) -> kaldirac {kal[-1]['gercek'] / v['taban']:.2f}x"
    )

    print("\n" + "=" * 80)
    print("3) BETIMLEYICI: OLU vs DIRI soguk trafo profili (havuz, n=%d)" % len(h))
    prof = {}
    for c in [
        "log_guc",
        "guc",
        "n",
        "ilk_gun",
        "son_gun",
        "kuyruk",
        "gun_araligi",
        "yogunluk",
        "eksik_gun",
        "kesintisiz",
        "maks_bosluk",
        "bosluk_say",
        "farkli_haftagunu",
        "haftasonu_orani",
        "dalga_boyu",
        "dalga_ilce",
        "soguk_id_yakin5",
        "kod_ilce_olu",
        "kod_guc_olu",
        "yeni_ilce_olu",
        "ilce_soguk_orani",
        "guc_sik",
    ]:
        if c not in h.columns:
            continue
        a = h.loc[h.olu == 1, c].astype(float)
        b = h.loc[h.olu == 0, c].astype(float)
        sd = np.sqrt((a.var() * len(a) + b.var() * len(b)) / (len(a) + len(b)))
        d = (a.mean() - b.mean()) / (sd + 1e-12)
        prof[c] = dict(olu_ort=float(a.mean()), diri_ort=float(b.mean()), cohen_d=float(d))
    for c, v2 in sorted(prof.items(), key=lambda kv: -abs(kv[1]["cohen_d"]))[:14]:
        print(
            f"  {c:22s} olu {v2['olu_ort']:10.3f} | diri {v2['diri_ort']:10.3f} | d {v2['cohen_d']:+.3f}"
        )
    SONUC["profil"] = prof

    # ilce yogunlasmasi
    ic = h.groupby("ilce").olu.agg(["size", "sum"])
    ic = ic[ic["size"] >= 5].sort_values("sum", ascending=False)
    top = ic.head(8)
    print(f"\n  ilce yogunlasmasi (>=5 soguk trafolu ilceler, n={len(ic)}):")
    for i, r in top.iterrows():
        print(
            f"    {i[:30]:30s} soguk {int(r['size']):3d} olu {int(r['sum']):3d} (%{100 * r['sum'] / r['size']:.0f})"
        )
    pay = float(top["sum"].sum() / h.olu.sum())
    print(
        f"  en olu-yogun 8 ilce tum olulerin %{100 * pay:.0f}'ini tasiyor "
        f"(bu ilceler soguklarin %{100 * top['size'].sum() / len(h):.0f}'i)"
    )
    SONUC["ilce_yogunlasma"] = dict(
        en_ust8_olu_payi=pay,
        en_ust8_soguk_payi=float(top["size"].sum() / len(h)),
        ilce_sayisi=int(len(ic)),
    )
    # ilce olu oraninin kesimler arasi KARARLILIGI (bu tasinabilir mi?)
    a = tab["2025-08-31"].groupby("ilce").olu.agg(["size", "mean"])
    b = tab["2025-11-30"].groupby("ilce").olu.agg(["size", "mean"])
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    j = j[(j["size_a"] >= 5) & (j["size_b"] >= 5)]
    r = float(np.corrcoef(j["mean_a"], j["mean_b"])[0, 1]) if len(j) > 3 else float("nan")
    print(
        f"\n  ilce OLU ORANI kararliligi (08-31 vs 11-30, >=5 trafolu {len(j)} ilce): r = {r:+.3f}"
    )
    SONUC["ilce_kararlilik"] = dict(n_ilce=int(len(j)), r=r)

    with open(os.path.join(BURA, "m9_guc_testi.json"), "w", encoding="utf-8") as fh:
        json.dump(SONUC, fh, ensure_ascii=False, indent=1, default=float)
    print("\nyazildi: m9_guc_testi.json")


if __name__ == "__main__":
    main()
