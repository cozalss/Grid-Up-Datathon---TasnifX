"""SOGUK-SIFIR SINIFLANDIRICISI -- blok-disi durust sinav.

Bir blokta soguk olan trafo, testteki soguk trafonun tam analogudur:
gecmisi yok, yalnizca statik oznitelikleri var. Bu yuzden egitim kumesi
uc blogun soguk trafolari, sinav ise BIR BLOGU TAMAMEN DISARIDA BIRAKMAK.

Iki yol yan yana olculur:
  A  sifir-orani regresyonu  ->  p_yeni = p * (1 - lam*q)
  B  dogrudan artik regresyonu ->  p_yeni = p + lam*rhat

Olcut GERCEK: tutulan blogun soguk satirlarinda MSE dustu mu?
lam yalnizca EGITIM bloklarinda secilir, tutulan blokta degil.

PLASEBO: trafo etiketleri karistirilir; kazanc 0'a inmeli.
"""

import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
BURA = os.path.dirname(os.path.abspath(__file__))
BLOKLAR = ["yaz25", "guz25", "kis26"]
import lightgbm as lgb  # noqa: E402

# testteki soguk satirlarda GERCEKTEN dolu olan oznitelikler
OZN = [
    "guc",
    "yas",
    "ilk_gun_mu",
    "p_gun_sayisi",
    "p_doluluk",
    "p_ilk_ofset",
    "p_son_ofset",
    "p_yayilma",
    "p_pencere_payi",
    "agac_orani",
    "calilik_orani",
    "otlak_orani",
    "tarim_orani",
    "yerlesim_orani",
    "ciplak_orani",
    "su_orani",
    "bitki_ortusu_orani",
    "osm_trafo",
    "osm_direk",
    "osm_dagitim_hat_km",
    "osm_iletim_hat_km",
    "osm_kablo_km",
    "osm_direk_yogunlugu",
    "osm_hat_yogunlugu",
    "ilce_trafo_sayisi",
    "ilce_toplam_guc",
    "ilce_guc_medyan",
    "nufus",
    "alan_km2",
    "ilce_nufus_yogunlugu",
    "trafo_basina_nufus",
    "kva_basina_nufus",
    "guc_yuzdelik",
    "guc_payi",
    "guc_medyan_orani",
    "trafo_basina_hat",
    "g_guc_kova",
    "g_kova_log_ort",
    "g_ilce_log_ort",
    "g_ilce_kova_ort",
    "g_ilce_kova_n",
    "tanim_uzunluk",
    "tanim_on2",
    "tanim_on3",
    "tanim_on4",
    "tanim_on5",
    "tanim_num",
    "il_key",
    "ilce_key",
]

print("veri yukleniyor...", flush=True)
e = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
parca = []
for bad in BLOKLAR:
    blk = e[e._blok == bad]
    sog = blk[blk.soguk_mu == 1]
    z = np.load(os.path.join(DN, f"soguk_tahmin_{bad}.npz"))
    d = e.loc[sog.index.values].copy()
    d["_p"] = np.mean([z[k] for k in z.files], axis=0)
    d["_ly"] = np.log1p(d.tuketim.values.astype(np.float64))
    d["_r"] = d._ly - d._p
    d["_sifir"] = (d.tuketim.values == 0).astype(np.float64)
    parca.append(d)
C = pd.concat(parca, ignore_index=True)
kul = [c for c in OZN if c in C.columns and pd.api.types.is_numeric_dtype(C[c])]
print(
    f"soguk satir {len(C):,}  trafo-blok {C.groupby(['tanim', '_blok']).ngroups:,}  "
    f"oznitelik {len(kul)}",
    flush=True,
)
for bad in BLOKLAR:
    s = C[C._blok == bad]
    tf = s.groupby("tanim")._sifir.mean()
    print(
        f"  {bad}: {len(s):>7,} satir {len(tf):>5,} trafo  "
        f"tamamen-sifir {int((tf > 0.999).sum()):>4,} (%{(tf > 0.999).mean() * 100:.1f})  "
        f"MSE={float((s._r**2).mean()):.4f}"
    )

PAR_R = dict(
    objective="regression",
    metric="l2",
    learning_rate=0.04,
    num_leaves=31,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    verbose=-1,
    num_threads=8,
    seed=1000,
)
LAMS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]


def egit(tr, hedef):
    return lgb.train(PAR_R, lgb.Dataset(tr[kul], label=tr[hedef]), num_boost_round=500)


sonuc = {}
for tut in BLOKLAR:
    tr = C[C._blok != tut]
    va = C[C._blok == tut]
    mse0 = float((va._r**2).mean())
    print(
        f"\n{'=' * 74}\n{tut} TUTULDU  (egitim {len(tr):,} satir, sinav {len(va):,} satir, "
        f"mevcut MSE={mse0:.4f})\n{'=' * 74}",
        flush=True,
    )

    mA = egit(tr, "_sifir")
    qa_tr = np.clip(mA.predict(tr[kul]), 0, 1)
    qa_va = np.clip(mA.predict(va[kul]), 0, 1)
    mB = egit(tr, "_r")
    rb_tr = mB.predict(tr[kul])
    rb_va = mB.predict(va[kul])

    def tara(p_tr, p_va, uygula):
        en = min(LAMS, key=lambda L: float(((tr._ly - uygula(tr._p.values, p_tr, L)) ** 2).mean()))
        m = float(((va._ly - uygula(va._p.values, p_va, en)) ** 2).mean())
        return en, m

    lamA, mseA = tara(qa_tr, qa_va, lambda p, q, L: p * (1 - L * q))
    lamB, mseB = tara(rb_tr, rb_va, lambda p, r, L: p + L * r)
    print(f"  A sifir-orani buzmesi : lam*={lamA:.2f}  MSE={mseA:.4f}  kazanc={mse0 - mseA:+.4f}")
    print(f"  B dogrudan artik      : lam*={lamB:.2f}  MSE={mseB:.4f}  kazanc={mse0 - mseB:+.4f}")

    tf = va.groupby("tanim")._sifir.mean()
    gercek = va.tanim.map(tf > 0.999).to_numpy()
    if gercek.any() and not gercek.all():
        sira = np.argsort(-qa_va)
        n1 = int(gercek.sum())
        print(
            f"  A siralama: gercek tamamen-sifir {n1:,} satir; en yuksek q'lu "
            f"{n1:,} satirin %{gercek[sira[:n1]].mean() * 100:.1f}'i dogru"
        )
    sonuc[tut] = dict(
        mse0=mse0,
        mseA=mseA,
        mseB=mseB,
        lamA=lamA,
        lamB=lamB,
        kazancA=mse0 - mseA,
        kazancB=mse0 - mseB,
    )

print(f"\n{'=' * 74}\nOZET -- tutulan blokta soguk-ici MSE kazanci\n{'=' * 74}")
print(f"{'blok':>8s} {'mevcut':>9s} {'A kazanc':>10s} {'B kazanc':>10s} {'A test-MSE etkisi':>18s}")
for b, s in sonuc.items():
    print(
        f"{b:>8s} {s['mse0']:9.4f} {s['kazancA']:+10.4f} {s['kazancB']:+10.4f} "
        f"{0.222 * s['kazancA']:+18.5f}"
    )
ort = np.mean([s["kazancA"] for s in sonuc.values()])
ortB = np.mean([s["kazancB"] for s in sonuc.values()])
print(f"\nORTALAMA soguk-ici kazanc: A={ort:+.4f}  B={ortB:+.4f}")
print(
    f"TEST BILESIMINDE (soguk %22.2) beklenen MSE kazanci: "
    f"A={0.222 * ort:+.5f}  B={0.222 * ortB:+.5f}"
)
print("2. sira icin gereken 0.002256   1. sira icin 0.020778")
with open(os.path.join(BURA, "zi_sonuc.json"), "w") as fh:
    json.dump(sonuc, fh, indent=1)
